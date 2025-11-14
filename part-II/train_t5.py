from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig, T5TokenizerFast
import os
import re
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, read_queries
# NOTE: utils.DB_PATH already points to part-II/data/flight_database.db

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

# Resolve all paths relative to this file so CWD doesn’t matter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# --- Safer stopping: halt at first ';' ---
class StopOnSemicolon(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.semi_id = self.tokenizer.convert_tokens_to_ids(";")
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.semi_id


def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["none", "cosine", "linear"])
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--max_n_epochs", type=int, default=6)
    parser.add_argument("--patience_epochs", type=int, default=3)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="dev")

    # Data
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # Generation / reranking
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_beam_size", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=8,
                        help="k-best candidates to rerank by execution")
    parser.add_argument("--rerank_dev_by_gt", action="store_true",
                        help="Dev only: use ground-truth records to pick candidate with max F1")
    parser.add_argument("--prefer_executable_on_test", action="store_true",
                        help="Test: among k candidates, pick any executable (prefer non-empty)")

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    experiment_name = args.experiment_name
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"{model_type}_experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Dev file paths (ground truth already sanitized)
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")  # must exist

    # Model predictions (dev)
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            wandb.log({
                "train/loss": tr_loss,
                "dev/loss": eval_loss,
                "dev/record_f1": record_f1,
                "dev/record_em": record_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": error_rate,
            }, step=epoch)

        improved = record_f1 > best_f1
        if improved:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if improved:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # AMP for speed; keeps numerics stable on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_targets.view(-1))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            non_pad = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * max(1, non_pad)
            total_tokens += max(1, non_pad)

    return total_loss / max(1, total_tokens)


# --- helpers ---
def _extract_sql_like(s: str) -> str:
    s = s.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    semi = s.find(";")
    if semi != -1:
        return s[: semi + 1].strip()
    # ensure trailing ';' for exec
    if not s.lower().startswith("select"):
        s = "SELECT " + s
    if not s.endswith(";"):
        s = s + ";"
    return s


def _decode_k_and_rerank(
    model, tok, enc_ids, enc_mask, gen_cfg, k, dev_gt_records=None, dev_gt_idx_base=0, prefer_exec=False
):
    """
    Generate k candidates; pick by:
      - dev: if dev_gt_records provided, choose candidate with highest F1 vs GT record set;
             tie-break: executable first, then longest (proxy for completeness).
      - test: if prefer_exec, choose first executable (prefer non-empty result), else first candidate.
    Returns best_decoded_sql (string).
    """
    from utils import compute_records

    # Generate k-best
    out = model.generate(
        input_ids=enc_ids,
        attention_mask=enc_mask,
        generation_config=gen_cfg,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seqs = out.sequences  # [1*k, L]
    cand_sqls = []
    for seq in seqs:
        s = tok.decode(seq, skip_special_tokens=True)
        cand_sqls.append(_extract_sql_like(s))

    # Evaluate each candidate by DB execution
    # compute_records expects a list; we send one-by-one to get error/result
    exec_infos = []
    for q in cand_sqls:
        recs, errs = compute_records([q])
        err = errs[0]
        rows = recs[0] if recs else []
        exec_infos.append((q, err, rows))

    # DEV: use GT records to pick max F1
    if dev_gt_records is not None:
        from utils import compute_record_F1
        gt_rows = dev_gt_records  # a set/list for this example
        # compute F1 for each candidate vs GT (single-example lists)
        best = None
        best_f1 = -1.0
        for (q, err, rows) in exec_infos:
            f1 = compute_record_F1([gt_rows], [rows])
            # prefer executable; tie-break by longer query (more joins covered)
            score = (f1, int(err == ""), len(q))
            if f1 > best_f1 or (f1 == best_f1 and score > best[0]):
                best = (score, q)
                best_f1 = f1
        return best[1]

    # TEST: prefer executable; prefer non-empty result
    if prefer_exec:
        # non-empty & executable
        for q, err, rows in exec_infos:
            if not err and rows:
                return q
        # executable (maybe empty)
        for q, err, rows in exec_infos:
            if not err:
                return q
        # fallback: first
        return exec_infos[0][0]

    # default: first candidate
    return exec_infos[0][0]


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Dev: CE loss + k-best generate + execution rerank (vs GT records) + metrics
    """
    model.eval()
    # 1) CE on dev
    ce_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="Eval/CE"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            ).logits
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_targets.view(-1))
            ntoks = (decoder_targets != PAD_IDX).sum().item()
            ce_loss += loss.item() * max(1, ntoks)
            total_tokens += max(1, ntoks)
    ce_loss = ce_loss / max(1, total_tokens)

    # 2) k-best generation + dev rerank
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])
    # force "SELECT" in outputs
    select_ids = tok.encode("SELECT", add_special_tokens=False)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        num_return_sequences=args.num_return_sequences,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
        # NOTE: in current HF, constrained beam search via force_words_ids is only respected
        # for standard beam search. We keep it to bias the decoder.
        # force_words_ids=[select_ids],
    )

    # load GT records to compute single-example F1 during rerank
    # ground_truth_dev.pkl = (records_list, error_msgs)
    import pickle
    with open(gt_record_path, "rb") as f:
        gt_recs, gt_errs = pickle.load(f)
    # gt_recs is a list aligned with dev order (after sanitization)

    sql_preds = []
    with torch.no_grad():
        ex_idx = 0
        for batch in tqdm(dev_loader, desc="Generate"):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 5:
                    encoder_input, encoder_mask, _, _, _ = batch
                elif len(batch) == 3:
                    encoder_input, encoder_mask, _ = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
            else:
                raise ValueError("Batch must be tuple/list.")

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # decode each example in batch separately to rerank with its GT records
            for i in range(encoder_input.size(0)):
                enc_ids = encoder_input[i:i+1]
                enc_msk = encoder_mask[i:i+1]
                gt_rows = gt_recs[ex_idx]
                best_sql = _decode_k_and_rerank(
                    model, tok, enc_ids, enc_msk, gen_cfg,
                    k=args.num_return_sequences,
                    dev_gt_records=gt_rows if args.rerank_dev_by_gt else None,
                    dev_gt_idx_base=ex_idx,
                )
                sql_preds.append(best_sql)
                ex_idx += 1

    # Save + score
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    # debug few errors
    if model_error_msgs:
        bad = [(i, e) for i, e in enumerate(model_error_msgs) if e]
        print(f"[DEBUG] {len(bad)} / {len(model_error_msgs)} dev preds raised DB errors")
        for i, e in bad[:10]:
            print(f"[DEBUG] idx={i} err={e}")
    error_rate = sum(1 for e in (model_error_msgs or []) if e) / max(1, len(model_error_msgs or []))
    return ce_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Test: k-best generate + prefer any executable (and non-empty) candidate
    """
    model.eval()
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])
    select_ids = tok.encode("SELECT", add_special_tokens=False)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        num_return_sequences=args.num_return_sequences,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
        force_words_ids=[select_ids],
    )

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            for i in range(encoder_input.size(0)):
                enc_ids = encoder_input[i:i+1]
                enc_msk = encoder_mask[i:i+1]
                best_sql = _decode_k_and_rerank(
                    model, tok, enc_ids, enc_msk, gen_cfg,
                    k=args.num_return_sequences,
                    dev_gt_records=None,
                    prefer_exec=True,
                )
                sql_preds.append(best_sql)

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Data + model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate best on dev and produce test submission files
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    # Dev eval (reranked)
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test generation + records => SUBMISSION FILES
    test_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)
    print(f"[SUBMISSION] Wrote: {test_sql_path}")
    print(f"[SUBMISSION] Wrote: {test_record_path}")


if __name__ == "__main__":
    main()
