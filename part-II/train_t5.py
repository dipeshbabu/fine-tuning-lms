from transformers import T5TokenizerFast, GenerationConfig, StoppingCriteria, StoppingCriteriaList
import os, re, argparse
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
from utils import compute_metrics, save_queries_and_records

# ======= Runtime speed knobs (safe) =======
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True
# ==========================================

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


class StopOnSemicolon(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.semi_id = self.tokenizer.convert_tokens_to_ids(";")
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -1].item() == self.semi_id


def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 or not")

    # Training
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["none", "cosine", "linear"])
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--max_n_epochs", type=int, default=6)
    parser.add_argument("--patience_epochs", type=int, default=3)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="dev")

    # Data
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # Final-generation settings (for last dev eval + test)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_beam_size", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=8)

    # Fast eval during training (still decodes unless CE-only is set)
    parser.add_argument("--fast_eval", action="store_true",
                        help="Use lighter generation params during between-epoch dev eval.")
    parser.add_argument("--train_eval_beams", type=int, default=2)
    parser.add_argument("--train_eval_k", type=int, default=2)
    parser.add_argument("--train_eval_max_new", type=int, default=64)

    # Reranking logic
    parser.add_argument("--rerank_dev_by_gt", action="store_true",
                        help="Dev only: choose candidate by max F1 vs ground-truth records")
    parser.add_argument("--prefer_executable_on_test", action="store_true",
                        help="Test: among k candidates, pick any executable (prefer non-empty)")

    # *** New: make training much faster ***
    parser.add_argument("--eval_ce_only", action="store_true",
                        help="During training: dev eval returns CE loss only (no generation/DB).")
    parser.add_argument("--skip_dev_eval", action="store_true",
                        help="Skip between-epoch dev eval entirely (final pass still runs).")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="If >0, cap train batches per epoch for quick iterations.")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="If >0, cap dev/test batches during CE and generation.")

    return parser.parse_args()


def _extract_sql_like(s: str) -> str:
    s = s.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    semi = s.find(";")
    if semi != -1:
        return s[: semi + 1].strip()
    if not s.lower().startswith("select"):
        s = "SELECT " + s
    if not s.endswith(";"):
        s = s + ";"
    return s


def _batch_generate_k(model, tok, enc_ids, enc_mask, gen_cfg, k):
    out = model.generate(
        input_ids=enc_ids,
        attention_mask=enc_mask,
        generation_config=gen_cfg,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seqs = out.sequences  # [batch*k, L]
    B = enc_ids.size(0)
    all_sqls = []
    for i in range(B):
        cand_strs = []
        start = i * k
        end = (i + 1) * k
        for seq in seqs[start:end]:
            s = tok.decode(seq, skip_special_tokens=True)
            cand_strs.append(_extract_sql_like(s))
        all_sqls.append(cand_strs)
    return all_sqls


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    experiment_name = args.experiment_name
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"{model_type}_experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        if args.skip_dev_eval:
            save_model(checkpoint_dir, model, best=False)
            continue

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, "
              f"Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
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

    # AMP + optional compile
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    compiled_forward = None
    try:
        compiled_forward = torch.compile(model)  # PyTorch 2.x
    except Exception:
        compiled_forward = model

    for bidx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(
        tqdm(train_loader, desc="Train")
    ):
        if args.max_train_batches and bidx >= args.max_train_batches:
            break

        optimizer.zero_grad(set_to_none=True)
        encoder_input = encoder_input.to(DEVICE, non_blocking=True)
        encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
        decoder_input = decoder_input.to(DEVICE, non_blocking=True)
        decoder_targets = decoder_targets.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = compiled_forward(
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


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Dev: CE loss; optionally (if not CE-only) batched k-best generation + rerank vs GT + metrics.
    """
    model.eval()
    # 1) CE on dev
    ce_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    with torch.no_grad():
        for bidx, (encoder_input, encoder_mask, decoder_input, decoder_targets, _) in enumerate(
            tqdm(dev_loader, desc="Eval/CE")
        ):
            if args.max_eval_batches and bidx >= args.max_eval_batches:
                break
            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
            decoder_input = decoder_input.to(DEVICE, non_blocking=True)
            decoder_targets = decoder_targets.to(DEVICE, non_blocking=True)
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

    # Fast path: CE-only during training to avoid generation/DB
    if getattr(args, "eval_ce_only", False):
        return ce_loss, 0.0, 0.0, 0.0, 0.0

    # 2) Generation params
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    if args.fast_eval:
        k = max(1, args.train_eval_k)
        beams = max(1, args.train_eval_beams)
        max_new = max(8, args.train_eval_max_new)
    else:
        k = max(1, args.num_return_sequences)
        beams = max(1, args.gen_beam_size)
        max_new = max(8, args.gen_max_new_tokens)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        num_beams=beams,
        num_return_sequences=k,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )

    # 3) Load GT records once
    import pickle
    with open(gt_record_path, "rb") as f:
        gt_recs, gt_errs = pickle.load(f)

    sql_preds = []
    err_msgs_accum = []
    ex_idx = 0

    from utils import compute_records, compute_record_F1

    with torch.no_grad():
        for bidx, batch in enumerate(tqdm(dev_loader, desc="Generate")):
            if args.max_eval_batches and bidx >= args.max_eval_batches:
                break

            # unpack batch (supports both 5-tuple or 3-tuple)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 5:
                    encoder_input, encoder_mask, _, _, _ = batch
                elif len(batch) == 3:
                    encoder_input, encoder_mask, _ = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
            else:
                raise ValueError("Batch must be tuple/list.")

            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
            B = encoder_input.size(0)

            # ONE generate call for the whole batch
            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            # Rerank per example
            for i in range(B):
                cands = all_cands[i]  # k SQL strings
                recs, errs = compute_records(cands)

                if args.rerank_dev_by_gt and ex_idx < len(gt_recs):
                    gt_rows = gt_recs[ex_idx]
                    best_q = cands[0]
                    best_score = (-1.0, 0, 0)  # (F1, is_executable, len(q))
                    for q, r, e in zip(cands, recs, errs):
                        f1 = compute_record_F1([gt_rows], [r])
                        score = (f1, int(e == ""), len(q))
                        if score > best_score:
                            best_score = score
                            best_q = q
                    sql_preds.append(best_q)
                    chosen_idx = cands.index(best_q)
                    err_msgs_accum.append(errs[chosen_idx])
                else:
                    # prefer executable and non-empty; fallback first
                    picked = None
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q; err_msgs_accum.append(e); break
                    if picked is None:
                        for q, r, e in zip(cands, recs, errs):
                            if not e:
                                picked = q; err_msgs_accum.append(e); break
                    if picked is None:
                        picked = cands[0]; err_msgs_accum.append(errs[0])
                    sql_preds.append(picked)

                ex_idx += 1

    # Save + score
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    if not model_error_msgs:
        model_error_msgs = err_msgs_accum

    bad = [(i, e) for i, e in enumerate(model_error_msgs) if e]
    if bad:
        print(f"[DEBUG] {len(bad)} / {len(model_error_msgs)} dev preds raised DB errors")
        for i, e in bad[:10]:
            print(f"[DEBUG] idx={i} err={e}")

    error_rate = sum(1 for e in (model_error_msgs or []) if e) / max(1, len(model_error_msgs or []))
    return ce_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Test: batch-generate k candidates per example + prefer executable (and non-empty) candidate.
    """
    model.eval()
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    k = max(1, args.num_return_sequences)
    beams = max(1, args.gen_beam_size)
    max_new = max(8, args.gen_max_new_tokens)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        num_beams=beams,
        num_return_sequences=k,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )

    from utils import compute_records

    sql_preds = []
    with torch.no_grad():
        for bidx, (encoder_input, encoder_mask, _) in enumerate(tqdm(test_loader, desc="Generate/Test")):
            if args.max_eval_batches and bidx >= args.max_eval_batches:
                break

            encoder_input = encoder_input.to(DEVICE, non_blocking=True)
            encoder_mask = encoder_mask.to(DEVICE, non_blocking=True)
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            for i in range(B):
                cands = all_cands[i]
                recs, errs = compute_records(cands)

                picked = None
                if args.prefer_executable_on_test:
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q; break
                    if picked is None:
                        for q, r, e in zip(cands, recs, errs):
                            if not e:
                                picked = q; break
                if picked is None:
                    picked = cands[0]
                sql_preds.append(picked)

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

    # Train (fast path: CE-only or skip dev eval)
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate best on dev and produce test submission files (full params)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    # Force full params for final dev eval
    was_fast = args.fast_eval
    was_ce_only = args.eval_ce_only
    args.fast_eval = False
    args.eval_ce_only = False
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    args.fast_eval = was_fast
    args.eval_ce_only = was_ce_only

    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, "
          f"Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test generation + records => SUBMISSION FILES
    test_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)
    print(f"[SUBMISSION] Wrote: {test_sql_path}")
    print(f"[SUBMISSION] Wrote: {test_record_path}")


if __name__ == "__main__":
    main()
