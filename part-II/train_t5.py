from transformers import T5TokenizerFast, GenerationConfig
import os
import re
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

try:
    import wandb
except Exception:
    wandb = None

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
    get_tokenizer,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records


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


# ----------------------------
# SQL cleanup & repair helpers
# ----------------------------

_SQL_KW = [
    "select", "from", "where", "group by", "order by", "having", "limit",
    "join", "inner join", "left join", "right join", "on", "and", "or",
    "distinct", "as", "asc", "desc", "count", "avg", "sum", "min", "max"
]

def _caps_keywords(sql: str) -> str:
    s = sql
    for kw in sorted(_SQL_KW, key=len, reverse=True):
        s = re.sub(rf"\b{kw}\b", kw.upper(), s, flags=re.IGNORECASE)
    return s

def _strip_comments_and_weird(sql: str) -> str:
    # Remove backticks and stray quotes that often show up
    s = sql.replace("`", "")
    # Remove SQL-style comments
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # Replace exotic unicode quotes
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return s

def _single_statement(sql: str) -> str:
    # Keep only the first semicolon-terminated statement, or until end if none.
    s = sql.strip()
    # If multiple ';' present, take the first statement only
    if ";" in s:
        s = s.split(";")[0] + ";"
    return s

def _ensure_select(sql: str) -> str:
    s = sql.strip()
    if not re.match(r"^\s*select\b", s, flags=re.IGNORECASE):
        # heurstic: prepend SELECT when model produces fragments like "airport_code, city_code ..."
        s = "SELECT " + s
    if not s.strip().endswith(";"):
        s = s.strip() + ";"
    return s

def _squeeze_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()

def _basic_column_fixes(sql: str) -> str:
    # Undo common broken tokens like trailing underscores
    s = re.sub(r"\b([A-Za-z_]+)_\b", r"\1", sql)
    # Replace 'TO' between comparisons -> often a split token; leave logic intact
    s = re.sub(r"\bAND\s*=\s*", " AND ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b=\s*AND\b", " AND ", s, flags=re.IGNORECASE)
    return s

def repair_sql(sql: str) -> str:
    """
    Aggressive but safe-ish single-statement sanitizer.
    It won't guarantee correctness, but it eliminates most syntax bombs that block execution.
    """
    s = sql.strip()
    s = _strip_comments_and_weird(s)
    s = _single_statement(s)
    s = _ensure_select(s)
    s = _squeeze_ws(s)
    s = _caps_keywords(s)
    s = _basic_column_fixes(s)
    # Prevent starting with SELECT ;
    if re.match(r"^\s*SELECT\s*;\s*$", s, flags=re.IGNORECASE):
        s = "SELECT 1;"
    return s


def extract_sql_like(s: str) -> str:
    """
    Extract a single SELECT ... ; span from raw decoded text.
    """
    s0 = s.strip()
    m = re.search(r"(select\s.+?;)", s0, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return _squeeze_ws(m.group(1))
    # Fallback: keep first line and make it look like a query
    s0 = s0.splitlines()[0]
    s0 = _single_statement(s0)
    s0 = _ensure_select(s0)
    s0 = _squeeze_ws(s0)
    return s0


# ----------------------------
# CLI
# ----------------------------

def get_args():
    p = argparse.ArgumentParser(description="T5 training loop (SQL)")

    # Model
    p.add_argument("--finetune", action="store_true", help="Finetune pretrained T5; else random init")

    # Optim/training
    p.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW"])
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_type", type=str, default="linear", choices=["none", "cosine", "linear"])
    p.add_argument("--num_warmup_epochs", type=int, default=0)
    p.add_argument("--max_n_epochs", type=int, default=6)
    p.add_argument("--patience_epochs", type=int, default=3)

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--experiment_name", type=str, default="dev")

    # Data
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)

    # Generation (final/default)
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gen_beam_size", type=int, default=8)
    p.add_argument("--num_return_sequences", type=int, default=8)
    p.add_argument("--length_penalty", type=float, default=0.6)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--repetition_penalty", type=float, default=1.05)

    # Fast eval during training
    p.add_argument("--fast_eval", action="store_true",
                   help="Light generation during between-epoch eval; final eval uses full params.")
    p.add_argument("--train_eval_beams", type=int, default=2)
    p.add_argument("--train_eval_k", type=int, default=2)
    p.add_argument("--train_eval_max_new", type=int, default=64)

    # Reranking logic
    p.add_argument("--rerank_dev_by_gt", action="store_true",
                   help="Dev only: choose candidate by max F1 vs ground-truth records")
    p.add_argument("--prefer_executable_on_test", action="store_true",
                   help="Test: among k candidates, pick any executable (prefer non-empty)")

    # Post-processing
    p.add_argument("--repair_sql", action="store_true", help="Apply sanitizer to each generated candidate")

    # (Optional) Schema prompting hook — enable in load_data.py later if desired.
    # p.add_argument("--prepend_schema", action="store_true")

    return p.parse_args()


# ----------------------------
# Core train/eval
# ----------------------------

def _batch_generate_k(model, tok, enc_ids, enc_mask, gen_cfg, k, do_repair: bool):
    """
    Generate k candidates per example for the batch in ONE call.
    Returns list[list[str]] length B; each inner list k decoded SQL strings (sanitized if do_repair).
    """
    with torch.no_grad():
        out = model.generate(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=False,
        )
    seqs = out.sequences  # [B*k, L]
    B = enc_ids.size(0)
    all_sqls = []
    for i in range(B):
        cand_strs = []
        start = i * k
        end = (i + 1) * k
        for seq in seqs[start:end]:
            s = tok.decode(seq, skip_special_tokens=True)
            q = extract_sql_like(s)
            if do_repair:
                q = repair_sql(q)
            cand_strs.append(q)
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

    # Dev file paths (GT already sanitized)
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")  # must exist

    # Model predictions (dev)
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"[{model_type}][train] Epoch {epoch}: Average train loss {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"[{model_type}][dev] Epoch {epoch}: Dev loss {eval_loss:.4f}, "
              f"Record F1 {record_f1:.4f}, Record EM {record_em:.4f}, SQL EM {sql_em:.4f}")
        print(f"[{model_type}][dev] Epoch {epoch}: {error_rate*100:.2f}% SQL errors")

        if args.use_wandb and wandb is not None:
            wandb.log({
                "train/loss": tr_loss,
                "dev/loss": eval_loss,
                "dev/record_f1": record_f1,
                "dev/record_em": record_em,
                "dev/sql_em": sql_em,
                "dev/error_rate": error_rate,
            }, step=epoch)

        improved = record_f1 > best_f1
        save_model(checkpoint_dir, model, best=False)
        if improved:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

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


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Dev: CE loss + batched k-best generation + (optional) rerank vs GT + metrics.
    Uses fast params during training epochs if --fast_eval is set; the final dev eval (in main) uses full params.
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

    # 2) Generation params (fast vs full)
    tok = get_tokenizer()
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
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
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
        for batch in tqdm(dev_loader, desc="Generate"):
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

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(
                model, tok, encoder_input, encoder_mask, gen_cfg, k, do_repair=args.repair_sql
            )

            # Rerank per example
            for i in range(B):
                cands = all_cands[i]  # k SQL strings
                recs, errs = compute_records(cands)

                if args.rerank_dev_by_gt and ex_idx < len(gt_recs):
                    gt_rows = gt_recs[ex_idx]
                    best_q = cands[0]
                    best_score = (-1.0, 0, 0)  # (F1, is_executable, -len(q) prefer shorter)
                    for q, r, e in zip(cands, recs, errs):
                        f1 = compute_record_F1([gt_rows], [r])
                        score = (f1, int(e == ""), -len(q))
                        if score > best_score:
                            best_score = score
                            best_q = q
                    sql_preds.append(best_q)
                    chosen_idx = cands.index(best_q)
                    err_msgs_accum.append(errs[chosen_idx])
                else:
                    # No GT: prefer executable & non-empty
                    picked = None
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q
                            err_msgs_accum.append(e)
                            break
                    if picked is None:
                        for q, r, e in zip(cands, recs, errs):
                            if not e:
                                picked = q
                                err_msgs_accum.append(e)
                                break
                    if picked is None:
                        picked = cands[0]
                        err_msgs_accum.append(errs[0])
                    sql_preds.append(picked)

                ex_idx += 1

    # Save + score
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    # If compute_metrics doesn't return per-example errors, fall back to ours
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
    tok = get_tokenizer()
    k = max(1, args.num_return_sequences)
    beams = max(1, args.gen_beam_size)
    max_new = max(8, args.gen_max_new_tokens)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new,
        num_beams=beams,
        num_return_sequences=k,
        do_sample=False,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )

    from utils import compute_records

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(
                model, tok, encoder_input, encoder_mask, gen_cfg, k, do_repair=args.repair_sql
            )

            for i in range(B):
                cands = all_cands[i]
                recs, errs = compute_records(cands)

                picked = None
                if args.prefer_executable_on_test:
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q
                            break
                    if picked is None:
                        for q, r, e in zip(cands, recs, errs):
                            if not e:
                                picked = q
                                break
                if picked is None:
                    picked = cands[0]
                sql_preds.append(picked)

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)


def main():
    args = get_args()
    if args.use_wandb and wandb is not None:
        setup_wandb(args)

    # Data + model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args).to(DEVICE)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train (between-epoch eval can be sped up with --fast_eval)
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate best on dev and produce test submission files (full params, NOT fast_eval)
    model = load_model_from_checkpoint(args, best=True, fallback_model=model).to(DEVICE)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    # Ensure final dev eval runs with full beams/k (ignore --fast_eval here)
    was_fast = args.fast_eval
    args.fast_eval = False
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    args.fast_eval = was_fast

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
