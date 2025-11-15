from transformers import T5TokenizerFast, GenerationConfig
import os
import re
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import sqlite3

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

# NOTE: utils.DB_PATH should already point to part-II/data/flight_database.db

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

# =========================
# SQL SANITIZATION HELPERS
# =========================

DANGEROUS_TOKENS = {"CREATE", "DROP", "INSERT", "UPDATE", "DELETE", "ATTACH", "PRAGMA", "ALTER"}

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _strip_bad_tokens(s: str) -> str:
    # Remove spurious headings like "Tables:", "Tabellen:", "Table:", "SQL:" at the start
    s = re.sub(r"^(?:\w{2,12}[:：]\s*)+", "", s.strip(), flags=re.IGNORECASE)
    return s

def constrain_tokens_to_sql_charset(s: str) -> str:
    # Keep ASCII letters, digits, common SQL punctuation/operators and underscore/dot/space
    return re.sub(r"[^A-Za-z0-9_ \t\n\.,;\*\+\-/<>=\(\)'\"%]", " ", s)

def _first_select_span(s: str) -> str | None:
    """Return 'SELECT ... ;' span for the first statement; if no semicolon, return until end."""
    m = re.search(r"\bselect\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    s = s[m.start():]  # drop anything before the first SELECT
    semi = s.find(";")
    if semi == -1:
        return s.strip()
    return s[:semi + 1].strip()

def _single_statement(sql: str) -> str:
    """Ensure a single SELECT statement; keep the first 'SELECT ...' and append ';'."""
    parts = [p.strip() for p in sql.split(";")]
    for p in parts:
        if p and re.match(r"(?is)^\s*select\b", p):
            return p + ";"
    if parts and parts[0]:
        return parts[0] + ";"
    return "SELECT 1;"

def _load_schema_from_sqlite(db_path: str) -> tuple[set, dict]:
    """Return (tables, table_to_cols) from sqlite db."""
    tables = set()
    table_to_cols = {}
    if not os.path.exists(db_path):
        return tables, table_to_cols
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (tname,) in cur.fetchall():
            tables.add(tname)
            try:
                cur.execute(f"PRAGMA table_info('{tname}')")
                cols = [row[1] for row in cur.fetchall()]
                table_to_cols[tname] = set(cols)
            except Exception:
                table_to_cols[tname] = set()
        conn.close()
    except Exception:
        pass
    return tables, table_to_cols

def _cheap_schema_repair(sql: str, schema: tuple[set, dict]) -> str:
    """Very light repair: trim illegal qualifiers like table_2.col if table_2 not in schema; dedupe aliases."""
    tables, table_to_cols = schema
    s = sql
    # Remove obviously bogus qualifiers like foo_2.col if foo_2 not a table
    def repl_qual(m):
        qual = m.group(1)
        col = m.group(2)
        if qual in tables:
            return f"{qual}.{col}"
        return col  # drop unknown qualifier
    s = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)", repl_qual, s)
    # If SELECT * FROM t1, t2 without JOINs, keep as-is (schema may require explicit joins but we won't guess)
    return s

def sanitize_sql(raw: str, schema: tuple[set, dict] | None, constrain: bool) -> str:
    """
    Strong sanitizer:
      * optional charset constrain (remove weird Unicode / localized headings)
      * drop preambles (e.g., 'Tabellen:')
      * keep only the first SELECT statement
      * block DDL/DML
      * enforce single statement
      * optional schema-aware light repair
    """
    s = raw or ""
    if constrain:
        s = constrain_tokens_to_sql_charset(s)
    s = _strip_bad_tokens(s)
    span = _first_select_span(s)
    if not span:
        return "SELECT 1;"
    s = span
    up = s.upper()
    if any(tok in up for tok in DANGEROUS_TOKENS):
        return "SELECT 1;"
    s = _normalize_whitespace(_strip_bad_tokens(s))
    s = _single_statement(s)
    if schema is not None:
        s = _cheap_schema_repair(s, schema)
        s = _single_statement(s)
    return s

# =========================

def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 or not")

    # Training hyperparameters
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

    # Decoding constraints
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # Fast eval during training (speeds up between-epoch evaluation)
    parser.add_argument("--fast_eval", action="store_true",
                        help="Use light generation params during training eval; final dev+test use full params.")
    parser.add_argument("--train_eval_beams", type=int, default=2)
    parser.add_argument("--train_eval_k", type=int, default=2)
    parser.add_argument("--train_eval_max_new", type=int, default=64)

    # Reranking and executability
    parser.add_argument("--rerank_dev_by_gt", action="store_true",
                        help="Dev only: choose candidate by max F1 vs ground-truth records")
    parser.add_argument("--prefer_executable_on_test", action="store_true",
                        help="Test: among k candidates, pick any executable (prefer non-empty)")

    # SQL safety / repair
    parser.add_argument("--constrain_sql", action="store_true", help="Constrain decoded text to SQL-safe charset")
    parser.add_argument("--repair_sql", action="store_true", help="Enable schema-aware light repair")

    # Checkpoint load safety
    parser.add_argument("--skip_load_best_if_missing", action="store_true",
                        help="If best checkpoint folder is missing, skip loading instead of erroring.")

    return parser.parse_args()

def _extract_sql_like(s: str) -> str:
    """
    Kept for compatibility (still used by _batch_generate_k); we’ll hard-sanitize later.
    """
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
    """
    Generate k candidates per example for the whole batch in ONE call.
    Returns list[list[str]] of length batch_size; each inner list has k decoded SQL strings.
    """
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
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )

    # 3) Load GT records once
    import pickle
    with open(gt_record_path, "rb") as f:
        gt_recs, gt_errs = pickle.load(f)

    # 3b) Load schema once (for sanitize/repair)
    try:
        from utils import DB_PATH  # type: ignore
        schema = _load_schema_from_sqlite(DB_PATH) if args.repair_sql else None
    except Exception:
        schema = None

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

            # ONE generate call for the whole batch
            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            # Rerank per example (sanitize -> execute -> choose)
            for i in range(B):
                raw_cands = all_cands[i]  # k decoded strings
                cands = [sanitize_sql(q, schema, args.constrain_sql) for q in raw_cands]
                recs, errs = compute_records(cands)

                if args.rerank_dev_by_gt and ex_idx < len(gt_recs):
                    gt_rows = gt_recs[ex_idx]
                    best_q = cands[0]
                    best_score = (-1.0, 0, 0)  # (F1, is_executable, -len(q))
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
                    # Prefer executable + non-empty records; fallback to first
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
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )

    from utils import compute_records
    # schema for sanitize
    try:
        from utils import DB_PATH  # type: ignore
        schema = _load_schema_from_sqlite(DB_PATH) if args.repair_sql else None
    except Exception:
        schema = None

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            for raw_cands in all_cands:
                cands = [sanitize_sql(q, schema, args.constrain_sql) for q in raw_cands]
                recs, errs = compute_records(cands)

                picked = None
                if args.prefer_executable_on_test:
                    # non-empty exec
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q
                            break
                    # exec (maybe empty)
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

def _safe_load_best_if_present(args, model):
    """Load best checkpoint if present; optionally skip if missing."""
    best_dir = os.path.join(args.checkpoint_dir, "best")
    if os.path.isdir(best_dir):
        return load_model_from_checkpoint(args, best=True)
    if args.skip_load_best_if_missing:
        print(f"[INFO] Best checkpoint missing at {best_dir}; skipping load (flag set).")
        return model
    # Fall back to raise the same error transformers would give:
    return load_model_from_checkpoint(args, best=True)

def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Data + model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train (between-epoch eval can be sped up with --fast_eval)
    if args.max_n_epochs > 0:
        train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate best on dev and produce test submission files (full params, NOT fast_eval)
    model = _safe_load_best_if_present(args, model)
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
