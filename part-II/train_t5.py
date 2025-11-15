from transformers import T5TokenizerFast, GenerationConfig
import os
import re
import argparse
from tqdm import tqdm
import sqlite3
from collections import defaultdict

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
from utils import compute_metrics, save_queries_and_records, DB_PATH

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

# Resolve paths relative to this file so CWD doesn’t matter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ------------------------------- SQL helpers --------------------------------- #

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "GROUP", "BY", "ORDER", "LIMIT",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AS", "HAVING", "DISTINCT",
]

def _ends_with_semicolon(s: str) -> str:
    s = s.strip()
    if not s.endswith(";"):
        s += ";"
    return s

def _normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = s.strip()
    return s

def _uppercase_keywords(s: str) -> str:
    def repl(m):
        k = m.group(0)
        return k.upper() if k.upper() in SQL_KEYWORDS else k
    return re.sub(r"\b[a-zA-Z]+\b", repl, s)

def _strip_bad_tokens(s: str) -> str:
    # Remove code fences and stray backticks, quotes balancing
    s = s.replace("```", " ").replace("`", " ")
    # Remove comments if any
    s = re.sub(r"--.*?$", " ", s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return s

def _first_select_clause(s: str) -> str:
    """
    Extract the first 'SELECT ... ;' span; fallback to best-effort.
    """
    s = s.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to cut at first semicolon
    semi = s.find(";")
    if semi != -1:
        return s[:semi + 1].strip()
    # Fallback: enforce a single-statement ending with semicolon
    return _ends_with_semicolon(s)

def _load_schema(db_path: str):
    """
    Returns: (tables: set[str], columns: dict[str, set[str]])
    """
    tables = set()
    columns = defaultdict(set)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for (tname,) in cur.fetchall():
            tables.add(tname)
            try:
                cur.execute(f"PRAGMA table_info({tname});")
                for row in cur.fetchall():
                    # row = (cid, name, type, notnull, dflt_value, pk)
                    columns[tname].add(row[1])
            except Exception:
                pass
    finally:
        conn.close()
    return tables, columns

def _guess_table_from_columns(sql: str, tables: set, columns: dict) -> str | None:
    """
    Heuristic: find table that has most of the bare column names referenced.
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql)
    # filter out obvious keywords
    toks = [t for t in tokens if t.upper() not in SQL_KEYWORDS]
    scores = {}
    for t in tables:
        colset = columns.get(t, set())
        scores[t] = sum(1 for tok in toks if tok in colset)
    # pick max if positive
    best = max(scores.items(), key=lambda x: x[1]) if scores else (None, 0)
    return best[0] if best and best[1] > 0 else None

def _fix_table_suffixes(sql: str, tables: set) -> str:
    """
    Replace table_name_# with table_name if base exists in schema (seen errors: flight_1, airport_service_2).
    """
    def repl(m):
        base = m.group(1)
        num = m.group(2)
        return base if base in tables else f"{base}_{num}"
    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)_(\d+)\b", repl, sql)

def _dedupe_aliases(sql: str) -> str:
    # prevent "airport_service_1.airport_code" ambiguity by removing suffixes consistently
    sql = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)_1\.", r"\1.", sql)
    sql = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)_2\.", r"\1.", sql)
    return sql

def _ensure_from_clause(sql: str, tables: set, columns: dict) -> str:
    if re.search(r"\bFROM\b", sql, flags=re.IGNORECASE):
        return sql
    # Try to pick a table from columns mentioned
    guessed = _guess_table_from_columns(sql, tables, columns)
    if not guessed:
        # Try common defaults if available
        for candidate in ["flight", "airport", "city", "airport_service", "flight_leg"]:
            if candidate in tables:
                guessed = candidate
                break
    if not guessed and tables:
        guessed = sorted(tables)[0]
    # Insert FROM right after SELECT list if missing
    # crude split: SELECT ... FROM ...
    # if user wrote conditions first, just prepend SELECT * FROM <guessed>
    if not re.match(r"(?is)^\s*SELECT\b", sql):
        return f"SELECT * FROM {guessed} WHERE " + re.sub(r"^(?is)(SELECT\b)?", "", sql)
    if "FROM" not in sql.upper():
        # try to split at WHERE/GROUP/ORDER/LIMIT
        m = re.search(r"(?i)\b(WHERE|GROUP|ORDER|LIMIT)\b", sql)
        if m:
            idx = m.start()
            return sql[:idx] + f" FROM {guessed} " + sql[idx:]
        else:
            return _ends_with_semicolon(sql + f" FROM {guessed}")
    return sql

def _tidy_where(sql: str) -> str:
    # Collapse 'WHERE AND' / leading AND/OR
    sql = re.sub(r"(?i)\bWHERE\s+(AND|OR)\b", "WHERE ", sql)
    sql = re.sub(r"(?i)\b(AND|OR)\s*(GROUP|ORDER|LIMIT)\b", r"\2", sql)
    sql = re.sub(r"(?i)\bWHERE\s*\)", ") ", sql)
    return sql

def _fix_trailing_ops(sql: str) -> str:
    # Replace 'col =' with 'col IS NOT NULL' to avoid syntax error (fallback)
    sql = re.sub(r"(?i)\b([A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*(?=(GROUP|ORDER|LIMIT|AND|OR|;|$))", r"\1 IS NOT NULL ", sql)
    # Remove ', ,' or ',,'
    sql = re.sub(r",\s*,", ", ", sql)
    return sql

def repair_sql(candidate: str, schema: tuple[set, dict]) -> str:
    """
    Conservative repair to reduce 'near "=" / "," / "and" / "__code"' syntax errors.
    """
    tables, columns = schema
    s = candidate or ""
    s = _strip_bad_tokens(s)
    s = _first_select_clause(s)
    s = _normalize_whitespace(s)
    # Basic cleanups
    s = _fix_table_suffixes(s, tables)
    s = _dedupe_aliases(s)
    s = _uppercase_keywords(s)
    s = _ensure_from_clause(s, tables, columns)
    s = _tidy_where(s)
    s = _fix_trailing_ops(s)
    s = _ends_with_semicolon(s)
    return s

def constrain_tokens_to_sql_charset(text: str) -> str:
    # Keep a safe charset for sqlite identifiers/operators
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.*=<>!,'\"()% -;\n")
    return "".join(ch for ch in text if ch in allowed)

# ------------------------------- Argparse ------------------------------------ #

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

    # Final-generation settings
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_beam_size", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=8)

    # Extra generation controls
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # Fast eval during training
    parser.add_argument("--fast_eval", action="store_true",
                        help="Use light generation params during training eval; final dev+test use full params.")
    parser.add_argument("--train_eval_beams", type=int, default=2)
    parser.add_argument("--train_eval_k", type=int, default=2)
    parser.add_argument("--train_eval_max_new", type=int, default=64)

    # Reranking logic
    parser.add_argument("--rerank_dev_by_gt", action="store_true",
                        help="Dev only: choose candidate by max F1 vs ground-truth records")
    parser.add_argument("--prefer_executable_on_test", action="store_true",
                        help="Test: among k candidates, pick any executable (prefer non-empty)")

    # Constrain/Repair toggles
    parser.add_argument("--constrain_sql", action="store_true",
                        help="Restrict decoded chars to a safe SQL charset.")
    parser.add_argument("--repair_sql", action="store_true",
                        help="Apply schema-aware repair before execution.")

    # Safety: skip loading best if no checkpoint exists
    parser.add_argument("--skip_load_best_if_missing", action="store_true",
                        help="If no best checkpoint found, skip loading instead of crashing.")
    return parser.parse_args()

# ----------------------------- Generation utils ------------------------------ #

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

def _batch_generate_k(model, tok, enc_ids, enc_mask, gen_cfg, k, constrain=False):
    """
    Generate k candidates per example for the whole batch in ONE call.
    Returns list[list[str]] length B; each inner list has k decoded SQL strings.
    """
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
            if constrain:
                s = constrain_tokens_to_sql_charset(s)
            cand_strs.append(_extract_sql_like(s))
        all_sqls.append(cand_strs)
    return all_sqls

# ------------------------------- Train / Eval -------------------------------- #

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
        print(f"[{model_type}][dev] Epoch {epoch}: Average train loss {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"[{model_type}][dev] Epoch {epoch}: Dev loss {eval_loss:.4f}, Record F1 {record_f1:.4f}, "
              f"Record EM {record_em:.4f}, SQL EM {sql_em:.4f}")
        print(f"[{model_type}][dev] Epoch {epoch}: {error_rate*100:.2f}% SQL errors")

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
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
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

def _make_generation_config(tok, max_new, beams, k, args):
    return GenerationConfig(
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

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Dev: CE loss + batched k-best generation + (optional) rerank vs GT + metrics.
    Uses fast params during training epochs if --fast_eval; final dev eval uses full params in main().
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
    gen_cfg = _make_generation_config(tok, max_new, beams, k, args)

    # 3) Load GT records once
    import pickle
    with open(gt_record_path, "rb") as f:
        gt_recs, _ = pickle.load(f)

    # Load schema once (for repair)
    schema = _load_schema(DB_PATH) if args.repair_sql else None

    sql_preds = []
    err_msgs_accum = []
    ex_idx = 0

    from utils import compute_records, compute_record_F1

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Generate"):
            # Unpack
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
                model, tok, encoder_input, encoder_mask, gen_cfg, k, constrain=args.constrain_sql
            )

            # Rerank per example
            for i in range(B):
                cands = all_cands[i]  # k SQL strings
                if args.repair_sql and schema is not None:
                    cands = [repair_sql(q, schema) for q in cands]

                recs, errs = compute_records(cands)

                if args.rerank_dev_by_gt and ex_idx < len(gt_recs):
                    gt_rows = gt_recs[ex_idx]
                    best_q = cands[0]
                    best_score = (-1.0, 0, 0)  # (F1, is_exec, -len_penalty)
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
                    # Prefer executable and non-empty; fallback to first
                    picked = None
                    # non-empty executable
                    for q, r, e in zip(cands, recs, errs):
                        if not e and r:
                            picked = q
                            err_msgs_accum.append(e)
                            break
                    # executable (maybe empty)
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
    gen_cfg = _make_generation_config(tok, max_new, beams, k, args)

    from utils import compute_records
    schema = _load_schema(DB_PATH) if args.repair_sql else None

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            all_cands = _batch_generate_k(
                model, tok, encoder_input, encoder_mask, gen_cfg, k, constrain=args.constrain_sql
            )

            for cands in all_cands:
                if args.repair_sql and schema is not None:
                    cands = [repair_sql(q, schema) for q in cands]

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

# ----------------------------------- Main ------------------------------------ #

def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Data + model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    if args.max_n_epochs > 0:
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        # Try to load best; optionally skip if missing
        try:
            model = load_model_from_checkpoint(args, best=True)
            model.eval()
        except Exception as e:
            if args.skip_load_best_if_missing:
                print(f"[WARN] Could not load best checkpoint; continuing with current model. Err: {e}")
            else:
                raise
    else:
        # No training: avoid loading a non-existent checkpoint
        print("[INFO] max_n_epochs=0: skipping training & checkpoint loading.")

    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    # Final dev eval (full params)
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
