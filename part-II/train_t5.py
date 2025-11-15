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
from typing import Dict, Set, Tuple

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, DB_PATH

# ------------------------------------------------------------------------------------
# Device / paths
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# Schema guard cache + helpers
# ------------------------------------------------------------------------------------
_SCHEMA_CACHE: Dict[str, Set[str]] = {}
_ALL_TABLES: Set[str] = set()

_ID_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')
_SQL_STOPWORDS = {
    "select","from","where","join","on","group","by","order","having","limit",
    "and","or","not","in","as","left","right","inner","outer","distinct","count",
    "sum","avg","min","max","like","between","is","null","asc","desc","union",
    "intersect","except"
}

def _load_schema() -> Tuple[Dict[str, Set[str]], Set[str]]:
    global _SCHEMA_CACHE, _ALL_TABLES
    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE, _ALL_TABLES
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    _ALL_TABLES = set(tables)
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cur.fetchall()]
            _SCHEMA_CACHE[t] = set(cols)
        except Exception:
            _SCHEMA_CACHE[t] = set()
    conn.close()
    return _SCHEMA_CACHE, _ALL_TABLES

def _schema_guard_score(q: str) -> Tuple[int,int,int]:
    """
    Returns (valid_table_refs, valid_column_refs, invalid_refs_penalty).
    Higher total 'score' is better; penalty is negative for invalid refs.
    """
    schema, all_tables = _load_schema()
    tokens = [t.lower() for t in _ID_RE.findall(q)]
    tbl_like = set()
    col_like = set()
    last_kw = None
    for tok in tokens:
        if tok in _SQL_STOPWORDS:
            last_kw = tok
            continue
        if last_kw in ("from","join"):
            tbl_like.add(tok)
        else:
            col_like.add(tok)

    valid_tbl = sum(1 for t in tbl_like if t in all_tables)
    invalid_tbl = sum(1 for t in tbl_like if t not in all_tables)

    valid_col = 0
    invalid_col = 0
    for tok in col_like:
        if tok in _SQL_STOPWORDS or tok in all_tables:
            continue
        if "." in tok:
            t, _, c = tok.partition(".")
            if t in schema and c in schema.get(t, set()):
                valid_col += 1
            else:
                invalid_col += 1
        else:
            appears = any(tok in cols for cols in schema.values())
            if appears:
                valid_col += 1
            else:
                invalid_col += 1

    penalty = -(2*invalid_tbl + 1*invalid_col)
    return valid_tbl, valid_col, penalty

# ------------------------------------------------------------------------------------
# Argparse
# ------------------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop (FT + optional SCR extra-credit)")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune T5 (otherwise scratch)")

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

    # Optional: also train from scratch (extra credit) in same run
    parser.add_argument("--also_train_scratch", action="store_true",
                        help="If set, run a second training loop with from-scratch model.")
    parser.add_argument("--scratch_experiment_name", type=str, default="scr_dev",
                        help="Experiment name for the from-scratch run")

    # Data
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # Final-generation settings (for last dev eval + test)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_beam_size", type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=8)

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

    # Write Gradescope aliases
    parser.add_argument("--write_gradescope_aliases", action="store_true",
                        help="Write copies to t5_ft_experiment_ec[_test].sql and t5_scr_experiment_ec[_test].sql")

    return parser.parse_args()

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def _extract_sql_like(s: str) -> str:
    """
    Heuristic extraction of the first SELECT ... ; span.
    Ensures we return a single SQL statement ending in ';'
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

def _make_generation_config(tok, max_new, beams, k):
    # discourage DDL/DML; keep SELECT-only vibes
    bad = tok(["update","delete","insert","drop","create","alter","truncate"], add_special_tokens=False).input_ids
    return GenerationConfig(
        max_new_tokens=max_new,
        num_beams=beams,
        num_return_sequences=k,
        do_sample=False,
        length_penalty=1.0,
        bad_words_ids=bad if bad else None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )

def _composite_dev_key(f1, is_exec, schema_score, non_empty, qlen):
    return (f1, is_exec, schema_score, non_empty, qlen)

def _composite_test_key(is_exec, schema_score, non_empty, qlen):
    return (is_exec, schema_score, non_empty, qlen)

def _write_gradescope_aliases(model_type, model_sql_path, test_sql_path,
                              model_record_path=None, test_record_path=None):
    """
    Create Gradescope-expected names alongside your standard outputs.
    model_type: "ft" or "scr"
    """
    import shutil
    # dev -> t5_ft_experiment_ec.sql / t5_scr_experiment_ec.sql
    # test -> t5_ft_experiment_ec_test.sql / t5_scr_experiment_ec_test.sql
    if model_type == "ft":
        dev_sql_alias  = os.path.join(RESULTS_DIR, "t5_ft_experiment_ec.sql")
        test_sql_alias = os.path.join(RESULTS_DIR, "t5_ft_experiment_ec_test.sql")
        dev_pkl_alias  = os.path.join(RECORDS_DIR, "t5_ft_experiment_ec.pkl")
        test_pkl_alias = os.path.join(RECORDS_DIR, "t5_ft_experiment_ec_test.pkl")
    else:
        dev_sql_alias  = os.path.join(RESULTS_DIR, "t5_scr_experiment_ec.sql")
        test_sql_alias = os.path.join(RESULTS_DIR, "t5_scr_experiment_ec_test.sql")
        dev_pkl_alias  = os.path.join(RECORDS_DIR, "t5_scr_experiment_ec.pkl")
        test_pkl_alias = os.path.join(RECORDS_DIR, "t5_scr_experiment_ec_test.pkl")

    if os.path.isfile(model_sql_path):
        shutil.copyfile(model_sql_path, dev_sql_alias)
    if test_sql_path and os.path.isfile(test_sql_path):
        shutil.copyfile(test_sql_path, test_sql_alias)

    if model_record_path and os.path.isfile(model_record_path):
        shutil.copyfile(model_record_path, dev_pkl_alias)
    if test_record_path and os.path.isfile(test_record_path):
        shutil.copyfile(test_record_path, test_pkl_alias)

# ------------------------------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------------------------------
def train(args, model, train_loader, dev_loader, optimizer, scheduler, model_type, experiment_name):
    best_f1 = -1.0
    epochs_since_improvement = 0

    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"{model_type}_experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")  # must exist

    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"[{model_type}][{experiment_name}] Epoch {epoch}: Average train loss {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"[{model_type}][{experiment_name}] Epoch {epoch}: Dev loss {eval_loss:.4f}, "
              f"Record F1 {record_f1:.4f}, Record EM {record_em:.4f}, SQL EM {sql_em:.4f}")
        print(f"[{model_type}][{experiment_name}] Epoch {epoch}: {error_rate*100:.2f}% SQL errors")

        if args.use_wandb:
            wandb.log({
                f"{model_type}/train_loss": tr_loss,
                f"{model_type}/dev_loss": eval_loss,
                f"{model_type}/dev_record_f1": record_f1,
                f"{model_type}/dev_record_em": record_em,
                f"{model_type}/dev_sql_em": sql_em,
                f"{model_type}/dev_error_rate": error_rate,
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
    # label smoothing helps
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

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
    Dev: CE loss + batched k-best generation + schema-aware rerank (+optional GT F1) + metrics.
    """
    model.eval()
    # 1) CE on dev
    ce_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
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

    gen_cfg = _make_generation_config(tok, max_new, beams, k)

    import pickle
    with open(gt_record_path, "rb") as f:
        gt_recs, gt_errs = pickle.load(f)

    sql_preds = []
    err_msgs_accum = []
    ex_idx = 0

    from utils import compute_records, compute_record_F1

    with torch.no_grad():
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
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            for i in range(B):
                cands = all_cands[i]
                recs, errs = compute_records(cands)

                # build scored candidates
                scored = []
                for q, r, e in zip(cands, recs, errs):
                    v_tbl, v_col, pen = _schema_guard_score(q.lower())
                    schema_score = 2*v_tbl + 1*v_col + pen
                    is_exec = int(e == "")
                    non_empty = int(bool(r))
                    scored.append((q, r, e, schema_score, is_exec, non_empty, len(q)))

                if args.rerank_dev_by_gt and ex_idx < len(gt_recs):
                    gt_rows = gt_recs[ex_idx]
                    best = None
                    for (q, r, e, schema_score, is_exec, non_empty, qlen) in scored:
                        f1 = compute_record_F1([gt_rows], [r])
                        key = _composite_dev_key(f1, is_exec, schema_score, non_empty, qlen)
                        if best is None or key > best[0]:
                            best = (key, q, e)
                    sql_preds.append(best[1])
                    err_msgs_accum.append(best[2])
                else:
                    best = None
                    for (q, r, e, schema_score, is_exec, non_empty, qlen) in scored:
                        key = _composite_test_key(is_exec, schema_score, non_empty, qlen)
                        if best is None or key > best[0]:
                            best = (key, q, e)
                    sql_preds.append(best[1])
                    err_msgs_accum.append(best[2])

                ex_idx += 1

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
    Test: batch-generate k candidates per example + schema-aware preference for exec/non-empty.
    """
    model.eval()
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    k = max(1, args.num_return_sequences)
    beams = max(1, args.gen_beam_size)
    max_new = max(8, args.gen_max_new_tokens)
    gen_cfg = _make_generation_config(tok, max_new, beams, k)

    from utils import compute_records

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)

            all_cands = _batch_generate_k(model, tok, encoder_input, encoder_mask, gen_cfg, k)

            for i in range(B):
                cands = all_cands[i]
                recs, errs = compute_records(cands)

                scored = []
                for q, r, e in zip(cands, recs, errs):
                    v_tbl, v_col, pen = _schema_guard_score(q.lower())
                    schema_score = 2*v_tbl + 1*v_col + pen
                    is_exec = int(e == "")
                    non_empty = int(bool(r))
                    scored.append((q, r, e, schema_score, is_exec, non_empty, len(q)))

                picked = None
                if args.prefer_executable_on_test:
                    # composite by exec/schema/non_empty/len
                    best = None
                    for (q, r, e, schema_score, is_exec, non_empty, qlen) in scored:
                        key = _composite_test_key(is_exec, schema_score, non_empty, qlen)
                        if best is None or key > best[0]:
                            best = (key, q, e)
                    picked = best[1]
                else:
                    picked = cands[0]
                sql_preds.append(picked)

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)

# ------------------------------------------------------------------------------------
# One full run (train→best dev→final dev metrics→test files→aliases)
# ------------------------------------------------------------------------------------
def run_one_experiment(args, finetune_flag: bool, experiment_name: str, write_aliases: bool):
    # Data + model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

    # Clone args but with chosen finetune flag
    class A: pass
    a = A()
    for k, v in vars(args).items():
        setattr(a, k, v)
    a.finetune = finetune_flag

    model_type = "ft" if a.finetune else "scr"

    model = initialize_model(a)
    optimizer, scheduler = initialize_optimizer_and_scheduler(a, model, len(train_loader))

    # Train
    train(a, model, train_loader, dev_loader, optimizer, scheduler, model_type, experiment_name)

    # Evaluate best on dev with full params
    model = load_model_from_checkpoint(a, best=True)
    model.eval()

    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    model_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    was_fast = a.fast_eval
    a.fast_eval = False
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        a, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    a.fast_eval = was_fast

    print(f"[{model_type}][{experiment_name}] Dev: loss={dev_loss:.4f}, "
          f"F1={dev_record_f1:.4f}, EM={dev_record_em:.4f}, SQL EM={dev_sql_em:.4f}, "
          f"err%={dev_error_rate*100:.2f}")

    # Test
    test_sql_path = os.path.join(RESULTS_DIR, f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join(RECORDS_DIR, f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(a, model, test_loader, test_sql_path, test_record_path)
    print(f"[SUBMISSION] Wrote: {test_sql_path}")
    print(f"[SUBMISSION] Wrote: {test_record_path}")

    # Gradescope aliases
    if write_aliases:
        _write_gradescope_aliases(model_type, model_sql_path, test_sql_path,
                                  model_record_path, test_record_path)
        print(f"[ALIASES] Wrote Gradescope copies for model_type={model_type}")

def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # 1) Run fine-tuned OR from-scratch (depending on --finetune)
    primary_model_type = "Fine-tuned" if args.finetune else "Scratch"
    print(f"=== Running primary experiment: {primary_model_type} | name={args.experiment_name} ===")
    run_one_experiment(args, finetune_flag=args.finetune,
                       experiment_name=args.experiment_name,
                       write_aliases=args.write_gradescope_aliases)

    # 2) Optionally also run a second (extra-credit) from-scratch experiment
    if args.also_train_scratch:
        print(f"=== Running EXTRA-CREDIT scratch experiment | name={args.scratch_experiment_name} ===")
        run_one_experiment(args, finetune_flag=False,
                           experiment_name=args.scratch_experiment_name,
                           write_aliases=args.write_gradescope_aliases)

if __name__ == "__main__":
    main()
