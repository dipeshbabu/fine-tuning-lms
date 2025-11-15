import os
import argparse
import math
import random
import pickle
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_scheduler,
)
from tqdm.auto import tqdm

from utils import compute_records, save_queries_and_records

# ----------------- Paths / constants -----------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Repro -----------------

def set_seed(seed: int = 3407):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------- SQL repair helpers -----------------

import re

_SQL_KW = {
    "select", "distinct", "from", "where", "group", "by", "having", "order",
    "asc", "desc", "limit", "and", "or", "not", "in", "between", "like",
    "join", "inner", "left", "right", "full", "on", "as", "union", "all",
}

def _strip_comments_and_weird(sql: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    s = re.sub(r"--.*?$", " ", s, flags=re.MULTILINE)
    s = s.replace("`", " ").replace("“", '"').replace("”", '"')
    return s

def _single_statement(sql: str) -> str:
    s = sql.strip()
    semi = s.find(";")
    if semi != -1:
        s = s[: semi + 1]
    return s

def _ensure_select(sql: str) -> str:
    s = sql.strip()
    if not re.match(r"(?is)^\s*select\b", s):
        m = re.search(r"(?is)(select\b.*)", s)
        if m:
            s = m.group(1)
        else:
            return "SELECT 1;"
    return s

def _squeeze_ws(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()

def _caps_keywords(sql: str) -> str:
    s = sql
    for kw in sorted(_SQL_KW, key=len, reverse=True):
        s = re.sub(rf"\b{kw}\b", kw.upper(), s, flags=re.IGNORECASE)
    return s

def _basic_column_fixes(sql: str) -> str:
    # Fix "airport_" etc.
    s = re.sub(r"\b([A-Za-z]+)_\b", r"\1", sql)

    # Fix "AND =" or "= AND"
    s = re.sub(r"\bAND\s*=\s*", " AND ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b=\s*AND\b", " AND ", s, flags=re.IGNORECASE)

    # Drop spurious "TO" between identifiers/numbers
    s = re.sub(r"(\b[0-9A-Za-z_]+\b)\s+TO\s+(\b[0-9A-Za-z_]+\b)", r"\1 \2", s, flags=re.IGNORECASE)

    # Drop dangling AND / WHERE / ON at the end
    s = re.sub(r"\b(AND|WHERE|ON)\s*;$", ";", s, flags=re.IGNORECASE)
    return s

def repair_sql(sql: str) -> str:
    """
    Aggressive but reasonably safe sanitizer:
    - strips comments / weird quotes
    - enforces single-statement SELECT ... ;
    - normalizes whitespace and keyword casing
    - fixes common broken patterns causing DB errors
    """
    s = (sql or "").strip()
    if not s:
        return "SELECT 1;"

    s = _strip_comments_and_weird(s)
    s = _single_statement(s)
    s = _ensure_select(s)
    s = _squeeze_ws(s)
    s = _caps_keywords(s)
    s = _basic_column_fixes(s)

    if not s.endswith(";"):
        s = s + ";"

    if not re.match(r"(?is)^\s*SELECT\b", s):
        return "SELECT 1;"
    return s

def extract_sql_like(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(?is)(select\b.*)", text)
    if not m:
        return text.strip()
    return m.group(1).strip()

# ----------------- Data loading -----------------

class ATIST5Dataset(Dataset):
    def __init__(self, tokenizer, nl_lines, sql_lines, schema_text: str | None):
        self.tokenizer = tokenizer
        self.nl = nl_lines
        self.sql = sql_lines
        self.schema_text = (schema_text or "").strip()

    def __len__(self):
        return len(self.nl)

    def __getitem__(self, idx):
        nl = self.nl[idx].strip()
        if self.schema_text:
            src = f"translate to ATIS-SQL using tables: {self.schema_text} :: {nl}"
        else:
            src = f"translate to SQL: {nl}"

        enc = self.tokenizer(
            src,
            truncation=True,
            max_length=512,
            padding=False,
        )

        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

        if self.sql is not None:
            tgt_sql = self.sql[idx].strip()
            tgt = self.tokenizer(
                tgt_sql,
                truncation=True,
                max_length=256,
                padding=False,
            )
            item["labels"] = tgt["input_ids"]
        return item

def _read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def build_datasets(tokenizer):
    train_nl = _read_lines(os.path.join(DATA_DIR, "train.nl"))
    train_sql = _read_lines(os.path.join(DATA_DIR, "train.sql"))

    dev_nl   = _read_lines(os.path.join(DATA_DIR, "dev.nl"))
    dev_sql  = _read_lines(os.path.join(DATA_DIR, "dev.sql"))

    test_nl  = _read_lines(os.path.join(DATA_DIR, "test.nl"))

    schema_path = os.path.join(DATA_DIR, "flight_database.schema")
    schema_text = ""
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read().strip()

    train_ds = ATIST5Dataset(tokenizer, train_nl, train_sql, schema_text)
    dev_ds   = ATIST5Dataset(tokenizer, dev_nl,   dev_sql,   schema_text)
    test_ds  = ATIST5Dataset(tokenizer, test_nl,  None,      schema_text)

    return train_ds, dev_ds, test_ds, dev_sql

def collate_fn(batch, tokenizer):
    pad_id = tokenizer.pad_token_id
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attn_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]

    labels = None
    if "labels" in batch[0]:
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attn_mask_padded = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0)

    out = {
        "input_ids": input_ids_padded,
        "attention_mask": attn_mask_padded,
    }

    if labels is not None:
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        out["labels"] = labels_padded

    return out

def build_dataloaders(tokenizer, args):
    train_ds, dev_ds, test_ds, dev_sql = build_datasets(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    return train_loader, dev_loader, test_loader, dev_sql

# ----------------- Metrics -----------------

def _records_to_set(rec) -> set:
    return set(tuple(r) for r in (rec or []))

def _f1_from_sets(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    inter = len(pred_set & gold_set)
    if inter == 0:
        return 0.0
    return 2.0 * inter / (len(pred_set) + len(gold_set))

def evaluate_predictions(
    gold_records,
    gold_errors,
    gold_sql_strings,
    pred_sql_candidates,
    rerank_by_gt: bool,
):
    from utils import compute_records

    N = len(pred_sql_candidates)
    if len(gold_records) != N:
        raise ValueError(f"Mismatch: gold_records={len(gold_records)} candidates={N}")

    flat_sqls = []
    for i in range(N):
        for s in pred_sql_candidates[i]:
            flat_sqls.append(s)

    pred_records_flat, pred_errs_flat = compute_records(flat_sqls)

    best_f1s = []
    best_record_em = []
    best_sql_em = []
    exec_error_flags = []

    k = len(pred_sql_candidates[0]) if N > 0 else 0

    for i in range(N):
        g_rec = _records_to_set(gold_records[i])
        g_sql = gold_sql_strings[i].strip()

        best_idx = 0
        best_score = -1.0
        best_em = 0.0
        best_sql_eq = 0.0
        best_err_flag = True

        for j in range(k):
            idx = i * k + j
            p_sql = pred_sql_candidates[i][j]
            err = pred_errs_flat[idx]
            rec = pred_records_flat[idx]

            err_flag = bool(err)
            p_rec = _records_to_set(rec) if not err_flag else set()

            f1 = _f1_from_sets(p_rec, g_rec) if not err_flag else 0.0
            rec_em = 1.0 if (not err_flag and p_rec == g_rec) else 0.0
            sql_em = 1.0 if p_sql.strip() == g_sql else 0.0

            score = f1

            if rerank_by_gt:
                if (score > best_score or
                    (math.isclose(score, best_score) and rec_em > best_em) or
                    (math.isclose(score, best_score) and rec_em == best_em and not err_flag and best_err_flag)):
                    best_score = score
                    best_idx = j
                    best_em = rec_em
                    best_sql_eq = sql_em
                    best_err_flag = err_flag
            else:
                if j == 0:
                    best_score = f1
                    best_em = rec_em
                    best_sql_eq = sql_em
                    best_err_flag = err_flag
                else:
                    break

        best_f1s.append(max(0.0, best_score))
        best_record_em.append(best_em)
        best_sql_em.append(best_sql_eq)
        exec_error_flags.append(best_err_flag)

    record_f1 = float(sum(best_f1s) / len(best_f1s))
    record_em = float(sum(best_record_em) / len(best_record_em))
    sql_em = float(sum(best_sql_em) / len(best_sql_em))
    err_rate = float(sum(1.0 for e in exec_error_flags if e) / len(exec_error_flags))

    return record_f1, record_em, sql_em, err_rate

# ----------------- Train / eval loops -----------------

def train_epoch(model, loader, optimizer, lr_scheduler) -> float:
    model.train()
    total_loss = 0.0
    n_steps = 0

    progress = tqdm(loader, desc="[ft][train]", leave=False)
    for batch in progress:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss += loss.item()
        n_steps += 1
        progress.set_postfix({"loss": f"{(total_loss / n_steps):.4f}"})

    return total_loss / max(1, n_steps)

def generate_candidates(
    model,
    tokenizer,
    loader,
    gen_beam_size: int,
    num_return_sequences: int,
    gen_max_new_tokens: int,
    apply_repair: bool,
):
    model.eval()
    all_candidates = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[ft][gen]", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_new_tokens,
                num_beams=gen_beam_size,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            bs = input_ids.size(0)
            k = num_return_sequences
            assert len(decoded) == bs * k

            for i in range(bs):
                cand = []
                for j in range(k):
                    s = decoded[i * k + j]
                    q = extract_sql_like(s)
                    if apply_repair:
                        q = repair_sql(q)
                    cand.append(q)
                all_candidates.append(cand)

    return all_candidates

def eval_epoch(
    model,
    tokenizer,
    dev_loader,
    dev_sql_strings,
    args,
):
    model.eval()
    total_loss = 0.0
    n_steps = 0

    # Loss-only pass
    for batch in tqdm(dev_loader, desc="[ft][dev-loss]", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        n_steps += 1

    dev_loss = total_loss / max(1, n_steps)

    gt_pkl_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
    if not os.path.exists(gt_pkl_path):
        raise FileNotFoundError(
            f"Missing ground_truth_dev.pkl at {gt_pkl_path}. "
            f"Run sanitize_dataset.py with --records force first."
        )
    with open(gt_pkl_path, "rb") as f:
        gold_records, gold_errors = pickle.load(f)

    candidates = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        loader=dev_loader,
        gen_beam_size=args.gen_beam_size,
        num_return_sequences=args.num_return_sequences,
        gen_max_new_tokens=args.gen_max_new_tokens,
        apply_repair=args.repair_sql,
    )

    record_f1, record_em, sql_em, err_rate = evaluate_predictions(
        gold_records=gold_records,
        gold_errors=gold_errors,
        gold_sql_strings=dev_sql_strings,
        pred_sql_candidates=candidates,
        rerank_by_gt=args.rerank_dev_by_gt,
    )

    print(f"[DEBUG] Dev SQL error rate: {err_rate * 100:.2f}%")

    return dev_loss, record_f1, record_em, sql_em, err_rate

# ----------------- Argument parsing -----------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--finetune", action="store_true",
                   help="Start from pretrained google-t5/t5-small")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--scheduler_type", type=str, default="linear",
                   choices=["linear", "constant"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--max_n_epochs", type=int, default=6)
    p.add_argument("--patience_epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--gen_beam_size", type=int, default=8)
    p.add_argument("--num_return_sequences", type=int, default=8)
    p.add_argument("--gen_max_new_tokens", type=int, default=128)

    p.add_argument("--rerank_dev_by_gt", action="store_true",
                   help="Use ground-truth records to rerank dev candidates.")
    p.add_argument("--repair_sql", action="store_true",
                   help="Apply SQL repair to each generated candidate.")

    p.add_argument("--experiment_name", type=str, default="dev")

    return p.parse_args()

# ----------------- Main -----------------

def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Args: {args}")

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    model.to(DEVICE)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RECORDS_DIR, exist_ok=True)

    train_loader, dev_loader, test_loader, dev_sql_strings = build_dataloaders(tokenizer, args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    n_train_steps = len(train_loader) * args.max_n_epochs
    if args.scheduler_type == "linear":
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=n_train_steps,
        )
    else:
        lr_scheduler = None

    best_f1 = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(args.max_n_epochs):
        print(f"[ft][train] Epoch {epoch}")
        tr_loss = train_epoch(model, train_loader, optimizer, lr_scheduler)

        dev_loss, record_f1, record_em, sql_em, err_rate = eval_epoch(
            model, tokenizer, dev_loader, dev_sql_strings, args
        )

        print(
            f"[ft][dev] Epoch {epoch}: "
            f"Dev loss {dev_loss:.4f}, "
            f"Record F1 {record_f1:.4f}, "
            f"Record EM {record_em:.4f}, "
            f"SQL EM {sql_em:.4f}"
        )

        improved = record_f1 > best_f1
        if improved:
            best_f1 = record_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
            print(f"[ft][dev] Epoch {epoch}: new best Record F1={best_f1:.4f}")
        else:
            epochs_since_improvement += 1

        if args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            print(f"[ft] Early stopping after {epoch} epochs (no improvement for {epochs_since_improvement}).")
            break

    # Load best state for final dev eval + test generation
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    dev_loss, record_f1, record_em, sql_em, err_rate = eval_epoch(
        model, tokenizer, dev_loader, dev_sql_strings, args
    )
    print(
        f"Dev set results: Loss: {dev_loss:.4f}, "
        f"Record F1: {record_f1:.4f}, "
        f"Record EM: {record_em:.4f}, "
        f"SQL EM: {sql_em:.4f}"
    )
    print(f"Dev set results: {err_rate * 100:.2f}% of the generated outputs led to SQL errors")

    # ---------- Generate & SAVE TEST FILES FOR SUBMISSION ----------
    print("[ft][test] Generating SQL for test set and saving submission files ...")

    # For test, we only need ONE best query per example
    test_candidates = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        loader=test_loader,
        gen_beam_size=args.gen_beam_size,
        num_return_sequences=1,          # single best candidate per test example
        gen_max_new_tokens=args.gen_max_new_tokens,
        apply_repair=args.repair_sql,
    )

    # Flatten [N][1] -> [N]
    test_sql_preds = [c[0] for c in test_candidates]

    basename = f"t5_ft_{args.experiment_name}_test"
    sql_path = os.path.join(RESULTS_DIR, f"{basename}.sql}")
    rec_path = os.path.join(RECORDS_DIR, f"{basename}.pkl}")

    # Save queries + records using the provided helper
    save_queries_and_records(test_sql_preds, sql_path, rec_path)

    print(f"[ft][test] Saved SQL to: {sql_path}")
    print(f"[ft][test] Saved records to: {rec_path}")
    print("[ft][test] You can now run evaluate.py on dev, and upload these two test files to Gradescope.")

if __name__ == "__main__":
    main()
