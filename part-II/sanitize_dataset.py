import os
import re
import time
from typing import List, Tuple

# --- Config ---
STRICT_MODE = True  # set False to keep erroring queries but log them

# --- Paths ---
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
DB_PATH     = os.path.join(DATA_DIR, "flight_database.db")  # utils uses this default too

# ---------- IO helpers ----------
def _read_lines(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _write_lines(p: str, lines: List[str]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def _backup(path: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = f"{path}.{ts}.bak"
    with open(path, "rb") as src, open(bak, "wb") as dst:
        dst.write(src.read())
    return bak

# ---------- SQL normalization / filters ----------
_SELECT_RE = re.compile(r"^\s*select\b", flags=re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")

def _looks_like_select(sql: str) -> bool:
    return bool(_SELECT_RE.match(sql or ""))

def _normalize_sql(sql: str) -> str:
    """Format-only cleanup; no semantic edits."""
    s = (sql or "").strip()

    # Strip paired quotes around entire line
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Fix fused "selectdistinct" if it happens
    s = re.sub(r"^\s*select\s*distinct", "SELECT DISTINCT", s, flags=re.IGNORECASE)

    # Collapse whitespace runs
    s = _MULTI_SPACE.sub(" ", s)

    # Keep only first statement if multiple ';'
    semi = s.find(";")
    if semi != -1:
        s = s[: semi + 1]

    # Ensure single trailing ';'
    if not s.endswith(";"):
        s = s + ";"

    # Normalize leading SELECT tokens
    if s.lower().startswith("select distinct"):
        s = "SELECT DISTINCT" + s[len("select distinct"):]
    elif s.lower().startswith("select"):
        s = "SELECT" + s[len("select"):]

    return s.strip()

def _sanitize_pairs(nl: List[str], sql: List[str]) -> Tuple[List[str], List[str], dict]:
    """Drop clearly malformed lines; normalize the rest (syntax-level only)."""
    keep_nl, keep_sql = [], []
    dropped_no_select = 0
    normalized = 0

    for x, y in zip(nl, sql):
        y_stripped = (y or "").strip()
        if not _looks_like_select(y_stripped):
            dropped_no_select += 1
            continue
        y_norm = _normalize_sql(y_stripped)
        if y_norm != y_stripped:
            normalized += 1
        keep_nl.append((x or "").strip())
        keep_sql.append(y_norm)

    stats = {
        "original": len(nl),
        "kept_after_syntax": len(keep_nl),
        "dropped_no_select": dropped_no_select,
        "normalized_sql_lines": normalized,
    }
    return keep_nl, keep_sql, stats

# ---------- DB validation ----------
def _validate_against_db(sql_list: List[str]) -> List[str]:
    """
    Execute each SQL against the SQLite DB and return error message list ("" means success).
    Uses the same logic as utils.compute_records to ensure consistency.
    """
    import sqlite3
    errors = []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for q in sql_list:
        try:
            cur.execute(q)
            _ = cur.fetchall()
            errors.append("")
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    conn.close()
    return errors

def _apply_db_validation(nl: List[str], sql: List[str], split: str) -> Tuple[List[str], List[str], dict]:
    """
    Run SQLs; drop pairs with DB errors if STRICT_MODE=True, otherwise keep and report.
    """
    errs = _validate_against_db(sql)
    bad_idx = [i for i, e in enumerate(errs) if e]

    if not bad_idx:
        return nl, sql, {"db_bad": 0, "kept": len(sql), "dropped_db_error": 0}

    print(f"[WARN] {split}: {len(bad_idx)} / {len(sql)} queries raise DB errors.")
    for i in bad_idx[:10]:
        print(f"  [ERR] idx={i} -> {errs[i]}")
    if len(bad_idx) > 10:
        print(f"  ... {len(bad_idx)-10} more")

    if STRICT_MODE:
        keep_nl, keep_sql = [], []
        dropped = 0
        bad_set = set(bad_idx)
        for i, (x, y) in enumerate(zip(nl, sql)):
            if i in bad_set:
                dropped += 1
                continue
            keep_nl.append(x)
            keep_sql.append(y)
        stats = {"db_bad": len(bad_idx), "kept": len(keep_sql), "dropped_db_error": dropped}
        return keep_nl, keep_sql, stats
    else:
        stats = {"db_bad": len(bad_idx), "kept": len(sql), "dropped_db_error": 0}
        return nl, sql, stats

# ---------- Per-split processing ----------
def _sanitize_split(split: str) -> None:
    """
    For 'train' and 'dev': sanitize (NL, SQL) pairs in-place + DB validation.
    For 'test': no SQL; just tidy NL.
    """
    nl_path = os.path.join(DATA_DIR, f"{split}.nl")
    if not os.path.exists(nl_path):
        raise FileNotFoundError(f"Missing file: {nl_path}")

    if split in ("train", "dev"):
        sql_path = os.path.join(DATA_DIR, f"{split}.sql")
        if not os.path.exists(sql_path):
            raise FileNotFoundError(f"Missing file: {sql_path}")

        nl = _read_lines(nl_path)
        sql = _read_lines(sql_path)

        if len(nl) != len(sql):
            print(f"[WARN] {split}: length mismatch before sanitize: NL={len(nl)} SQL={len(sql)}")

        # Backups
        nl_bak  = _backup(nl_path)
        sql_bak = _backup(sql_path)

        # 1) Syntax-level cleanup
        nl_syn, sql_syn, s_stats = _sanitize_pairs(nl, sql)

        # 2) DB-level validation
        nl_ok, sql_ok, d_stats = _apply_db_validation(nl_syn, sql_syn, split=split)

        if len(nl_ok) == 0:
            raise RuntimeError(f"All {split} lines eliminated after validation; aborting to protect data.")

        # Write back
        _write_lines(nl_path, nl_ok)
        _write_lines(sql_path, sql_ok)

        print(f"[OK] {split}: wrote sanitized files in-place.")
        print(f"     backup NL  -> {nl_bak}")
        print(f"     backup SQL -> {sql_bak}")
        print(f"     syntax stats: {s_stats}")
        print(f"     db stats:     {d_stats}")

    elif split == "test":
        # Just tidy NL
        nl = _read_lines(nl_path)
        nl_bak = _backup(nl_path)
        nl_s = [ln.strip() for ln in nl if ln.strip()]
        _write_lines(nl_path, nl_s)
        print(f"[OK] {split}: wrote sanitized NL in-place.")
        print(f"     backup NL  -> {nl_bak}")
        print(f"     stats: original={len(nl)} kept={len(nl_s)} dropped_empty={len(nl) - len(nl_s)}")
    else:
        raise ValueError(f"Unknown split: {split}")

# ---------- Rebuild dev ground-truth ----------
def _rebuild_dev_ground_truth_records():
    from utils import read_queries, compute_records
    os.makedirs(RECORDS_DIR, exist_ok=True)

    dev_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_pkl_path  = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")

    if os.path.exists(gt_pkl_path):
        bak = _backup(gt_pkl_path)
        print(f"[INFO] Backed up existing ground truth records -> {bak}")

    print("[INFO] Recomputing dev ground-truth records …")
    qs = read_queries(dev_sql_path)
    recs, errs = compute_records(qs)
    err_cnt = sum(1 for e in errs if e)
    if err_cnt:
        print(f"[WARN] After sanitize, {err_cnt} / {len(errs)} dev SQL still raise DB errors.")
        for i, e in enumerate(errs):
            if e:
                print(f"  [ERR] idx={i} -> {e}")
                if i >= 15:
                    break
    else:
        print("[OK] All dev SQL executed successfully.")

    import pickle
    with open(gt_pkl_path, "wb") as f:
        pickle.dump((recs, errs), f)
    print(f"[OK] Saved rebuilt ground truth records -> {gt_pkl_path}")

def main():
    print(f"[INFO] Sanitizing in-place under: {DATA_DIR}")
    _sanitize_split("train")
    _sanitize_split("dev")
    _sanitize_split("test")
    _rebuild_dev_ground_truth_records()
    print("[DONE] In-place sanitize complete.")

if __name__ == "__main__":
    main()
