import os
import re
import time
import argparse
from typing import List, Tuple

# ---------- Defaults / Paths ----------
STRICT_MODE_DEFAULT = True  # can override via --strict / --no-strict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
DB_PATH     = os.path.join(DATA_DIR, "flight_database.db")  # used by utils.compute_records

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

    # Fix fused "selectdistinct" and normalize "select distinct"
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

    # Normalize leading SELECT tokens casing
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

# ---------- Parallel DB validation (via utils.compute_records) ----------
def _validate_via_utils(sql_list: List[str], threads: int, pq_timeout: float, tot_timeout: float) -> list:
    """
    Run SQLs through utils.compute_records (parallel, per-query timeouts).
    Returns the list of error messages ("" on success).
    """
    # Set env knobs for utils.compute_records
    os.environ["EVAL_THREADS"] = str(threads)
    os.environ["PER_QUERY_TIMEOUT_SECS"] = str(pq_timeout)
    os.environ["EVAL_TOTAL_TIMEOUT_SECS"] = str(tot_timeout)
    # Optional: allow many failures without early break in sanitize
    os.environ["MAX_ERROR_FRACTION"] = "1.0"

    from utils import compute_records
    _, errs = compute_records(sql_list)
    return errs

def _apply_db_validation(
    nl: List[str],
    sql: List[str],
    split: str,
    strict_mode: bool,
    validate_mode: str,
    threads: int,
    pq_timeout: float,
    tot_timeout: float,
    sample_k: int | None = None,
) -> Tuple[List[str], List[str], dict]:
    """
    Validate SQLs on DB depending on validate_mode.
      - "none": skip DB validation
      - "all": validate full split
      - "dev-only": only validate if split=="dev"
      - "sample": validate a head subset (first K queries)
    Returns possibly filtered (nl, sql) and stats.
    """
    if validate_mode == "none":
        return nl, sql, {"db_validation": "skipped", "kept": len(sql), "dropped_db_error": 0}

    if validate_mode == "dev-only" and split != "dev":
        return nl, sql, {"db_validation": "skipped_non_dev", "kept": len(sql), "dropped_db_error": 0}

    to_validate_idx = list(range(len(sql)))
    if validate_mode == "sample":
        k = min(sample_k or 128, len(sql))
        to_validate_idx = list(range(k))

    sql_to_check = [sql[i] for i in to_validate_idx]
    errs = _validate_via_utils(sql_to_check, threads=threads, pq_timeout=pq_timeout, tot_timeout=tot_timeout)

    bad_local = [i for i, e in enumerate(errs) if e]
    if not bad_local:
        return nl, sql, {"db_bad": 0, "checked": len(sql_to_check), "kept": len(sql), "dropped_db_error": 0}

    # Map local indices back to global
    bad_global = set(to_validate_idx[i] for i in bad_local)

    print(f"[WARN] {split}: {len(bad_global)} / {len(sql_to_check)} checked queries raise DB errors.")
    for j, gi in enumerate(sorted(bad_global)):
        if j >= 10: break
        print(f"  [ERR] idx={gi} -> (hidden message; see sanitizer logs in compute_records phase)")

    if strict_mode:
        keep_nl, keep_sql = [], []
        dropped = 0
        for i, (x, y) in enumerate(zip(nl, sql)):
            if i in bad_global:
                dropped += 1
                continue
            keep_nl.append(x)
            keep_sql.append(y)
        stats = {"db_bad": len(bad_global), "checked": len(sql_to_check), "kept": len(keep_sql), "dropped_db_error": dropped}
        return keep_nl, keep_sql, stats
    else:
        stats = {"db_bad": len(bad_global), "checked": len(sql_to_check), "kept": len(sql), "dropped_db_error": 0}
        return nl, sql, stats

# ---------- Per-split processing ----------
def _sanitize_split(
    split: str,
    strict_mode: bool,
    validate_mode: str,
    threads: int,
    pq_timeout: float,
    tot_timeout: float,
    sample_k: int | None,
) -> None:
    """
    For 'train' and 'dev': sanitize (NL, SQL) pairs in-place + optional DB validation.
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

        # 2) DB-level validation (parallel via utils)
        nl_ok, sql_ok, d_stats = _apply_db_validation(
            nl_syn, sql_syn, split=split, strict_mode=strict_mode,
            validate_mode=validate_mode, threads=threads,
            pq_timeout=pq_timeout, tot_timeout=tot_timeout, sample_k=sample_k
        )

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
def _rebuild_dev_ground_truth_records(threads: int, pq_timeout: float, tot_timeout: float):
    from utils import read_queries, compute_records
    os.makedirs(RECORDS_DIR, exist_ok=True)

    dev_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_pkl_path  = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")

    if os.path.exists(gt_pkl_path):
        bak = _backup(gt_pkl_path)
        print(f"[INFO] Backed up existing ground truth records -> {bak}")

    # Pass env knobs to compute_records for speed/robustness
    os.environ["EVAL_THREADS"] = str(threads)
    os.environ["PER_QUERY_TIMEOUT_SECS"] = str(pq_timeout)
    os.environ["EVAL_TOTAL_TIMEOUT_SECS"] = str(tot_timeout)
    os.environ["MAX_ERROR_FRACTION"] = "1.0"

    print("[INFO] Recomputing dev ground-truth records …")
    qs = read_queries(dev_sql_path)
    recs, errs = compute_records(qs)
    err_cnt = sum(1 for e in errs if e)
    if err_cnt:
        print(f"[WARN] After sanitize, {err_cnt} / {len(errs)} dev SQL still raise DB errors (kept for parity).")
    else:
        print("[OK] All dev SQL executed successfully.")

    import pickle
    with open(gt_pkl_path, "wb") as f:
        pickle.dump((recs, errs), f)
    print(f"[OK] Saved rebuilt ground truth records -> {gt_pkl_path}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Sanitize NL/SQL dataset in-place, with optional DB validation and records rebuild.")
    p.add_argument("--inplace", action="store_true", help="(kept for compatibility; sanitize is always in-place)")
    p.add_argument("--validate", type=str, default="dev-only",
                   choices=["none", "all", "dev-only", "sample"],
                   help="DB validation scope. 'sample' validates only a head subset.")
    p.add_argument("--sample-k", type=int, default=128, help="K examples to check when --validate sample")
    p.add_argument("--records", type=str, default="auto",
                   choices=["auto", "force", "skip"],
                   help="Rebuild records/ground_truth_dev.pkl: auto (default), force, or skip.")
    p.add_argument("--strict", dest="strict", action="store_true", default=STRICT_MODE_DEFAULT,
                   help="Drop examples that fail DB validation.")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="Keep examples even if DB validation fails (just logs).")

    # performance knobs
    p.add_argument("--threads", type=int, default=int(os.getenv("EVAL_THREADS", "8")))
    p.add_argument("--per-query-timeout", type=float, default=float(os.getenv("PER_QUERY_TIMEOUT_SECS", "2.5")))
    p.add_argument("--total-timeout", type=float, default=float(os.getenv("EVAL_TOTAL_TIMEOUT_SECS", "90")))
    return p.parse_args()

def main():
    args = parse_args()

    print(f"[INFO] Sanitizing in-place under: {DATA_DIR}")
    # Train/dev/test
    _sanitize_split("train", strict_mode=args.strict, validate_mode=args.validate,
                    threads=args.threads, pq_timeout=args.per_query_timeout,
                    tot_timeout=args.total_timeout, sample_k=args.sample_k)
    _sanitize_split("dev",   strict_mode=args.strict, validate_mode=args.validate,
                    threads=args.threads, pq_timeout=args.per_query_timeout,
                    tot_timeout=args.total_timeout, sample_k=args.sample_k)
    _sanitize_split("test",  strict_mode=args.strict, validate_mode="none",
                    threads=args.threads, pq_timeout=args.per_query_timeout,
                    tot_timeout=args.total_timeout, sample_k=args.sample_k)

    # Records rebuild policy
    if args.records == "skip":
        print("[INFO] Skipping ground truth record rebuild per --records=skip.")
    elif args.records == "force":
        _rebuild_dev_ground_truth_records(args.threads, args.per_query_timeout, args.total_timeout)
    else:
        # auto: rebuild if file missing
        gt_pkl_path  = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")
        if not os.path.exists(gt_pkl_path):
            _rebuild_dev_ground_truth_records(args.threads, args.per_query_timeout, args.total_timeout)
        else:
            print("[INFO] ground_truth_dev.pkl exists; not rebuilding (use --records force to override).")

    print("[DONE] In-place sanitize complete.")

if __name__ == "__main__":
    main()
