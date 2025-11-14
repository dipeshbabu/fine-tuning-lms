import os
import re
import time
from typing import List, Tuple

# --- Paths (resolve relative to this file) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")

# --- Utils from your project to regenerate ground-truth records ---
# We import lazily inside main() to avoid circular import during packaging
# from utils import read_queries, compute_records

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
_SELECT_DISTINCT_FUSER = re.compile(
    r"^\s*select\s*distinct", flags=re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")


def _looks_like_select(sql: str) -> bool:
    return bool(_SELECT_RE.match(sql or ""))


def _normalize_sql(sql: str) -> str:
    """Format-only cleanup; no semantic edits."""
    s = (sql or "").strip()

    # If there are accidental quotes wrapping everything, strip them.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Ensure a space in 'SELECTDISTINCT' if it ever appears fused.
    s = re.sub(r"^\s*select\s*distinct",
               "SELECT DISTINCT", s, flags=re.IGNORECASE)

    # Collapse whitespace runs
    s = _MULTI_SPACE.sub(" ", s)

    # If there are multiple semicolons, keep the first statement up to first ';'
    semi = s.find(";")
    if semi != -1:
        s = s[: semi + 1]

    # Ensure we end with a single ';'
    if not s.endswith(";"):
        s = s + ";"

    # Uppercase the leading SELECT/SELECT DISTINCT token(s) for consistency
    if s.lower().startswith("select distinct"):
        s = "SELECT DISTINCT" + s[len("select distinct"):]

    elif s.lower().startswith("select"):
        s = "SELECT" + s[len("select"):]

    return s.strip()


def _sanitize_pairs(nl: List[str], sql: List[str]) -> Tuple[List[str], List[str], dict]:
    """Drop clearly malformed lines; normalize the rest."""
    assert len(nl) == len(sql), "NL/SQL length mismatch before sanitize"

    keep_nl, keep_sql = [], []
    dropped_no_select = 0
    normalized = 0

    for x, y in zip(nl, sql):
        y_stripped = (y or "").strip()
        if not _looks_like_select(y_stripped):
            # Drop pairs without a SELECT at the start — these poison training/eval.
            dropped_no_select += 1
            continue

        y_norm = _normalize_sql(y_stripped)
        if y_norm != y_stripped:
            normalized += 1

        keep_nl.append(x.strip())
        keep_sql.append(y_norm)

    stats = {
        "original": len(nl),
        "kept": len(keep_nl),
        "dropped_no_select": dropped_no_select,
        "normalized_sql_lines": normalized,
    }
    return keep_nl, keep_sql, stats


def _sanitize_split(split: str) -> None:
    """
    For 'train' and 'dev': sanitize (NL, SQL) pairs in-place.
    For 'test': no SQL; just trim NL lines.
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
            print(
                f"[WARN] {split}: length mismatch before sanitize: NL={len(nl)} SQL={len(sql)}. Proceeding…")

        # Backup originals
        nl_bak = _backup(nl_path)
        sql_bak = _backup(sql_path)

        # Sanitize
        nl_s, sql_s, stats = _sanitize_pairs(nl, sql)
        if len(nl_s) == 0:
            raise RuntimeError(
                f"All {split} lines dropped as malformed; aborting to protect data.")

        _write_lines(nl_path, nl_s)
        _write_lines(sql_path, sql_s)

        print(f"[OK] {split}: wrote sanitized files in-place.")
        print(f"     backup NL  -> {nl_bak}")
        print(f"     backup SQL -> {sql_bak}")
        print(f"     stats: {stats}")

    elif split == "test":
        # Just tidy NL
        nl = _read_lines(nl_path)
        nl_bak = _backup(nl_path)
        nl_s = [ln.strip() for ln in nl if ln.strip()]
        _write_lines(nl_path, nl_s)
        print(f"[OK] {split}: wrote sanitized NL in-place.")
        print(f"     backup NL  -> {nl_bak}")
        print(
            f"     stats: original={len(nl)} kept={len(nl_s)} dropped_empty={len(nl) - len(nl_s)}")
    else:
        raise ValueError(f"Unknown split: {split}")


def _rebuild_dev_ground_truth_records():
    """Recompute records/ground_truth_dev.pkl from sanitized dev.sql."""
    from utils import read_queries, compute_records  # lazy import to avoid tool-order issues
    os.makedirs(RECORDS_DIR, exist_ok=True)

    dev_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_pkl_path = os.path.join(RECORDS_DIR, "ground_truth_dev.pkl")

    # Backup existing pickle if present
    if os.path.exists(gt_pkl_path):
        bak = _backup(gt_pkl_path)
        print(f"[INFO] Backed up existing ground truth records -> {bak}")

    print("[INFO] Recomputing dev ground-truth records … this may take a minute.")
    qs = read_queries(dev_sql_path)
    recs, errs = compute_records(qs)

    # Show a small summary of SQL execution errors (should be ~0 after sanitize)
    err_cnt = sum(1 for e in errs if e)
    if err_cnt:
        print(
            f"[WARN] {err_cnt} / {len(errs)} dev SQL queries still raise DB errors after sanitize.")
        for i, e in enumerate(errs):
            if e:
                print(f"  [ERR] idx={i} -> {e}")
                if i >= 10:
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
