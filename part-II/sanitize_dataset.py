import os
import re
import sqlite3
import argparse
from typing import List, Tuple

# --- default locations: inside part-II/ ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "data")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "data_clean")


def _read_lines(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_lines(p: str, lines: List[str]):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write((ln or "").strip() + "\n")


def _first_select_block(sql: str) -> str:
    """Trim to first SELECT... and first ';', ensure trailing ';'."""
    s = (sql or "").strip()
    m = re.search(r"(select\s.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    semi = s.find(";")
    if semi != -1:
        s = s[: semi + 1]
    if s and not s.endswith(";"):
        s += ";"
    return s


def _is_valid_sql(sql: str, db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        # EXPLAIN validates syntax/table names without running full query
        cur.execute("EXPLAIN QUERY PLAN " + sql)
        conn.close()
        return True
    except Exception:
        return False


def _sanitize_split(split: str, data_dir: str, out_dir: str, db_path: str) -> Tuple[int, int, int]:
    nl_path = os.path.join(data_dir, f"{split}.nl")
    sql_path = os.path.join(data_dir, f"{split}.sql")
    has_sql = os.path.exists(sql_path)

    if not os.path.exists(nl_path):
        raise FileNotFoundError(
            f"Missing NL file for split '{split}': {nl_path}")

    nl = _read_lines(nl_path)
    sql = _read_lines(sql_path) if has_sql else []

    out_nl, out_sql = [], []
    dropped, fixed = 0, 0

    if has_sql:
        if len(nl) != len(sql):
            raise ValueError(
                f"Size mismatch for {split}: NL={len(nl)} vs SQL={len(sql)}")

        for x, y in zip(nl, sql):
            raw = (y or "").strip()
            y1 = _first_select_block(raw)

            # must start with SELECT
            if not re.search(r"^\s*select\s", y1, flags=re.IGNORECASE):
                dropped += 1
                continue

            # validate; if invalid, try the same trimmed form again (no-op) then drop
            if not _is_valid_sql(y1, db_path):
                # nothing else to salvage here; drop
                dropped += 1
                continue

            out_nl.append((x or "").strip())
            out_sql.append(y1)
            if y1 != raw:
                fixed += 1

        _write_lines(os.path.join(out_dir, f"{split}.nl"), out_nl)
        _write_lines(os.path.join(out_dir, f"{split}.sql"), out_sql)
        kept = len(out_nl)
    else:
        # test split (no SQL)
        _write_lines(os.path.join(out_dir, f"{split}.nl"), nl)
        kept = len(nl)

    return kept, fixed, dropped


def main():
    ap = argparse.ArgumentParser(
        description="Sanitize NL/SQL dataset in part-II/")
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="Input data dir (default: part-II/data)")
    ap.add_argument("--out-dir",  default=DEFAULT_OUT,
                    help="Output cleaned dir (default: part-II/data_clean)")
    ap.add_argument("--db", default=None,
                    help="Path to SQLite DB (default: <data-dir>/flight_database.db)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    db_path = args.db or os.path.join(data_dir, "flight_database.db")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found at {db_path}")

    os.makedirs(out_dir, exist_ok=True)

    # train/dev/test
    kt, ft, dt = _sanitize_split("train", data_dir, out_dir, db_path)
    print(f"[train] kept={kt} fixed={ft} dropped={dt}")
    kd, fd, dd = _sanitize_split("dev",   data_dir, out_dir, db_path)
    print(f"[dev]   kept={kd} fixed={fd} dropped={dd}")

    # test has only NL
    try:
        kx, fx, dx = _sanitize_split("test",  data_dir, out_dir, db_path)
        print(f"[test]  kept={kx}")
    except FileNotFoundError:
        print("[test]  no test split found; skipping")


if __name__ == "__main__":
    main()
#
