import os
import re
import sqlite3
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA = os.path.join(ROOT, "data")
DATA_CLEAN = os.path.join(ROOT, "data_clean")
DB_PATH = os.path.join(ROOT, "data", "flight_database.db")

os.makedirs(DATA_CLEAN, exist_ok=True)


def _read_lines(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _write_lines(p: str, lines: List[str]):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write((ln or "").strip() + "\n")


def _first_select_block(sql: str) -> str:
    s = (sql or "").strip()
    m = re.search(r"(select\s.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    # cut at first semicolon if any
    semi = s.find(";")
    if semi != -1:
        s = s[: semi + 1]
    # normalize: ensure a single tail semicolon for teaching a clean stop
    if not s.endswith(";"):
        s += ";"
    return s


def _is_valid_sql(sql: str, db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("EXPLAIN QUERY PLAN " + sql)
        conn.close()
        return True
    except Exception:
        return False


def _sanitize_split(split: str) -> Tuple[int, int, int]:
    nl_path = os.path.join(DATA, f"{split}.nl")
    sql_path = os.path.join(DATA, f"{split}.sql")
    has_sql = os.path.exists(sql_path)

    nl = _read_lines(nl_path)
    sql = _read_lines(sql_path) if has_sql else []

    out_nl, out_sql = [], []
    dropped, fixed = 0, 0

    if has_sql:
        assert len(nl) == len(
            sql), f"size mismatch for {split}: {len(nl)} vs {len(sql)}"
        for x, y in zip(nl, sql):
            raw = y
            y = _first_select_block(y)

            if not re.search(r"^\s*select\s", y, flags=re.IGNORECASE):
                dropped += 1
                continue

            if not _is_valid_sql(y, DB_PATH):
                y2 = _first_select_block(raw)
                if not _is_valid_sql(y2, DB_PATH):
                    dropped += 1
                    continue
                y = y2
                fixed += 1

            out_nl.append((x or "").strip())
            out_sql.append((y or "").strip())

        _write_lines(os.path.join(DATA_CLEAN, f"{split}.nl"), out_nl)
        _write_lines(os.path.join(DATA_CLEAN, f"{split}.sql"), out_sql)
        kept = len(out_nl)
    else:
        _write_lines(os.path.join(DATA_CLEAN, f"{split}.nl"), nl)
        kept = len(nl)

    return kept, fixed, dropped


def main():
    kt, ft, dt = _sanitize_split("train")
    kd, fd, dd = _sanitize_split("dev")
    kx, fx, dx = _sanitize_split("test")
    print(f"[train] kept={kt} fixed={ft} dropped={dt}")
    print(f"[dev]   kept={kd} fixed={fd} dropped={dd}")
    print(f"[test]  kept={kx}")


if __name__ == "__main__":
    main()
