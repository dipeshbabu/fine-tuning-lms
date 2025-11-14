import os, glob, statistics, argparse
from transformers import T5TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]

def _latest_backup(prefix: str):
    cands = sorted(glob.glob(os.path.join(DATA, f"{prefix}.*.bak")))
    return cands[-1] if cands else None

def _encode_len_list(lines):
    # tokenized length per line (no specials)
    return [len(tok.encode(x, add_special_tokens=False)) for x in lines]

def _observed_vocab(lines):
    vocab = set()
    for x in lines:
        vocab.update(tok.encode(x, add_special_tokens=False))
    return vocab

def _split_stats(nl_path, sql_path=None):
    nl = read_lines(nl_path)
    sql = read_lines(sql_path) if sql_path else None
    if sql is not None:
        assert len(nl) == len(sql), f"Mismatch: NL={len(nl)} SQL={len(sql)}"

    nl_tok_lens = _encode_len_list(nl)
    nl_vocab = _observed_vocab(nl)

    sql_tok_lens, sql_vocab = None, None
    if sql is not None:
        sql_tok_lens = _encode_len_list(sql)
        sql_vocab = _observed_vocab(sql)

    return {
        "n_examples": len(nl),
        "mean_nl_len": round(statistics.mean(nl_tok_lens), 2) if nl_tok_lens else 0.0,
        "mean_sql_len": round(statistics.mean(sql_tok_lens), 2) if sql_tok_lens else None,
        "nl_vocab_observed": len(nl_vocab),
        "sql_vocab_observed": len(sql_vocab) if sql_vocab is not None else None,
    }

def _print_row(tag, r):
    print(f"[{tag}] examples={r['n_examples']}  "
          f"mean_nl_len={r['mean_nl_len']}  "
          f"mean_sql_len={r['mean_sql_len']}  "
          f"nl_vocab={r['nl_vocab_observed']}  "
          f"sql_vocab={r['sql_vocab_observed']}")

def _latex_row(split_label, r):
    # For Table 1/2 rows
    mean_sql = r['mean_sql_len'] if r['mean_sql_len'] is not None else "-"
    sql_vocab = r['sql_vocab_observed'] if r['sql_vocab_observed'] is not None else "-"
    return (f"{split_label} & {r['n_examples']} & {r['mean_nl_len']} & {mean_sql} & "
            f"{r['nl_vocab_observed']} & {sql_vocab} \\\\")

def compute(split: str, use_backups: bool):
    # current (AFTER preprocessing)
    nl_now = os.path.join(DATA, f"{split}.nl")
    sql_now = os.path.join(DATA, f"{split}.sql") if split != "test" else None
    after_stats = _split_stats(nl_now, sql_now)

    before_stats = None
    if use_backups:
        nl_bak = _latest_backup(f"{split}.nl")
        sql_bak = _latest_backup(f"{split}.sql") if split != "test" else None
        if nl_bak and (split == "test" or sql_bak):
            before_stats = _split_stats(nl_bak, sql_bak)
    return before_stats, after_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev"],
                    help="Which splits to summarize (choices include train dev test)")
    ap.add_argument("--before-from-backups", action="store_true",
                    help="Also report stats from latest *.bak backups (pre-preprocessing)")
    ap.add_argument("--latex", action="store_true",
                    help="Print LaTeX table rows (useful for report Tables 1/2)")
    args = ap.parse_args()

    for sp in args.splits:
        before, after = compute(sp, args.before_from_backups)

        if args.before_from_backups and before:
            _print_row(f"{sp} BEFORE", before)
        _print_row(f"{sp} AFTER", after)

        if args.latex:
            if args.before_from_backups and before:
                print("% LaTeX (BEFORE)")
                print(_latex_row(sp.capitalize(), before))
            print("% LaTeX (AFTER)")
            print(_latex_row(sp.capitalize(), after))

if __name__ == "__main__":
    main()
