import os, glob, statistics, argparse
from transformers import T5TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def _read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]

def _stats(nl_path, sql_path=None):
    nl = _read_lines(nl_path)
    sql = _read_lines(sql_path) if sql_path and os.path.exists(sql_path) else None
    if sql is not None:
        assert len(nl) == len(sql), f"Mismatch: NL={len(nl)} SQL={len(sql)} in {nl_path} / {sql_path}"

    nl_tok_lens = [len(tok.encode(x, add_special_tokens=False)) for x in nl]
    nl_vocab = set()
    for x in nl:
        nl_vocab.update(tok.encode(x, add_special_tokens=False))

    sql_tok_lens, sql_vocab = [], set()
    if sql is not None:
        for y in sql:
            ids = tok.encode(y, add_special_tokens=False)
            sql_tok_lens.append(len(ids))
            sql_vocab.update(ids)

    return {
        "n_examples": len(nl),
        "mean_nl_len": round(statistics.mean(nl_tok_lens), 2),
        "mean_sql_len": round(statistics.mean(sql_tok_lens), 2) if sql is not None else None,
        "nl_vocab_observed": len(nl_vocab),
        "sql_vocab_observed": len(sql_vocab) if sql is not None else None,
    }

def _latest_backup(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None

def _print_row(tag, s, latex=False):
    if not latex:
        print(f"[{tag}] examples={s['n_examples']}  mean_nl_len={s['mean_nl_len']}  "
              f"mean_sql_len={s['mean_sql_len']}  nl_vocab={s['nl_vocab_observed']}  "
              f"sql_vocab={s['sql_vocab_observed']}")
    else:
        # LaTeX line: <Split> & <#ex> & <mean NL> & <mean SQL> & <NL vocab> & <SQL vocab> \\
        ex = s['n_examples']
        mnl = f"{s['mean_nl_len']:.2f}"
        msql = f"{s['mean_sql_len']:.2f}" if s['mean_sql_len'] is not None else "--"
        nlv = s['nl_vocab_observed']
        sqlv = s['sql_vocab_observed'] if s['sql_vocab_observed'] is not None else "--"
        print(f"{tag} & {ex} & {mnl} & {msql} & {nlv} & {sqlv} \\\\")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-from-backups", action="store_true",
                    help="Also compute BEFORE stats from latest .bak files if present")
    ap.add_argument("--latex", action="store_true", help="Emit LaTeX table rows")
    args = ap.parse_args()

    for split in ["train", "dev"]:
        nl = os.path.join(DATA, f"{split}.nl")
        sql = os.path.join(DATA, f"{split}.sql")
        if not os.path.exists(nl):
            raise FileNotFoundError(nl)
        if split != "test" and not os.path.exists(sql):
            raise FileNotFoundError(sql)

        # AFTER
        s_after = _stats(nl, None if split=="test" else sql)
        _print_row(f"{split}", s_after, latex=args.latex)

        # BEFORE from latest backups, if requested and exists
        if args.before_from_backups:
            nl_bak = _latest_backup(os.path.join(DATA, f"{split}.nl.*.bak"))
            sql_bak = _latest_backup(os.path.join(DATA, f"{split}.sql.*.bak")) if split != "test" else None
            if nl_bak and (split=="test" or sql_bak):
                s_before = _stats(nl_bak, None if split=="test" else sql_bak)
                _print_row(f"{split} BEFORE", s_before, latex=args.latex)

if __name__ == "__main__":
    main()
