import os, statistics
from transformers import T5TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def read_lines(p): 
    return [ln.strip() for ln in open(p, "r", encoding="utf-8").read().splitlines() if ln.strip()]

def stats(split):
    nl = read_lines(os.path.join(DATA, f"{split}.nl"))
    sql = None
    if split != "test":
        sql = read_lines(os.path.join(DATA, f"{split}.sql"))
        assert len(nl) == len(sql), f"Mismatch for {split}: NL={len(nl)} SQL={len(sql)}"

    nl_tok_lens = [len(tok.encode(x, add_special_tokens=False)) for x in nl]
    sql_tok_lens = []
    sql_vocab = set()
    if sql:
        for y in sql:
            ids = tok.encode(y, add_special_tokens=False)
            sql_tok_lens.append(len(ids))
            sql_vocab.update(ids)

    nl_vocab = set()
    for x in nl:
        nl_vocab.update(tok.encode(x, add_special_tokens=False))

    res = {
        "n_examples": len(nl),
        "mean_nl_len": round(statistics.mean(nl_tok_lens), 2),
        "mean_sql_len": round(statistics.mean(sql_tok_lens), 2) if sql else None,
        "nl_vocab_observed": len(nl_vocab),
        "sql_vocab_observed": len(sql_vocab) if sql else None,
    }
    return res

for sp in ["train", "dev"]:
    r = stats(sp)
    print(f"[{sp}] examples={r['n_examples']}  mean_nl_len={r['mean_nl_len']}  "
          f"mean_sql_len={r['mean_sql_len']}  nl_vocab={r['nl_vocab_observed']}  "
          f"sql_vocab={r['sql_vocab_observed']}")
