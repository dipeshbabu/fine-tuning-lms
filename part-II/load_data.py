import os
import re
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

# Local utils
from prompting_utils import read_schema

PAD_IDX = 0  # T5 uses 0 for pad

# Resolve data folder relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")

def _shift_right(labels, pad_token_id: int):
    """Create decoder inputs by shifting targets to the right (T5-style)."""
    bsz, seqlen = labels.shape
    start_tokens = torch.full((bsz, 1), pad_token_id, dtype=labels.dtype)
    shifted = torch.cat([start_tokens, labels[:, :-1]], dim=1)
    return shifted

def _load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines()]

# ---------- Schema hint (read once) ----------
_SCHEMA = read_schema(os.path.join(DATA_ROOT, "flight_database.schema"))

def _schema_hint(schema, ncols=3, maxtables=10):
    pieces = []
    for t, cols in schema.get("tables", {}).items():
        if not cols:
            continue
        pieces.append(f"{t}({', '.join(cols[:ncols])}...)")
    return " ; ".join(pieces[:maxtables])

_SCHEMA_HINT = _schema_hint(_SCHEMA)

# ---------- Optional light delex ----------
_NUM = re.compile(r"\b\d+\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

def _delex(text: str) -> str:
    text = _DATE.sub("<DATE>", text)
    text = _TIME.sub("<TIME>", text)
    text = _NUM.sub("<NUM>", text)
    return text

class T5Dataset(Dataset):
    def __init__(self, data_folder, split, use_delex=True):
        """
        Data processing for T5 text2SQL.
        - Tokenizer: google-t5/t5-small for both encoder & decoder
        - Adds a short schema hint to the input
        - Optionally delexicalizes numbers/dates/times in NL
        """
        self.data_folder = data_folder
        self.split = split
        self.use_delex = use_delex
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_id = self.tokenizer.pad_token_id  # 0 for T5
        self.items = self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl = _load_lines(nl_path)

        items = []
        if split == "test":
            for x in nl:
                items.append({"nl": x, "sql": None})
        else:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql = _load_lines(sql_path)
            assert len(nl) == len(sql), f"NL/SQL size mismatch for {split}: {len(nl)} vs {len(sql)}"
            for x, y in zip(nl, sql):
                items.append({"nl": x, "sql": y})
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        nl = ex["nl"]
        if self.use_delex:
            nl = _delex(nl)

        # Schema-aware prefix (compact)
        prefix = f"translate to ATIS-SQL using tables: {_SCHEMA_HINT} :: "
        nl_in = prefix + nl

        enc = self.tokenizer(nl_in, truncation=True, max_length=256, return_tensors=None)
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)

        if self.split == "test":
            return {"encoder_ids": enc_ids}

        sql = ex["sql"]
        dec = self.tokenizer(sql, truncation=True, max_length=256, return_tensors=None)
        dec_ids = torch.tensor(dec["input_ids"], dtype=torch.long)

        labels = dec_ids
        initial_dec_inp = torch.tensor([self.pad_id], dtype=torch.long)

        return {
            "encoder_ids": enc_ids,
            "decoder_labels": labels,
            "initial_decoder_input": initial_dec_inp,
        }

def normal_collate_fn(batch):
    encs = [b["encoder_ids"] for b in batch]
    dec_tgts = [b["decoder_labels"] for b in batch]
    init_inputs = [b["initial_decoder_input"] for b in batch]

    enc_padded = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (enc_padded != PAD_IDX).long()

    tgt_padded = pad_sequence(dec_tgts, batch_first=True, padding_value=PAD_IDX)
    dec_inputs = _shift_right(tgt_padded, pad_token_id=PAD_IDX)

    init_padded = pad_sequence(init_inputs, batch_first=True, padding_value=PAD_IDX)

    return enc_padded, encoder_mask, dec_inputs, tgt_padded, init_padded

def test_collate_fn(batch):
    encs = [b["encoder_ids"] for b in batch]
    enc_padded = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (enc_padded != PAD_IDX).long()
    initial_dec = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return enc_padded, encoder_mask, initial_dec

def get_dataloader(batch_size, split):
    dset = T5Dataset(DATA_ROOT, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader
