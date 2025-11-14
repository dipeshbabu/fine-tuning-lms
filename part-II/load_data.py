import torch
from transformers import T5TokenizerFast
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')

PAD_IDX = 0

# resolve data folder relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")

# ---- schema helpers ----


def _read_text_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _find_schema_text(data_root: str) -> str:
    """
    Try a few common filenames to locate a schema file that describes the SQLite DB.
    Falls back to empty string if not found (safe).
    """
    candidates = [
        os.path.join(data_root, "schema.txt"),
        os.path.join(data_root, "database.schema"),
        os.path.join(data_root, "flight_database.schema"),
        os.path.join(data_root, "tables.schema"),
        os.path.join(data_root, "schema"),
    ]
    for p in candidates:
        txt = _read_text_if_exists(p)
        if txt:
            return txt
    return ""


def _shift_right(labels, pad_token_id: int):
    """Create decoder inputs by shifting targets to the right (T5-style)."""
    bsz, seqlen = labels.shape
    start_tokens = torch.full((bsz, 1), pad_token_id, dtype=labels.dtype)
    shifted = torch.cat([start_tokens, labels[:, :-1]], dim=1)
    return shifted


def _load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines()]


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        """
        Provides inputs/targets for T5:
          encoder input  : 'translate to SQL given schema:\\n<SCHEMA>\\nQuestion: <NL>\\nSQL:'
          decoder target : the gold SQL
        On test split, decoder target is omitted.
        """
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_id = self.tokenizer.pad_token_id  # 0 for T5

        # load full schema text once
        self.schema_text = _find_schema_text(self.data_folder)
        # modest guard if schema is huge
        if len(self.schema_text) > 5000:
            self.schema_text = self.schema_text[:5000]

        self.items = self.process_data(data_folder, split, self.tokenizer)

    def _make_encoder_string(self, nl: str) -> str:
        """
        Compose the encoder prompt with schema context.
        """
        if self.schema_text:
            return (
                "translate to SQL given schema:\n"
                f"{self.schema_text}\n"
                f"Question: {nl}\nSQL:"
            )
        else:
            return f"translate to SQL: {nl}"

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl = _load_lines(nl_path)

        items = []
        if split == "test":
            for x in nl:
                items.append({"nl": x, "sql": None})
        else:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql = _load_lines(sql_path)
            assert len(nl) == len(sql), f"NL/SQL size mismatch for {split}"
            for x, y in zip(nl, sql):
                items.append({"nl": x, "sql": y})
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        nl = ex["nl"]
        enc_text = self._make_encoder_string(nl)

        # Allow more room so schema fits
        enc = self.tokenizer(
            enc_text, truncation=True, max_length=512, return_tensors=None
        )
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)

        if self.split == "test":
            return {"encoder_ids": enc_ids}

        sql = ex["sql"]
        # targets usually fit in <=256
        dec = self.tokenizer(sql, truncation=True,
                             max_length=256, return_tensors=None)
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

    tgt_padded = pad_sequence(
        dec_tgts, batch_first=True, padding_value=PAD_IDX)
    dec_inputs = _shift_right(tgt_padded, pad_token_id=PAD_IDX)

    init_padded = pad_sequence(
        init_inputs, batch_first=True, padding_value=PAD_IDX)

    return enc_padded, encoder_mask, dec_inputs, tgt_padded, init_padded


def test_collate_fn(batch):
    encs = [b["encoder_ids"] for b in batch]
    enc_padded = pad_sequence(encs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (enc_padded != PAD_IDX).long()
    initial_dec = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return enc_padded, encoder_mask, initial_dec


def get_dataloader(batch_size, split):
    data_folder = DATA_ROOT
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader
