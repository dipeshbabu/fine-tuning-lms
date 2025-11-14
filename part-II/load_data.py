import os
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

nltk.download('punkt')

PAD_IDX = 0

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
DATA_CLEAN_ROOT = os.path.join(SCRIPT_DIR, "data_clean")


def _choose_data_root():
    # Prefer cleaned data if present
    return DATA_CLEAN_ROOT if os.path.isdir(DATA_CLEAN_ROOT) else DATA_ROOT


def _load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines()]


def _shift_right(labels, pad_token_id: int):
    bsz, seqlen = labels.shape
    start_tokens = torch.full((bsz, 1), pad_token_id, dtype=labels.dtype)
    shifted = torch.cat([start_tokens, labels[:, :-1]], dim=1)
    return shifted


class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_id = self.tokenizer.pad_token_id  # 0 for T5
        root = _choose_data_root()
        self.items = self.process_data(root, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl = _load_lines(nl_path)

        items = []
        if split == "test":
            for x in nl:
                # schema-aware prefix helps
                nl_in = f"translate to SQL for the ATIS flight DB: {x}"
                items.append({"nl": nl_in, "sql": None})
        else:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql = _load_lines(sql_path)
            assert len(nl) == len(sql), f"NL/SQL size mismatch for {split}"
            for x, y in zip(nl, sql):
                nl_in = f"translate to SQL for the ATIS flight DB: {x}"
                items.append({"nl": nl_in, "sql": y})
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        nl = ex["nl"]

        enc = self.tokenizer(nl, truncation=True,
                             max_length=256, return_tensors=None)
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)

        if self.split == "test":
            return {"encoder_ids": enc_ids}

        sql = ex["sql"]
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
    data_folder = _choose_data_root()
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader
