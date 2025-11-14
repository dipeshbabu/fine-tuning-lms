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
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_id = self.tokenizer.pad_token_id  # 0 for T5
        self.items = self.process_data(data_folder, split, self.tokenizer)

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
        # Small task prefix helps T5 a bit
        nl_in = f"translate to SQL: {nl}"

        enc = self.tokenizer(nl_in, truncation=True,
                             max_length=256, return_tensors=None)
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)

        if self.split == "test":
            # For test we only return encoder inputs
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
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
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
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
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

    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,            # tune to your machine; 0 on Windows if issues
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False,
    )
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
