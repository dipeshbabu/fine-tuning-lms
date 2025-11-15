import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, use_schema=False):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.use_schema = use_schema
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        # Load schema if enabled
        schema_prefix = ""
        if self.use_schema:
            from schema_info import COMPACT_SCHEMA
            schema_prefix = COMPACT_SCHEMA + " | "
        
        processed_data = []
        
        # For test set, we don't have SQL queries
        if split == 'test':
            for nl in nl_queries:
                # Add schema prefix if enabled
                input_text = schema_prefix + nl
                
                # Tokenize the natural language input
                encoder_inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_input_ids = encoder_inputs['input_ids'].squeeze(0)
                
                # Use extra_id_0 as the beginning of sentence token for decoder
                decoder_bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
                
                processed_data.append({
                    'encoder_input_ids': encoder_input_ids,
                    'decoder_bos_token_id': decoder_bos_token_id
                })
        else:
            # For train and dev sets, we have both NL and SQL
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_path)
            
            assert len(nl_queries) == len(sql_queries), f"Mismatch in data sizes for {split}"
            
            for nl, sql in zip(nl_queries, sql_queries):
                # Add schema prefix if enabled
                input_text = schema_prefix + nl
                
                # Tokenize the natural language input
                encoder_inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_input_ids = encoder_inputs['input_ids'].squeeze(0)
                
                # Tokenize the SQL output
                decoder_inputs = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
                decoder_input_ids = decoder_inputs['input_ids'].squeeze(0)
                
                # Use extra_id_0 as the beginning of sentence token for decoder
                decoder_bos_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
                
                processed_data.append({
                    'encoder_input_ids': encoder_input_ids,
                    'decoder_input_ids': decoder_input_ids,
                    'decoder_bos_token_id': decoder_bos_token_id
                })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    # Extract encoder and decoder sequences
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    decoder_bos_tokens = [item['decoder_bos_token_id'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Prepare decoder inputs: prepend BOS token to decoder inputs
    decoder_inputs_with_bos = []
    for bos_token, dec_ids in zip(decoder_bos_tokens, decoder_input_ids):
        # Prepend BOS token to decoder input
        decoder_input = torch.cat([torch.tensor([bos_token]), dec_ids])
        decoder_inputs_with_bos.append(decoder_input)
    
    # Pad decoder inputs (with BOS prepended)
    decoder_inputs = pad_sequence(decoder_inputs_with_bos, batch_first=True, padding_value=PAD_IDX)
    
    # Decoder targets are the original decoder input ids (without BOS) padded
    # This is shifted by one position - targets are what decoder should predict at each step
    decoder_targets_list = [item['decoder_input_ids'] for item in batch]
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)
    
    # Initial decoder inputs for evaluation (just the BOS tokens)
    initial_decoder_inputs = torch.tensor(decoder_bos_tokens).unsqueeze(1)
    
    return encoder_ids, encoder_mask, decoder_inputs[:, :-1], decoder_targets, initial_decoder_inputs

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
    # Extract encoder sequences and BOS tokens
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_bos_tokens = [item['decoder_bos_token_id'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs for generation (just the BOS tokens)
    initial_decoder_inputs = torch.tensor(decoder_bos_tokens).unsqueeze(1)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, use_schema=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, use_schema=use_schema)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, use_schema=False):
    train_loader = get_dataloader(batch_size, "train", use_schema=use_schema)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema=use_schema)
    test_loader = get_dataloader(test_batch_size, "test", use_schema=use_schema)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load training data (for few-shot examples)
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    
    # Load development data
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Load test data (no labels)
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x
