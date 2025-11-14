from transformers import StoppingCriteria, StoppingCriteriaList
import os
import re
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

# Resolve all paths relative to this file so CWD doesn’t matter
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "records")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


class StopOnSemicolon(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.semi_id = self.tokenizer.convert_tokens_to_ids(";")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Stop if the last generated token is ';'
        return input_ids[0, -1].item() == self.semi_id


def get_args():
    """
    Arguments for training. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument("--finetune", action="store_true",
                        help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                        choices=["AdamW"], help="What optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument("--num_warmup_epochs", type=int, default=0,
                        help="Warmup epochs if using a scheduler")
    parser.add_argument("--max_n_epochs", type=int, default=3,
                        help="How many epochs to train the model for")
    parser.add_argument("--patience_epochs", type=int, default=2,
                        help="Early stop if dev F1 does not improve")

    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for experiment tracking")
    parser.add_argument("--experiment_name", type=str, default="dev",
                        help="Experiment name (used in filenames)")

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    # Generation hyperparameters (used in eval/test)
    parser.add_argument("--gen_max_new_tokens", type=int, default=32)
    parser.add_argument("--gen_beam_size", type=int, default=4)

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    experiment_name = args.experiment_name  # use CLI value

    checkpoint_dir = os.path.join(
        CHECKPOINTS_DIR, f"{model_type}_experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Dev file paths
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(
        RECORDS_DIR, "ground_truth_dev.pkl")  # must exist

    # Where to write model predictions for dev
    model_sql_path = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(
            f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, "
            f"Record EM: {record_em}, SQL EM: {sql_em}"
        )
        print(
            f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": tr_loss,
                    "dev/loss": eval_loss,
                    "dev/record_f1": record_f1,
                    "dev/record_em": record_em,
                    "dev/sql_em": sql_em,
                    "dev/error_rate": error_rate,
                },
                step=epoch,
            )

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    # ignore_index masks PAD_IDX in the loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Train"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        # Flatten for CE
        loss = criterion(logits.view(-1, logits.size(-1)),
                         decoder_targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            non_pad = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * max(1, non_pad)
            total_tokens += max(1, non_pad)

    return total_loss / max(1, total_tokens)


def _extract_sql_like(s: str) -> str:
    """
    Safer extraction for generated sequences:
    - prefer the first 'SELECT ... ;' block (case-insensitive)
    - else, trim at first ';'
    - else, return stripped string
    """
    s = s.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    semi = s.find(";")
    if semi != -1:
        return s[: semi + 1].strip()
    return s


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    """
    Evaluation loop on dev: CE loss + generation + metrics.
    """
    model.eval()
    # 1) Cross-entropy on dev
    ce_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="Eval/CE"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            ).logits

            loss = criterion(logits.view(-1, logits.size(-1)),
                             decoder_targets.view(-1))
            ntoks = (decoder_targets != PAD_IDX).sum().item()
            ce_loss += loss.item() * max(1, ntoks)
            total_tokens += max(1, ntoks)

    ce_loss = ce_loss / max(1, total_tokens)

    # 2) Generation -> save -> metrics
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        do_sample=False
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,  # T5 decoder starts from pad
    )

    sql_preds = []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Generate/Dev"):
            # works for both 5-tuple (train/dev) or 3-tuple (if reused)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 5:
                    encoder_input, encoder_mask, _, _, _ = batch
                elif len(batch) == 3:
                    encoder_input, encoder_mask, _ = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
            else:
                raise ValueError("Batch must be a tuple/list.")

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            out = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_cfg,
                stopping_criteria=stoppers,
            )
            for seq in out:
                s = tok.decode(seq, skip_special_tokens=True)
                sql_preds.append(_extract_sql_like(s))

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)

    save_queries_and_records(sql_preds, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = sum(1 for e in (model_error_msgs or []) if e) / \
        max(1, len(model_error_msgs or []))
    return ce_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """
    Inference on the test set: generate SQL and compute records; no ground truth.
    """
    model.eval()
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])
    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        do_sample=False
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        decoder_start_token_id=tok.pad_token_id,  # T5 decoder starts from pad
    )

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_dec in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            out = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_cfg,
                stopping_criteria=stoppers,
            )
            for seq in out:
                s = tok.decode(seq, skip_special_tokens=True)
                sql_preds.append(_extract_sql_like(s))

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate (reload best)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    # Dev set eval (with your experiment name)
    experiment_name = args.experiment_name
    model_type = "ft" if args.finetune else "scr"
    gt_sql_path = os.path.join(DATA_DIR, "dev.sql")
    gt_record_path = os.path.join(
        RECORDS_DIR, "ground_truth_dev.pkl")  # must exist already
    model_sql_path = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(
        f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, "
        f"Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}"
    )
    print(
        f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    test_sql_path = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)


if __name__ == "__main__":
    main()
