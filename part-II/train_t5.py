from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig, T5TokenizerFast
import os
import re
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data, DATA_ROOT
from utils import compute_metrics, save_queries_and_records, ensure_dev_ground_truth_records

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RECORDS_DIR = os.path.join(
    SCRIPT_DIR, "records_clean" if DATA_ROOT.endswith("data_clean") else "records")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


class StopOnSemicolon(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.semi_id = tokenizer.convert_tokens_to_ids(";")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return (input_ids[:, -1] == self.semi_id).any().item()


def get_args():
    p = argparse.ArgumentParser(description="T5 training loop")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--optimizer_type", type=str,
                   default="AdamW", choices=["AdamW"])
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_type", type=str, default="linear",
                   choices=["none", "cosine", "linear"])
    p.add_argument("--num_warmup_epochs", type=int, default=0)
    p.add_argument("--max_n_epochs", type=int, default=10)
    p.add_argument("--patience_epochs", type=int, default=3)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--experiment_name", type=str, default="dev")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--gen_max_new_tokens", type=int, default=128)
    p.add_argument("--gen_beam_size", type=int, default=8)
    p.add_argument("--fast_eval_every", type=int, default=2)
    return p.parse_args()


def _extract_sql_like(s: str) -> str:
    s = s.strip()
    m = re.search(r"(select\s.+?;)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    semi = s.find(";")
    if semi != -1:
        return s[: semi + 1].strip()
    return s


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1.0
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    experiment_name = args.experiment_name
    checkpoint_dir = os.path.join(
        CHECKPOINTS_DIR, f"{model_type}_experiments", experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    gt_sql_path = os.path.join(DATA_ROOT, "dev.sql")
    gt_record_path = ensure_dev_ground_truth_records(DATA_ROOT, RECORDS_DIR)

    model_sql_path = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    model_record_path = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        fast = (epoch % max(1, args.fast_eval_every) != 0)
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path, fast=fast
        )
        tag = "Eval/CE (fast)" if fast else "Eval/CE"
        print(f"{tag}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        if not fast:
            print(
                f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            wandb.log(
                {
                    "train/loss": tr_loss, "dev/loss": eval_loss,
                    "dev/record_f1": record_f1, "dev/record_em": record_em, "dev/sql_em": sql_em,
                    "dev/error_rate": error_rate if not fast else -1.0,
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
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

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

        loss = criterion(logits.view(-1, logits.size(-1)),
                         decoder_targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            ntoks = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * max(1, ntoks)
            total_tokens += max(1, ntoks)

    return total_loss / max(1, total_tokens)


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path, fast=False):
    model.eval()
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

    ce_loss /= max(1, total_tokens)
    if fast:
        return ce_loss, 0.0, 0.0, 0.0, 0.0

    # --------- Generation (seed decoder with "SELECT ") ----------
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])

    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        # NOTE: no decoder_start_token_id, we'll pass decoder_input_ids instead
    )

    select_prefix_ids = tok.encode("SELECT ", add_special_tokens=False)
    select_prefix = torch.tensor(
        select_prefix_ids, dtype=torch.long, device=DEVICE)

    sql_preds = []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Generate"):
            if isinstance(batch, (list, tuple)):
                if len(batch) == 5:
                    encoder_input, encoder_mask, _, _, _ = batch
                elif len(batch) == 3:
                    encoder_input, encoder_mask, _ = batch
                else:
                    raise ValueError(f"Unexpected batch len: {len(batch)}")
            else:
                raise ValueError("Batch must be tuple/list")

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)
            # seed the decoder with "SELECT "
            start_ids = select_prefix.unsqueeze(0).repeat(B, 1)  # (B, Lprefix)

            out = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=start_ids,
                generation_config=gen_cfg,
                stopping_criteria=stoppers,
            )
            for seq in out:
                s = tok.decode(seq, skip_special_tokens=True)
                sql_preds.append(_extract_sql_like(s))

    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    save_queries_and_records(sql_preds, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    if model_error_msgs:
        bad = [(i, e) for i, e in enumerate(model_error_msgs) if e]
        print(
            f"[DEBUG] {len(bad)} / {len(model_error_msgs)} dev preds raised DB errors")
        for i, e in bad[:10]:
            print(f"[DEBUG] idx={i} err={e}")

    error_rate = sum(1 for e in (model_error_msgs or []) if e) / \
        max(1, len(model_error_msgs or []))
    return ce_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    stoppers = StoppingCriteriaList([StopOnSemicolon(tok)])

    gen_cfg = GenerationConfig(
        max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.gen_beam_size,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    select_prefix_ids = tok.encode("SELECT ", add_special_tokens=False)
    select_prefix = torch.tensor(
        select_prefix_ids, dtype=torch.long, device=DEVICE)

    sql_preds = []
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Generate/Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            B = encoder_input.size(0)
            start_ids = select_prefix.unsqueeze(0).repeat(B, 1)

            out = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=start_ids,
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
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader))

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    model = load_model_from_checkpoint(args, best=True).eval()

    model_type = "ft" if args.finetune else "scr"
    experiment_name = args.experiment_name

    gt_sql_path = os.path.join(DATA_ROOT, "dev.sql")
    gt_record_path = ensure_dev_ground_truth_records(DATA_ROOT, RECORDS_DIR)

    dev_sql_out = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}.sql")
    dev_rec_out = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}.pkl")
    _ = eval_epoch(args, model, dev_loader, gt_sql_path,
                   dev_sql_out, gt_record_path, dev_rec_out, fast=False)

    test_sql_out = os.path.join(
        RESULTS_DIR, f"t5_{model_type}_{experiment_name}_test.sql")
    test_rec_out = os.path.join(
        RECORDS_DIR, f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_out, test_rec_out)


if __name__ == "__main__":
    main()
