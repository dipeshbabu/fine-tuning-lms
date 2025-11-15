import os

import torch

import transformers
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5Config, AutoConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    # Minimal safe setup
    if getattr(args, "use_wandb", False):
        wandb.init(project="t5-text2sql",
                   name=args.experiment_name or "experiment", config=vars(args))


def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small")
    else:
        cfg = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(cfg)
    model.to(DEVICE)
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    sub = "best" if best else "last"
    out_dir = os.path.join(checkpoint_dir, sub)
    mkdir(out_dir)
    model.save_pretrained(out_dir)
    

def load_model_from_checkpoint(args, best: bool = True, fallback_model=None):
    """
    Load a T5 model from a local checkpoint directory that was saved by `save_model`.
    If the checkpoint directory does not exist or is invalid, return `fallback_model`
    if provided; otherwise re-raise the exception.

    This function only loads from LOCAL DISK paths. It does NOT try to fetch from HF Hub.
    """
    # Construct the local path exactly like `save_model` wrote it
    model_type = "ft" if getattr(args, "finetune", False) else "scr"
    experiment_name = getattr(args, "experiment_name", "dev")
    root = getattr(args, "checkpoint_dir", None)

    if root is None:
        # Default to the same layout used in train_t5.py
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent
        root = script_dir / "checkpoints" / f"{model_type}_experiments" / experiment_name

    load_dir = os.path.join(str(root), "best" if best else "last")

    try:
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {load_dir}")

        # Force local loading: pass a config if present, otherwise let transformers infer it.
        if os.path.isfile(os.path.join(load_dir, "config.json")):
            config = AutoConfig.from_pretrained(load_dir)
            model = T5ForConditionalGeneration.from_pretrained(load_dir, config=config)
        else:
            model = T5ForConditionalGeneration.from_pretrained(load_dir)

        return model
    except Exception as e:
        if fallback_model is not None:
            print(f"[WARN] Could not load checkpoint at '{load_dir}': {e}\n"
                  f"[WARN] Falling back to the in-memory model.")
            return fallback_model
        # No fallback provided: bubble up
        raise


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(
        model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_tokenizer():
    """
    Factory to keep tokenizer creation in one place.
    """
    return T5TokenizerFast.from_pretrained("google-t5/t5-small")
