import os
import argparse
import random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available(
) else torch.device('cpu')  # you can add mps

FEW_SHOT_EXAMPLES = []  # will be populated in main()


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, schema_txt=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
        * schema_txt (str, optional): The database schema text
    '''
    parts = []
    if schema_txt:
        parts.append(
            "You are a helpful assistant that writes correct SQL for the given flight database.\n")
        parts.append("Database schema:\n")
        parts.append(schema_txt.strip()[:4000])  # cap context
        parts.append("\n---\n")

    if k > 0 and FEW_SHOT_EXAMPLES:
        k = min(k, len(FEW_SHOT_EXAMPLES))
        parts.append(f"Examples ({k}):\n")
        for demo_nl, demo_sql in FEW_SHOT_EXAMPLES[:k]:
            parts.append(f"NL: {demo_nl}\nSQL:\n{demo_sql}\n")
        parts.append("\n---\n")

    parts.append(f"NL: {sentence}\nSQL:")
    return "\n".join(parts)


def exp_kshot(tokenizer, model, inputs, k):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
    '''
    raw_outputs = []
    extracted_queries = []

    for i, sentence in tqdm(enumerate(inputs)):
        # Looking at the prompt may also help
        prompt = create_prompt(sentence, k)

        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # You should set MAX_NEW_TOKENS
        outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
        # How does the response look like? You may need to parse it
        response = tokenizer.decode(outputs[0])
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_path, model_path, gt_query_records, model_query_records):
    """
    Write predicted SQL to file, compute records, and return metrics and error rate.
    """
    # save queries and records (this will also compute records)
    # Make sure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_query_records), exist_ok=True)

    # model_path already written by caller if desired; we directly save here:
    # (safe to overwrite)
    # Note: we assume caller passes a list of strings in eval_y for ground-truth (dev only)
    # but compute_metrics reads GT from files.
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_path, model_path, gt_query_records, model_query_records
    )
    error_rate = sum(1 for e in (model_error_msgs or []) if e) / \
        max(1, len(model_error_msgs or []))
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)

    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         torch_dtype=torch.bfloat16,
                                                         config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         torch_dtype=torch.bfloat16).to(DEVICE)
    else:
        raise ValueError("Unsupported model name")
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    global FEW_SHOT_EXAMPLES

    args = get_args()
    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)
    FEW_SHOT_EXAMPLES = list(zip(train_x, train_y))

    schema_txt = read_schema(os.path.join(
        data_folder, "flight_database.schema"))

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(
        args.model, args.quantization)

    for eval_split in ["dev", "test"]:
        eval_x = dev_x if eval_split == "dev" else test_x

        # Generate SQL
        raw_outputs, extracted_queries = exp_kshot(
            tokenizer, model, eval_x, args.shot, schema_txt)

        # Save predictions and records
        os.makedirs("results", exist_ok=True)
        os.makedirs("records", exist_ok=True)
        tag = f"{args.model}_{args.experiment_name}_{eval_split}"
        model_sql_path = os.path.join(
            "results", f"gemma_{args.experiment_name}_{eval_split}.sql")
        model_record_path = os.path.join(
            "records", f"gemma_{args.experiment_name}_{eval_split}.pkl")

        # write queries file
        with open(model_sql_path, "w", encoding="utf-8") as f:
            for q in extracted_queries:
                f.write(q.strip() + "\n")

        # compute and save records for these queries
        from utils import compute_records
        recs, errs = compute_records(extracted_queries)
        import pickle
        with open(model_record_path, "wb") as pf:
            pickle.dump((recs, errs), pf)

        gt_sql_path = os.path.join(
            'data', f'{eval_split}.sql') if eval_split == "dev" else os.path.join('data', 'test.sql')
        gt_record_path = os.path.join('records', f'{eval_split}_gt_records.pkl') if eval_split == "dev" else os.path.join(
            'records', 'test_gt_records.pkl')
        # For test, leaderboard uses held-out; keep formats identical. If you don't have GT records for test locally,
        # you can still print metrics for dev and skip for test.
        if eval_split == "dev" and os.path.exists(gt_sql_path) and os.path.exists(gt_record_path):
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x, dev_y,
                gt_path=gt_sql_path,
                model_path=model_sql_path,
                gt_query_records=gt_record_path,
                model_query_records=model_record_path
            )
            print(
                f"{eval_split} set results: Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(
                f"{eval_split} set: {error_rate*100:.2f}% of generations caused SQL errors")
            # Save logs
            os.makedirs("logs", exist_ok=True)
            save_logs(os.path.join(
                "logs", f"{tag}.log"), sql_em, record_em, record_f1, model_error_msgs)
        else:
            print(
                f"{eval_split} predictions saved to {model_sql_path} / {model_record_path}")


if __name__ == "__main__":
    main()
