import argparse
import json
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import DPOTrainer, DPOConfig
from utils import set_seed, get_quantization_config, get_annotation_dataset, DPO_training
from peft import LoraConfig
import sys
sys.path.append('..')

np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--strong_model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quantization_bit", type=int)

    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--max_prompt_length", type=int)

    parser.add_argument("--r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)

    script_args = parser.parse_args()
    script_args.model_name_or_path = f"../model/{script_args.strong_model_name}"

    return script_args

if __name__ == '__main__':
    script_args = get_args()
    set_seed(script_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, trust_remote_code=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config, model_dtype = get_quantization_config(script_args)

    strong_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )

    reformatted_data = get_annotation_dataset(script_args.dataset_name)
    raw_dataset = Dataset.from_list(reformatted_data)

    DPO_training(script_args, bnb_config, raw_dataset, strong_model, tokenizer, "STF_strong_student")