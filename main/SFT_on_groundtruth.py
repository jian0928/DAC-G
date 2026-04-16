import argparse
import numpy as np
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_seed, get_quantization_config, DPO_training, get_raw_dataset, format_DPO_data
from peft import prepare_model_for_kbit_training
import sys
sys.path.append('..')

np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak_model_name", type=str)
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
    script_args.model_name_or_path = f"../model/{script_args.weak_model_name}"
    script_args.wandb_run_name = f"DPO_create_weak_model-{script_args.dataset_name}-{script_args.weak_model_name}-LR{script_args.learning_rate}-Beta{script_args.beta}"

    return script_args

if __name__ == '__main__':
    script_args = get_args()
    set_seed(script_args.seed)

    wandb.init(
        project="DPO_create_weak_model",
        name=script_args.wandb_run_name,
        config=vars(script_args)
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config, model_dtype = get_quantization_config(script_args)

    weak_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        dtype=model_dtype,
        device_map="auto",
        quantization_config=bnb_config,
    )
    weak_model = prepare_model_for_kbit_training(weak_model)
    weak_model.gradient_checkpointing_enable()
    weak_model.enable_input_require_grads()

    reformatted_data = get_raw_dataset(script_args.dataset_name, "ann")
    reformatted_data = format_DPO_data(reformatted_data)
    raw_dataset = Dataset.from_list(reformatted_data)

    DPO_training(script_args, bnb_config, raw_dataset, weak_model, tokenizer, "create_weak_supervior")