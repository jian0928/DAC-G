import argparse
import logging
import os
import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_quantization_config, get_raw_dataset

def evaluate_ann_accuracy(raw_data, ann_data):
    if len(raw_data) != len(ann_data):
        exit()

    right = 0
    for i, (raw, ann) in enumerate(tqdm(zip(raw_data, ann_data))):
        if raw['id'] != ann['id']:
            exit()
        if raw['pref-resp'] == ann['pref-resp']:
            right += 1
    return right / len(raw_data), right

def load_eval_model(base_model_path, lora_adapter_path, quantization_bit):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class Args:
        def __init__(self, q_bit): self.quantization_bit = q_bit

    bnb_config, _ = get_quantization_config(Args(quantization_bit))

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval()
    return model, tokenizer

def get_logprobs(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    per_token_logps = torch.gather(log_probs, dim=2, index=gather_labels.unsqueeze(2)).squeeze(2)
    mask = (shift_labels != -100)
    return (per_token_logps * mask).sum(-1)

@torch.no_grad()
def evaluate_accuracy(model, tokenizer, test_dataset):
    accuracy_list = []
    for item in tqdm(test_dataset):
        prompt_str = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else str(item["prompt"])
        if item.get("pref-resp") == "response_A":
            chosen_str = item["response_A"]["content"]
            rejected_str = item["response_B"]["content"]
        else:
            chosen_str = item["response_B"]["content"]
            rejected_str = item["response_A"]["content"]

        def encode(p, res):
            full_text = f"User: {p}\nAssistant: {res}"
            inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
            prompt_header = f"User: {p}\nAssistant: "
            prompt_ids = tokenizer(prompt_header, return_tensors="pt")["input_ids"]
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_ids.shape[1]] = -100
            return inputs, labels

        c_in, c_lab = encode(prompt_str, chosen_str)
        c_lp = get_logprobs(model(**c_in).logits, c_lab)
        r_in, r_lab = encode(prompt_str, rejected_str)
        r_lp = get_logprobs(model(**r_in).logits, r_lab)
        accuracy_list.append(1 if c_lp > r_lp else 0)
    return np.mean(accuracy_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--quantization_bit", type=int)
    script_args = parser.parse_args()

    script_args.BASE_MODEL = f"../model/{script_args.model_name}"
    script_args.LORA_PATH = f"./results/create_weak_supervior/{script_args.dataset_name}_{script_args.model_name}_final_checkpoint"

    model, tokenizer = load_eval_model(script_args.BASE_MODEL, script_args.LORA_PATH, script_args.quantization_bit)
    test_data = get_raw_dataset(script_args.dataset_name, "test")
    
    eval_size = min(500, len(test_data))
    test_data_sampled = test_data[:eval_size]

    acc = evaluate_accuracy(model, tokenizer, test_data_sampled)

    result_str = (
        f"Model: {script_args.model_name} | "
        f"Dataset: {script_args.dataset_name} | "
        f"Accuracy: {acc:.3%}\n"
    )
    
    output_dir = f"./results/{script_args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{script_args.dataset_name}_model_eval.txt")
    with open(log_path, 'a', encoding="utf-8") as f:
        f.write(result_str)