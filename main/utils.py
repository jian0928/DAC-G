import json
import glob
import random
from datetime import datetime
import numpy as np
import torch
from knockknock import wechat_sender
from peft import LoraConfig
from transformers import is_torch_available, is_torch_npu_available, is_torch_xpu_available, is_tf_available, \
    BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

def cal_train_time(total_time):
    total_seconds = int(total_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}"

def get_detect_dataset(dataset_name):
    with open(f"./results/{dataset_name}/{dataset_name}_detect.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_raw_dataset(dataset_name, tag):
    if tag == "train":
        pattern = f"./datasets/{dataset_name}/train_trans_train_*.json"
    elif tag == "ann":
        pattern = f"./datasets/{dataset_name}/train_trans_ann_*.json"
    elif tag == "test":
        pattern = f"./datasets/{dataset_name}/test_trans.json"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Missing: {pattern}")
    with open(files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_annotation_dataset(dataset_name):
    with open(f"./results/{dataset_name}/{dataset_name}_annotation.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_extract_key_dataset(dataset_name):
    with open(f"./results/{dataset_name}/{dataset_name}_extract_key.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_DPO_data(data_list):
    for data in data_list:
        if data['pref-resp'] == 'response_A':
            data['chosen'] = data['response_A']
            data['rejected'] = data['response_B']
        else:
            data['chosen'] = data['response_B']
            data['rejected'] = data['response_A']
        del data['pref-resp'], data['response_A'], data['response_B']
    return data_list

def get_quantization_config(args):
    if args.quantization_bit == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_dtype = torch.float16
    elif args.quantization_bit == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
        )
        model_dtype = torch.float16
    else:
        bnb_config = None
        model_dtype = torch.float16
    return bnb_config, model_dtype

def sigmoid_scores(scores, temperature):
    return 1 / (1 + np.exp(-scores / temperature))

def DPO_training(script_args, bnb_config, raw_dataset, model, tokenizer, tag):
    script_args.model_kwargs = {
        "quantization_config": bnb_config,
    }

    training_args = DPOConfig(
        output_dir="./results",
        num_train_epochs=script_args.num_train_epochs,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=script_args.warmup_ratio,
        save_strategy="epoch",
        logging_steps=script_args.logging_steps,
        beta=script_args.beta,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        report_to="wandb",
        run_name=f"DPO_{tag}-{script_args.dataset_name}-{script_args.weak_model_name}",
    )

    peft_config = LoraConfig(
        r=script_args.r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    def clean_dpo_example(example):
        prompt = str(example.get("prompt", ""))
        chosen = str(example.get("chosen", ""))
        rejected = str(example.get("rejected", ""))
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    train_dataset = raw_dataset.map(clean_dpo_example, load_from_cache_file=False)
    train_dataset = train_dataset.filter(
        lambda x: x["prompt"] and x["chosen"] and x["rejected"],
        load_from_cache_file=False
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    start_time = datetime.now()
    dpo_trainer.train()

    output_dir = f"{training_args.output_dir}/{tag}/{script_args.dataset_name}_{script_args.weak_model_name}_final_checkpoint"
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    end_time = datetime.now()
    print(cal_train_time(end_time - start_time))