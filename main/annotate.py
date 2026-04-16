import argparse
import json
import torch
import torch.nn.functional as F
from knockknock import wechat_sender
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from evaluate import evaluate_ann_accuracy
from utils import get_quantization_config, get_detect_dataset, get_raw_dataset, sigmoid_scores

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak_model_name", type=str)
    parser.add_argument("--train_dataset_name", type=str)
    parser.add_argument("--quantization_bit", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--max_prompt_length", type=int)

    args = parser.parse_args()

    args.output_data_path = f"./results/{args.train_dataset_name}/{args.train_dataset_name}_annotated.json"
    args.adapter_path = f"./results/create_weak_supervior/{args.train_dataset_name}_{args.weak_model_name}_final_checkpoint"
    args.base_model_path = f'../model/{args.weak_model_name}'
    return args

def load_models_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    bnb_config, model_dtype = get_quantization_config(args)

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=model_dtype,
    )
    ref_model.eval()

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=model_dtype,
    )
    policy_model = PeftModel.from_pretrained(policy_model, args.adapter_path)
    policy_model = policy_model.merge_and_unload()
    policy_model.eval()

    return policy_model, ref_model, tokenizer

def format_prompt(prompt_list, tokenizer):
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = str(prompt_list)

    return formatted_prompt

def get_response_log_prob(model, tokenizer, formatted_prompt, response_text, args):
    full_text = formatted_prompt + response_text + tokenizer.eos_token

    prompt_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=args.max_prompt_length,
        truncation=True
    ).to(model.device)

    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=args.max_length,
        truncation=True,
        padding=False
    ).to(model.device)

    input_ids = full_inputs.input_ids
    attention_mask = full_inputs.attention_mask

    prompt_length = prompt_inputs.input_ids.shape[1]
    response_token_ids = input_ids[:, prompt_length:]
    response_length = response_token_ids.shape[1]

    if response_length <= 0:
        return -float('inf')

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    response_logits = logits[:, (prompt_length - 1): -1, :]
    response_labels = input_ids[:, prompt_length:]

    if response_logits.shape[1] != response_labels.shape[1]:
        min_len = min(response_logits.shape[1], response_labels.shape[1])
        response_logits = response_logits[:, :min_len, :]
        response_labels = response_labels[:, :min_len]
        if min_len == 0:
            return -float('inf')

    log_probs = F.log_softmax(response_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=response_labels.unsqueeze(-1)
    ).squeeze(-1)

    total_log_prob = token_log_probs.sum().item()

    return total_log_prob

def detect_data_to_dic(detect_data):
    dic = {}
    for item in detect_data:
        dic[f"{item['id']}"] = {
            'response_A_prob': item['probability']['response_A_prob'],
            'response_B_prob': item['probability']['response_B_prob'],
        }
    return dic

def max_prob_in_dic(dic, id, resp):
    tag = "response_A_prob" if resp == 'A' else "response_B_prob"
    prob_list = []
    for (k, v) in dic[f'{id}'][tag].items():
        prob_list.append(v)
    return float(format(max(prob_list), '.3f'))

def calculate_fusion_score(s_weak, s_det, alpha, beta, threshold):
    base_score = (1 - alpha) * (1 - s_weak) + alpha * s_det
    indicator = float(s_det > threshold)
    diff = max(0, s_det - s_weak)
    correction = beta * indicator * diff
    y_final = base_score + correction
    return max(0.0, min(1.0, y_final))

def annotate():
    args = get_args()

    policy_model, ref_model, tokenizer = load_models_and_tokenizer(args)

    raw_data = get_raw_dataset(args.train_dataset_name, "ann")
    detect_data = get_detect_dataset(args.train_dataset_name)

    detect_data_dic = detect_data_to_dic(detect_data)

    annotated_results = []
    for item in tqdm(raw_data, desc="Annotating"):
        prompt_list = item["prompt"]
        resp_A_dict = item["response_A"]
        resp_B_dict = item["response_B"]

        resp_A_text = resp_A_dict["content"]
        resp_B_text = resp_B_dict["content"]

        formatted_prompt = format_prompt(prompt_list, tokenizer)

        policy_log_prob_A = get_response_log_prob(policy_model, tokenizer, formatted_prompt, resp_A_text, args)
        policy_log_prob_B = get_response_log_prob(policy_model, tokenizer, formatted_prompt, resp_B_text, args)

        ref_log_prob_A = get_response_log_prob(ref_model, tokenizer, formatted_prompt, resp_A_text, args)
        ref_log_prob_B = get_response_log_prob(ref_model, tokenizer, formatted_prompt, resp_B_text, args)

        dpo_score_A = sigmoid_scores(policy_log_prob_A - ref_log_prob_A, )
        dpo_score_B = sigmoid_scores(policy_log_prob_B - ref_log_prob_B, )

        harm_A = max_prob_in_dic(detect_data_dic, item["id"], 'A')
        harm_B = max_prob_in_dic(detect_data_dic, item["id"], 'B')

        final_score_A = calculate_fusion_score(dpo_score_A, harm_A, alpha, beta, threshold)
        final_score_B = calculate_fusion_score(dpo_score_B, harm_B, alpha=, beta, threshold)

        beta_score_diff = args.beta * (final_score_A - final_score_B)
        prob_A_is_chosen = torch.sigmoid(torch.tensor(beta_score_diff, dtype=torch.float32)).item()
        prob_B_is_chosen = 1.0 - prob_A_is_chosen

        if final_score_A < final_score_B:
            pref_resp = 'response_A'
            chosen = resp_A_dict
            rejected = resp_B_dict
            chosen_dpo_score = final_score_A
            rejected_dpo_score = final_score_B
            preference_confidence = prob_A_is_chosen
        else:
            pref_resp = 'response_B'
            chosen = resp_B_dict
            rejected = resp_A_dict
            chosen_dpo_score = final_score_B
            rejected_dpo_score = final_score_A
            preference_confidence = prob_B_is_chosen

        result_item = {
            "id": item['id'],
            "prompt": prompt_list,
            "chosen": chosen,
            "rejected": rejected,
            "pref-resp": pref_resp,
            "model_preference_confidence": preference_confidence,
            "dpo_score_chosen": chosen_dpo_score,
            "dpo_score_rejected": rejected_dpo_score,
            "details": {
                "response_A": resp_A_dict,
                "response_B": resp_B_dict,
                "prob_A_is_chosen": prob_A_is_chosen,
                "prob_B_is_chosen": prob_B_is_chosen,
                "dpo_score_A": dpo_score_A,
                "dpo_score_B": dpo_score_B,
                "harm_A": harm_A,
                "harm_B": harm_B,
                "final_score_A": final_score_A,
                "final_score_B": final_score_B,
                "policy_log_prob_A": policy_log_prob_A,
                "policy_log_prob_B": policy_log_prob_B,
                "ref_log_prob_A": ref_log_prob_A,
                "ref_log_prob_B": ref_log_prob_B
            }
        }
        annotated_results.append(result_item)

    with open(args.output_data_path, 'w', encoding='utf-8') as f:
        json.dump(annotated_results, f, ensure_ascii=False, indent=4)

    del policy_model, ref_model, tokenizer

    accuracy, right_count = evaluate_ann_accuracy(raw_data, annotated_results)
    
    with open(f"./results/{args.train_dataset_name}/{args.train_dataset_name}_annotated.txt", 'a') as f:
        strr = f"my_method:accuracy:{round(accuracy, 3)}, right_count:{right_count}, total:{len(raw_data)}\n"
        f.write(strr)

if __name__ == '__main__':
    annotate()