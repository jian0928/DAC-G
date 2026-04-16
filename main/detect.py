import argparse
import json
import math
import os
import torch
from knockknock import wechat_sender
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_extract_key_dataset, get_raw_dataset

safe_token = "No"
unsafe_token = "Yes"
nlogprobs = 20

def parse_output(output):
    prob_of_risk = None, None
    if nlogprobs > 0:
        list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = get_probabilities(list_index_logprobs_i)
            prob_of_risk = prob[1]
    return prob_of_risk.item()

def get_probabilities(logprobs):
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += math.exp(logprob)
    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )
    return probabilities

def save_detect_results(prob_results, script_args):
    raw_data = get_raw_dataset(script_args.dataset_name, "ann")
    new_datasets = []
    for line in tqdm(raw_data):
        format_data = {
            "id": line['id'],
            "prompt": line['prompt'],
            "response_A": line['response_A'],
            "response_B": line['response_B'],
            "probability": {
                "response_A_prob": prob_results[f"prob_{line['id']}"]["A_rules_prob"],
                "response_B_prob": prob_results[f"prob_{line['id']}"]["B_rules_prob"],
            }
        }
        new_datasets.append(format_data)
    output_filepath = f"./results/{script_args.dataset_name}/{script_args.dataset_name}_detect.json"
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(new_datasets, f, ensure_ascii=False, indent=4)
    print("Saved detect annotated_results to {}".format(output_filepath))

def detect():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    script_args = parser.parse_args()

    model_path = "../model/granite-guardian-3.2-3b-a800m"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16
    )
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data = get_extract_key_dataset(script_args.dataset_name)
    with open("./config/saft_rules.json", 'r', encoding='utf-8') as f:
        rule_dic = json.load(f)

    prob_results = {}
    for item in tqdm(data):
        prompt = item["prompt"]
        response_A = item["response_A"]
        response_B = item["response_B"]
        messagesA = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_A}]
        messagesB = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_B}]

        A_rules_prob = {}
        B_rules_prob = {}
        for rule, description in rule_dic.items():
            guardian_config = {"risk_name": "personal_information", "risk_definition": f"{description}"}

            def _detect(messages):
                input_ids = tokenizer.apply_chat_template(
                    messages, guardian_config=guardian_config, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                model.eval()
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        do_sample=False,
                        max_new_tokens=20,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                return parse_output(output)

            A_rules_prob.update({rule: _detect(messagesA)})
            B_rules_prob.update({rule: _detect(messagesB)})
            
        prob_results.update({
            "id": item['id'],
            f"prob_{item['id']}": {
                "A_rules_prob": A_rules_prob,
                "B_rules_prob": B_rules_prob,
            }
        })
    save_detect_results(prob_results, script_args)

if __name__ == "__main__":
    detect()