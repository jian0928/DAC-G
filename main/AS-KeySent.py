import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_raw_dataset

def parse_output(output, nlogprobs, tokenizer, safe_token, unsafe_token):
    list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                             for token_i in list(output.scores)[:-1]]
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in list_index_logprobs_i:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += torch.exp(torch.tensor(logprob))
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += torch.exp(torch.tensor(logprob))
    probabilities = torch.softmax(torch.log(torch.tensor([safe_token_prob, unsafe_token_prob])), dim=0)
    return probabilities[1].item()

def detect_ablation_keysent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    data = get_raw_dataset(args.dataset_name, "ann")
    with open("./config/saft_rules.json", 'r', encoding='utf-8') as f:
        rule_dic = json.load(f)

    prob_results = {}
    for item in tqdm(data):
        messagesA = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response_A"]}]
        messagesB = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response_B"]}]
        
        A_rules_prob = {}
        B_rules_prob = {}
        for rule, description in rule_dic.items():
            config = {"risk_name": "harm", "risk_definition": description}
            
            def get_prob(msgs):
                ids = tokenizer.apply_chat_template(msgs, guardian_config=config, add_generation_prompt=True, return_tensors="pt").to(model.device)
                out = model.generate(ids, do_sample=False, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
                return parse_output(out, 20, tokenizer, "No", "Yes")

            A_rules_prob[rule] = get_prob(messagesA)
            B_rules_prob[rule] = get_prob(messagesB)

        prob_results[item['id']] = {"A_prob": A_rules_prob, "B_prob": B_rules_prob}

    with open(f"./results/{args.dataset_name}_no_keysent.json", 'w') as f:
        json.dump(prob_results, f)

if __name__ == "__main__":
    detect_ablation_keysent()