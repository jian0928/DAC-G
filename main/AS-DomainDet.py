import argparse
import json
from utils import get_raw_dataset

def evaluate_ablation_domaindet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    args = parser.parse_args()

    with open(f'./results/{args.dataset_name}_annotated.json', 'r') as f:
        data_set = json.load(f)
    raw_data = get_raw_dataset(args.dataset_name, "ann")

    annotated_results = []
    for entry in data_set:
        details = entry.get('details', {})
        score_A = float(details['dpo_score_A'])
        score_B = float(details['dpo_score_B'])
        
        pref = "response_A" if score_A > score_B else "response_B"
        annotated_results.append({'id': entry.get('id'), 'pref-resp': pref})

    right = 0
    raw_dict = {item['id']: item['pref-resp'] for item in raw_data}
    for ann in annotated_results:
        if raw_dict.get(ann['id']) == ann['pref-resp']:
            right += 1
    
    acc = right / len(raw_data)
    print(f"Ablation DomainDet Acc: {acc}")

if __name__ == "__main__":
    evaluate_ablation_domaindet()