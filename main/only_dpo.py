import json
from tqdm import tqdm
from utils import sigmoid_scores, get_raw_dataset
from evaluate import evaluate_ann_accuracy

def preprocess_data(json_data):
    preprocessed_data = []
    for entry in json_data:
        details = entry.get('details', {})
        preprocessed_data.append({
            'id': entry.get('id'),
            'dpo_score_A': float(details['dpo_score_A']),
            'dpo_score_B': float(details['dpo_score_B']),
            'harm_A': float(details['harm_A']),
            'harm_B': float(details['harm_B']),
        })
    return preprocessed_data

if __name__ == '__main__':
    train_dataset_name = "SafeRLHF"
    with open(f'./results/{train_dataset_name}/{train_dataset_name}_annotated.json', 'r', encoding='utf-8') as f:
        data_set = json.load(f)
    
    raw_data = get_raw_dataset(train_dataset_name, "ann")
    preprocessed_data = preprocess_data(data_set)

    annotated_results = []
    for item in tqdm(preprocessed_data):
        if item["dpo_score_A"] > item["dpo_score_B"]:
            annotated_results.append({
                'id': item['id'],
                'pref-resp' : "response_A"
            })
        else:
            annotated_results.append({
                'id': item['id'],
                'pref-resp': "response_B"
            })

    accuracy, right_count = evaluate_ann_accuracy(raw_data, annotated_results)
    
    print(train_dataset_name, "\nonly dpo:accuracy: ", round(accuracy, 3), "right_count: ", right_count, "total:", len(raw_data))