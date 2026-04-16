import argparse
import json
import numpy as np

def sigmoid(x, temp):
    return 1 / (1 + np.exp(-x / temp))

def fusion_ablation_unifusion():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--temp", type=float)
    args = parser.parse_args()

    with open(f"./results/{args.dataset_name}_annotated.json", 'r') as f:
        dpo_data = json.load(f)
    with open(f"./results/{args.dataset_name}_detect.json", 'r') as f:
        det_data = json.load(f)

    det_dict = {item['id']: item['probability'] for item in det_data}
    
    results = []
    for item in dpo_data:
        qid = item['id']
        s_weak_A = sigmoid(float(item['details']['dpo_score_A']), args.temp)
        s_weak_B = sigmoid(float(item['details']['dpo_score_B']), args.temp)
        
        s_det_A = np.mean(list(det_dict[qid]['response_A_prob'].values()))
        s_det_B = np.mean(list(det_dict[qid]['response_B_prob'].values()))

        s_final_A = (1 - args.alpha) * s_weak_A + args.alpha * s_det_A
        s_final_B = (1 - args.alpha) * s_weak_B + args.alpha * s_det_B

        pref = "response_A" if s_final_A > s_final_B else "response_B"
        results.append({"id": qid, "pref-resp": pref})

    with open(f"./results/{args.dataset_name}_unifusion.json", 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    fusion_ablation_unifusion()