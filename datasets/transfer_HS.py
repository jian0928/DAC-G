import json
import argparse
from collections import defaultdict
import os

def read_jsonl_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                yield json.loads(stripped_line)


def parse_multi_turn_prompt(prompt_str):
    parts = prompt_str.split("<extra_id_1>")

    dialogue = []
    current_role = "user"

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if part.startswith("Assistant"):
            content = part[len("Assistant"):].strip()
            if content:
                dialogue.append({"role": "assistant", "content": content})
                current_role = "assistant"
        elif part.startswith("User"):
            content = part[len("User"):].strip()
            if content:
                dialogue.append({"role": "user", "content": content})
                current_role = "user"
        else:
            if not dialogue:
                dialogue.append({"role": "user", "content": part})
            else:
                pass

    return dialogue


def create_preference_dataset(input_file_path, output_file_path):
    prompt_groups = defaultdict(list)
    total_samples = 0

    for item in read_jsonl_generator(input_file_path):
        total_samples += 1
        prompt_key = item.get('prompt', '')

        response_info = {
            'text': item.get('response', ''),
            'helpfulness': item.get('helpfulness', None),
            'prompt_id': item.get('prompt_id', None)
        }

        if prompt_key and response_info['helpfulness'] is not None:
            prompt_groups[prompt_key].append(response_info)

    preference_data = []
    pair_id_counter = 1

    for prompt_key, responses in prompt_groups.items():
        if len(responses) < 2:
            continue
        scores = [r['helpfulness'] for r in responses]

        if len(set(scores)) <= 1:
            continue

        max_score = max(scores)
        best_response = next(r for r in responses if r['helpfulness'] == max_score)

        min_score = min(scores)
       
        worst_response = next(r for r in responses if r['helpfulness'] == min_score and r is not best_response)

        
        if best_response is None or worst_response is None:
            continue

        parsed_prompt = parse_multi_turn_prompt(prompt_key)

        data_item = {
            "id": pair_id_counter,
            "pref-resp": "response_A",
            "prompt": parsed_prompt,
            "response_A": {
                "role": "assistant",
                "content": best_response['text']
            },
            "response_B": {
                "role": "assistant",
                "content": worst_response['text']
            },
            "gap": abs(best_response['helpfulness'] - worst_response['helpfulness'])
        }

        preference_data.append(data_item)
        pair_id_counter += 1

    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        type=str,
    )

    args = parser.parse_args()

    create_preference_dataset(args.input_file, args.output_file)


if __name__ == "__main__":
    main()