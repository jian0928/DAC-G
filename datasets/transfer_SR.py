import json
import os
import argparse
import random


def process_pku_data(data_string, limit=None, random_mode=False):
    transformed_list = []

    all_lines = [line for line in data_string.strip().split('\n') if line.strip()]
    total_lines = len(all_lines)

    processing_lines = all_lines

    use_limit = limit is not None and limit > 0

    if use_limit:
        if limit < total_lines:
            if random_mode:
                print(f"Randomly sampling {limit} entries from {total_lines} total entries.")
                processing_lines = random.sample(all_lines, limit)
            else:
                print(f"Processing the first {limit} entries from {total_lines} total entries.")
                processing_lines = all_lines[:limit]
        else:
            print(f"Limit ({limit}) is larger than total lines ({total_lines}). Processing all.")

    for line in processing_lines:
        try:
            original_data = json.loads(line)

            prompt_text = original_data.get('prompt', "")
            resp_0_text = original_data.get('response_0', "")
            resp_1_text = original_data.get('response_1', "")

            is_safe_0 = original_data.get('is_response_0_safe', False)
            is_safe_1 = original_data.get('is_response_1_safe', False)
            better_id = original_data.get('better_response_id')
            safer_id = original_data.get('safer_response_id')

            chosen_id = None

            if is_safe_0 and is_safe_1:
                chosen_id = better_id
            elif is_safe_0:
                chosen_id = 0
            elif is_safe_1:
                chosen_id = 1
            else:
                if safer_id is not None:
                    chosen_id = safer_id
                else:
                    chosen_id = better_id

            if chosen_id is None:
                continue

            pref_resp_str = "response_A" if chosen_id == 0 else "response_B"

            new_obj = {
                "id": len(transformed_list),
                "pref-resp": pref_resp_str,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                "response_A": {
                    "role": "assistant",
                    "content": resp_0_text
                },
                "response_B": {
                    "role": "assistant",
                    "content": resp_1_text
                }
            }

            transformed_list.append(new_obj)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line}")
            print(f"Error: {e}\n")

    final_json_output = json.dumps(transformed_list, indent=2, ensure_ascii=False)

    return final_json_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()

    input_path = args.input_file

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    output_path = f"{args.input_file.split('.')[0]}_trans.json"

    print(f"Processing {input_path}...")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            file_content = f_in.read()

        transformed_json = process_pku_data(file_content, args.limit, args.random)

        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(transformed_json)

        print(f"Successfully transformed and saved to {output_path}")

    except Exception as e:
        print(f"Could not process file {input_path}: {e}")
        import traceback
        traceback.print_exc()

    print("Processing complete.")


if __name__ == "__main__":
    main()