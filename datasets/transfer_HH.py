import json
import os
import argparse
import random


def transform_message(message_obj):
    """
    Transforms a single message object.
    - Changes 'human' role to 'user'
    - Changes 'text' key to 'content'
    """
    if not isinstance(message_obj, dict):
        return message_obj

    new_message = {}

    if 'role' in message_obj:
        if message_obj['role'] == 'human':
            new_message['role'] = 'user'
        else:
            new_message['role'] = message_obj['role']

    if 'text' in message_obj:
        new_message['content'] = message_obj['text']

    for key, value in message_obj.items():
        if key not in ['role', 'text']:
            new_message[key] = value

    return new_message


def process_data(data_string, limit=None):
    """
    Processes the multi-line JSONL string and transforms it
    into a standard JSON list string.
    """
    transformed_list = []

    all_lines = [line for line in data_string.strip().split('\n') if line.strip()]
    total_lines = len(all_lines)

    use_limit = limit is not None and limit > 0
    selected_lines = []
c
    if use_limit and limit < total_lines:
        print(f"Randomly selecting {limit} entries from a total of {total_lines}.")
        selected_lines = random.sample(all_lines, limit)
    else:
        if use_limit:
            print(f"Limit ({limit}) >= total entries ({total_lines}). Processing all entries.")
        else:
            print(f"No limit set. Processing all {total_lines} entries.")
        selected_lines = all_lines

    print(f"Processing {len(selected_lines)} entries...")

    for line in selected_lines:
        llen = len(transformed_list)

        try:
            original_data = json.loads(line)

            transformed_data = {'id': llen, "pref-resp": "response_A"}

            if 'context' in original_data:
                transformed_data['prompt'] = [transform_message(msg) for msg in original_data['context']]

            if 'chosen' in original_data:
                transformed_data['response_A'] = transform_message(original_data['chosen'])

            if 'rejected' in original_data:
                transformed_data['response_B'] = transform_message(original_data['rejected'])

            transformed_list.append(transformed_data)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line}")
            print(f"Error: {e}\n")

    final_json_output = json.dumps(transformed_list, indent=2, ensure_ascii=False)

    return final_json_output


def main():
    """
    Main function to read a file, process it,
    and write to a new file in the same directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--limit", type=int,)

    args = parser.parse_args()

    input_path = args.input_file

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    output_path = f"{args.input_file.split('.')[0]}_trans.json"

    print(f"Processing {args.input_file}...")

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            file_content = f_in.read()

        transformed_json = process_data(file_content, args.limit)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(transformed_json)

    except Exception as e:
        print(f"Could not process file {input_path}: {e}")


if __name__ == "__main__":
    main()



