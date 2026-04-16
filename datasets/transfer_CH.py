import json
import os
import argparse
import random
import pandas as pd


def sanitize_content(content):
    if content is None:
        return ""

    if isinstance(content, bytes):
        return content.decode('utf-8')

    if hasattr(content, 'tolist'):
        return content.tolist()

    if hasattr(content, 'item'):
        return content.item()

    return content


def extract_response_text(data):

    data = sanitize_content(data)

    if isinstance(data, str):
        return data

    if isinstance(data, list):

        if not data:
            return ""

        last_message = data[-1]

        if isinstance(last_message, dict):
            return last_message.get('content', last_message.get('text', ""))

    return str(data)


def process_parquet_data(input_path, limit=None, random_mode=False):
    transformed_list = []

    try:
        df = pd.read_parquet(input_path)
        total_rows = len(df)

        use_limit = limit is not None and limit > 0

        for index, row in enumerate(df.itertuples(index=False)):

            raw_prompt = getattr(row, 'prompt', "")
            raw_chosen = getattr(row, 'chosen', "")
            raw_rejected = getattr(row, 'rejected', "")

            prompt_content = extract_response_text(raw_prompt) if isinstance(raw_prompt, list) else sanitize_content(
                raw_prompt)

            chosen_content = extract_response_text(raw_chosen)
            rejected_content = extract_response_text(raw_rejected)

            if chosen_content == rejected_content:
                continue

            new_obj = {
                "id": len(transformed_list),
                "pref-resp": "response_A",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                "response_A": {
                    "role": "assistant",
                    "content": chosen_content
                },
                "response_B": {
                    "role": "assistant",
                    "content": rejected_content
                }
            }

            transformed_list.append(new_obj)

    except Exception as e:
        print(f"Error processing parquet data: {e}")
        raise e

    try:
        final_json_output = json.dumps(transformed_list, indent=2, ensure_ascii=False)
    except TypeError as e:
        print("JSON Serialization failed. Checking data types in first entry:")
        if transformed_list:
            print(transformed_list[0])
        raise e

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

    base, _ = os.path.splitext(input_path)
    output_path = f"{args.input_file.split('.')[0]}_trans.json"

    print(f"Processing {input_path}...")

    try:
        transformed_json = process_parquet_data(input_path, args.limit, args.random)

        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(transformed_json)

        print(f"Successfully transformed and saved to {output_path}")

    except ImportError:
        print("Error: Missing dependencies. Please install pandas and pyarrow:")
        print("pip install pandas pyarrow")
    except Exception as e:
        print(f"Could not process file {input_path}")
        import traceback
        traceback.print_exc()

    print("Processing complete.")


if __name__ == "__main__":
    main()