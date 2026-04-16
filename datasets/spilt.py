import json
import os
import random
import argparse


def split_and_save_json(
        input_path: str,
        output_dir: str,
        split_ratio: float,
        seed: int = 42
):

    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    print(f"total_size: {total_size}")

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    random.shuffle(data)
    split_index = int(total_size * split_ratio)

    part1 = data[:split_index]
    part2 = data[split_index:]

    size_part1 = len(part1)
    size_part2 = len(part2)


    name1 = f"{(input_path.split('/')[-1]).split('.')[0]}_train_{size_part1}.json"
    name2 = f"{(input_path.split('/')[-1]).split('.')[0]}_ann_{size_part2}.json"
    output_path_part1 = os.path.join(output_dir, name1)
    output_path_part2 = os.path.join(output_dir, name2)

    print(f" result (ratio: {split_ratio:.2f}): ")
    print(f"   - {name1} (Part 1): {size_part1}   ({size_part1 / total_size:.1%})")
    print(f"   - {name2} (Part 2): {size_part2}   ({size_part2 / total_size:.1%})")

    with open(output_path_part1, 'w', encoding='utf-8') as f:
        json.dump(part1, f, ensure_ascii=False, indent=4)

    with open(output_path_part2, 'w', encoding='utf-8') as f:
        json.dump(part2, f, ensure_ascii=False, indent=4)

    print(f"save2: {os.path.abspath(output_path_part1)} 和 {os.path.abspath(output_path_part2)}")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--input_path',
        type=str,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
    )

    parser.add_argument(
        '-r', '--ratio',
        type=float,
    )


    parser.add_argument(
        '--seed',
        type=int,
    )


    args = parser.parse_args()

    split_and_save_json(
        input_path=args.input_path,
        output_dir=args.output_dir,
        split_ratio=args.ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()