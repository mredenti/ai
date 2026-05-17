"""
Download a HuggingFace dataset and save it to disk.

Usage:
    python download_dataset.py
    python download_dataset.py --dataset_id foo/bar
    python download_dataset.py --output_dir /path/to/save
"""

import argparse
from collections import Counter
from datasets import load_dataset

## Config -----------------------------
DATASET_ID   = "databricks/databricks-dolly-15k"
OUTPUT_DIR   = "./data/databricks-dolly-15k"
NUM_EXAMPLES = 3


## Main -----------------------------
def main(dataset_id: str, output_dir: str, num_examples: int):
    print(f"Downloading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id)

    ## Overview -----------------------------
    print(f"\n{dataset}\n")

    for split_name, split in dataset.items():
        print(f"## Split: '{split_name}' -----------------------------")
        print(f"  Rows    : {len(split):,}")
        print(f"  Columns : {split.column_names}")
        print(f"  Features: {split.features}")

        # Rough word count as a proxy for token budget
        if "instruction" in split.column_names and "response" in split.column_names:
            total_words = sum(
                len(row["instruction"].split()) + len(row["response"].split())
                for row in split
            )
            print(f"  ~Words  : {total_words:,}  (~{total_words // 750:,} pages)")

        # Category distribution (dolly-specific; skipped if column absent)
        if "category" in split.column_names:
            cats = Counter(split["category"])
            print(f"  Categories ({len(cats)}):")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"    {cat:<30} {count:>5}")

        ## Sample examples -----------------------------
        print(f"\n## First {num_examples} examples from '{split_name}' -----------------------------")
        for i, row in enumerate(split.select(range(num_examples))):
            print(f"\n  [Example {i}]")
            for col, val in row.items():
                preview = str(val).replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                print(f"    {col}: {preview}")

    ## Save -----------------------------
    dataset.save_to_disk(output_dir)
    print(f"\nDataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and inspect a HuggingFace dataset")
    parser.add_argument("--dataset_id",   default=DATASET_ID,   help="HuggingFace dataset ID")
    parser.add_argument("--output_dir",   default=OUTPUT_DIR,   help="Where to save the dataset")
    parser.add_argument("--num_examples", default=NUM_EXAMPLES, type=int, help="Examples to print")
    args = parser.parse_args()

    main(args.dataset_id, args.output_dir, args.num_examples)
