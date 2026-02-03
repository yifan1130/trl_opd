#!/usr/bin/env python
"""
Convert HuggingFace Countdown-Task-GOLD dataset to verl parquet format.

The TRL script uses the dataset with 'messages' column:
  [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

For verl with on-policy distillation (lmbda=1.0), we need:
  - prompt: [{"role": "system", ...}, {"role": "user", ...}]  (NO assistant - student generates)
  - reward_model: {"ground_truth": "target", "style": "rule"}

This keeps the prompt identical to what TRL GOLDTrainer would use.
"""

import pandas as pd
from datasets import load_dataset
import os


def convert_countdown_to_verl(
    dataset_name: str = "HuggingFaceTB/Countdown-Task-GOLD",
    subset: str = "verified_Qwen3-4B-Instruct-2507",
    output_path: str = "/data/cxu/tree_grpo/YYF42_data_train/train_countdown_task.parquet"
):
    """Convert Countdown dataset to verl format."""

    print(f"Loading dataset: {dataset_name} ({subset})")
    ds = load_dataset(dataset_name, subset, split="train")

    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Convert to verl format
    processed_data = []

    for i, example in enumerate(ds):
        # Get the messages from dataset
        messages = example.get("messages", [])

        # Extract prompt: keep system and user messages, remove assistant
        # This is exactly what TRL GOLDTrainer does for on-policy (lmbda=1.0)
        prompt = [msg for msg in messages if msg["role"] != "assistant"]

        # Get target number for reward verification
        target = example.get("target", None)

        # Create verl format entry
        verl_entry = {
            "data_source": "countdown_task",
            "prompt": prompt,
            "ability": "MATH",
            "reward_model": {
                "ground_truth": str(target) if target is not None else "",
                "style": "rule"
            },
            "extra_info": {
                "index": f"countdown_{i}",
                "nums": example.get("nums", []),
                "target": target,
            },
            "index": f"countdown_{i}"
        }

        processed_data.append(verl_entry)

    # Create DataFrame and save
    df = pd.DataFrame(processed_data)

    print(f"\nConverted {len(df)} examples")
    print(f"Saving to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(output_path, index=False)

    # Show sample
    print("\n=== Sample entry ===")
    sample = df.iloc[0]
    print(f"prompt: {sample['prompt']}")
    print(f"reward_model: {sample['reward_model']}")

    return output_path


if __name__ == "__main__":
    convert_countdown_to_verl()
