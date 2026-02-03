#!/usr/bin/env python
# Training script for DAPO-Math dataset using on-policy distillation
# Pure on-policy (lmbda=1.0): student generates completions, learns from teacher logits
#
# Teacher: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# Student: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

from dataclasses import dataclass, field

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from trl import TrlParser
from trl.experimental.gold import GOLDConfig, GOLDTrainer


@dataclass
class ScriptArguments:
    """Script-specific arguments."""

    # Model configuration
    student_model: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "Student model name or path"},
    )
    teacher_model: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        metadata={"help": "Teacher model name or path"},
    )

    # Dataset configuration
    dataset_path: str = field(
        default="/data/cxu/tree_grpo/YYF42_data_train/train_DAPO-Math-17k_sampled.parquet",
        metadata={"help": "Path to parquet dataset"},
    )


def load_and_prepare_dataset(dataset_path: str) -> Dataset:
    """
    Load DAPO-Math dataset and prepare it for GOLDTrainer.

    The dataset has:
      - prompt: [{"role": "user", "content": "..."}]  (numpy array)
      - reward_model: {"ground_truth": "answer", "style": "rule"}

    We need to convert it to:
      - messages: [{"role": "user", ...}, {"role": "assistant", "content": ""}]

    The empty assistant message is a placeholder - with lmbda=1.0,
    the student will generate its own completion which replaces this.
    """
    # Load parquet
    df = pd.read_parquet(dataset_path)

    # Prepare messages format
    processed_data = []
    for idx, row in df.iterrows():
        # Convert numpy array to list if needed
        prompt_messages = row["prompt"]
        if hasattr(prompt_messages, "tolist"):
            prompt_messages = prompt_messages.tolist()

        # Add a placeholder assistant message
        # This will be replaced by student-generated completion when lmbda=1.0
        messages = prompt_messages + [{"role": "assistant", "content": ""}]

        processed_data.append({
            "messages": messages,
            # Keep ground truth for potential reward/verification
            "ground_truth": row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else "",
        })

    return Dataset.from_list(processed_data)


def main():
    # Parse arguments
    parser = TrlParser((ScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_and_config()

    # Set teacher model init kwargs to load on single device
    if training_args.teacher_model_init_kwargs is None:
        training_args.teacher_model_init_kwargs = {}
    training_args.teacher_model_init_kwargs["device_map"] = None
    if "torch_dtype" not in training_args.teacher_model_init_kwargs:
        training_args.teacher_model_init_kwargs["torch_dtype"] = "bfloat16"

    # Load and prepare dataset
    print(f"Loading dataset from {script_args.dataset_path}")
    dataset = load_and_prepare_dataset(script_args.dataset_path)
    print(f"Dataset size: {len(dataset)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.student_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = GOLDTrainer(
        model=script_args.student_model,
        teacher_model=script_args.teacher_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("=" * 60)
    print("On-Policy Distillation Training")
    print("=" * 60)
    print(f"Student: {script_args.student_model}")
    print(f"Teacher: {script_args.teacher_model}")
    print(f"Dataset: {script_args.dataset_path}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Lambda (on-policy fraction): {training_args.lmbda}")
    print(f"Beta (KL interpolation): {training_args.beta}")
    print(f"Max completion length: {training_args.max_completion_length}")
    print("=" * 60)

    trainer.train()

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
