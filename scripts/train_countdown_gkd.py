#!/usr/bin/env python
# Training script for Countdown task using GKD (Generalized Knowledge Distillation)
# GKD uses standard JSD/KL loss WITHOUT cross-tokenizer alignment
# Use this when student and teacher share the same tokenizer
#
# Teacher: Qwen/Qwen3-4B-Instruct-2507
# Student: Qwen/Qwen2.5-1.5B-Instruct
# Loss: Reverse KL (beta=1.0) with standard token-level distillation

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import TrlParser
from trl.experimental.gold import GOLDConfig, GOLDTrainer


@dataclass
class ScriptArguments:
    """Script-specific arguments."""

    # Model configuration
    student_model: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Student model name or path"},
    )
    teacher_model: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "Teacher model name or path"},
    )

    # Dataset configuration
    dataset_name: str = field(
        default="HuggingFaceTB/Countdown-Task-GOLD",
        metadata={"help": "Dataset name on HuggingFace"},
    )
    dataset_config: str = field(
        default="verified_Qwen3-4B-Instruct-2507",
        metadata={"help": "Dataset configuration/subset name"},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )


def main():
    # Parse arguments
    parser = TrlParser((ScriptArguments, GOLDConfig))
    script_args, training_args = parser.parse_args_and_config()

    # Set teacher model init kwargs to load on single device (avoid device_map='auto' splitting)
    # This is required when using DeepSpeed or multi-GPU training
    if training_args.teacher_model_init_kwargs is None:
        training_args.teacher_model_init_kwargs = {}
    training_args.teacher_model_init_kwargs["device_map"] = None
    # torch_dtype is required by GOLDTrainer when teacher_model_init_kwargs is set
    if "torch_dtype" not in training_args.teacher_model_init_kwargs:
        training_args.teacher_model_init_kwargs["torch_dtype"] = "bfloat16"

    # Load the Countdown dataset
    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        split=script_args.dataset_split,
    )

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
    print("GKD Training (Same-Tokenizer Distillation)")
    print("=" * 60)
    print(f"Student: {script_args.student_model}")
    print(f"Teacher: {script_args.teacher_model}")
    print(f"Dataset: {script_args.dataset_name} ({script_args.dataset_config})")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Lambda (on-policy fraction): {training_args.lmbda}")
    print(f"Beta (KL interpolation): {training_args.beta}")
    print(f"ULD Loss: {training_args.use_uld_loss}")
    print("=" * 60)

    trainer.train()

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
