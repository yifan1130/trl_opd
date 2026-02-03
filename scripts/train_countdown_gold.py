#!/usr/bin/env python
# Training script for Countdown task using GOLD (General Online Logit Distillation)
# GOLD enables cross-tokenizer distillation using ULD loss
#
# Teacher: Qwen/Qwen3-4B-Instruct-2507
# Student: Qwen/Qwen2.5-1.5B-Instruct
# Loss: Reverse KL (beta=1.0) with ULD for cross-tokenizer alignment

from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer


def main():
    # Model configuration
    student_model = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_model = "Qwen/Qwen3-4B-Instruct-2507"

    # Load the Countdown dataset (verified by Qwen3-4B teacher)
    dataset = load_dataset(
        "HuggingFaceTB/Countdown-Task-GOLD",
        "verified_Qwen3-4B-Instruct-2507",
        split="train",
    )

    print(f"Dataset size: {len(dataset)} examples")

    # Training configuration - GOLD with ULD for cross-tokenizer distillation
    training_args = GOLDConfig(
        output_dir="./outputs/gold-countdown-qwen3-4b-to-qwen2.5-1.5b",

        # Learning rate - use 1e-5 to 5e-5 (NOT the default 1e-7)
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # Batch size and training duration
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=3,

        # Generation settings
        max_completion_length=512,
        temperature=1.0,

        # ========== GOLD-specific: Cross-tokenizer ULD loss ==========
        use_uld_loss=True,  # Enable ULD for cross-tokenizer alignment
        uld_use_hybrid_loss=True,
        teacher_tokenizer_name_or_path=teacher_model,
        uld_crossentropy_weight=0.0,
        uld_distillation_weight=1.0,

        # On-policy distillation settings
        lmbda=1.0,  # Fully on-policy (best performance per blog)
        beta=1.0,   # Reverse KL divergence (beta=0 is forward KL, beta=1 is reverse KL)

        # Precision
        bf16=True,

        # Logging and saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # Optional: enable completion logging
        log_completions=True,
        log_completions_steps=100,
    )

    # Initialize trainer
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print("=" * 60)
    print("GOLD Training (Cross-Tokenizer Distillation)")
    print("=" * 60)
    print(f"Student: {student_model}")
    print(f"Teacher: {teacher_model}")
    print(f"Dataset: HuggingFaceTB/Countdown-Task-GOLD (verified_Qwen3-4B-Instruct-2507)")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Lambda (on-policy fraction): {training_args.lmbda}")
    print(f"Beta (KL interpolation): {training_args.beta} (Reverse KL)")
    print(f"ULD Loss: {training_args.use_uld_loss}")
    print("=" * 60)

    trainer.train()

    # Save the final model
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
