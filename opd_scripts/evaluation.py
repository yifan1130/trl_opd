"""
Countdown Task Evaluation Script
Evaluates language models on the Countdown mathematical reasoning task using vLLM for efficient inference.
The Countdown task requires models to use a set of numbers exactly once to reach a target value.
Example usage:
    python countdown_eval.py \\
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
        --output_dir ./results
For more details on the Countdown task, see: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from trl import TrlParser


@dataclass
class EvaluateScriptArguments:
    """Arguments for Countdown task evaluation.
    Attributes:
        model_name_or_path: HuggingFace model ID or local path
        model_revision: Model revision/branch to use
        system_prompt: Optional custom system prompt to override default
        dataset_name: HuggingFace dataset ID for Countdown tasks
        dataset_split: Dataset split to evaluate on
        dataset_config: Dataset configuration name
        output_dir: Directory to save results and detailed outputs
        num_samples: Number of examples to evaluate (use -1 for full dataset)
        num_generations: Number of generations per example for pass@k computation
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        num_proc: Number of processes for evaluation
        seed: Random seed for dataset shuffling
    """
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Optional[str] = "main"
    system_prompt: Optional[str] = None
    dataset_name: str = "HuggingFaceTB/Countdown-Task-GOLD"
    dataset_split: str = "test"
    dataset_config: str = "test"
    output_dir: str = ""
    num_samples: int = 500
    num_generations: int = 4
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 4096
    batch_size: int = 16
    num_proc: int = 4
    seed: int = 42


def pass_at_k(n: int, c: int, k: int) -> float:
    """Computes pass@k metric given n samples and c correct samples.
    This metric estimates the probability that at least one of k samples is correct
    when sampling k times from n total samples with c correct ones.
    Implementation from https://arxiv.org/abs/2107.03374 (Codex paper)
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to consider (k in pass@k)
    Returns:
        pass@k score between 0.0 and 1.0
    Example:
        If we generate 4 solutions and 2 are correct, pass@1 estimates the probability
        that a single random sample would be correct.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate(example):
    """Evaluates model completions on the Countdown task.
    The Countdown task requires using a set of numbers exactly once with basic arithmetic
    operations (+, -, *, /) to reach a target value. Completions are evaluated on:
    1. Format: Answer must be in <answer>...</answer> tags
    2. Number usage: All numbers must be used exactly once
    3. Safety: Only numbers and basic operators allowed
    4. Correctness: Equation must evaluate to the target value
    Args:
        example: Dataset example containing:
            - generations: List of model-generated completions
            - target: Target value to reach
            - nums: List of available numbers
    Returns:
        Updated example dict with added fields:
            - accuracy: List of 0.0/1.0 for each generation
            - pass@1:4: Pass@1 metric from 4 generations
            - pass@4:1: Pass@4 metric from 4 generations
    """
    completions = example["generations"]
    target = example["target"]
    numbers = example["nums"]
    accuracy = []
    for completion in completions:
        try:
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                accuracy.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            equation = equation.split("=")[0].strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                accuracy.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                accuracy.append(0.0)
                continue
            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(target)) < 1e-5:
                accuracy.append(1.0)
            else:
                accuracy.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            accuracy.append(0.0)
    example["pass@1:4"] = pass_at_k(len(accuracy), sum(accuracy), 1)
    example["pass@4:1"] = pass_at_k(len(accuracy), sum(accuracy), 4)
    example["accuracy"] = accuracy
    return example


def batch_generation(batch, tokenizer, llm, sampling_params, script_args):
    # Generate responses for each prompt in the batch
    prompts = batch["prompt"]
    if script_args.system_prompt is not None:
        for sample in prompts:
            for turn in sample:
                if turn["role"] == "system":
                    turn["content"] = script_args.system_prompt

    prompts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True, continue_final_message=False
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    completions = []
    for output in outputs:
        generations = [gen.text for gen in output.outputs]
        completions.append(generations)
    batch["generations"] = completions
    return batch


def main(script_args: EvaluateScriptArguments):
    """Main evaluation pipeline."""
    print("=" * 80)
    print("Countdown Task Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {script_args.model_name_or_path}")
    print(f"  Revision: {script_args.model_revision}")
    print(f"  Dataset: {script_args.dataset_name} ({script_args.dataset_config}/{script_args.dataset_split})")
    print(f"  Samples: {script_args.num_samples if script_args.num_samples > 0 else 'all'}")
    print(f"  Generations per sample: {script_args.num_generations}")
    print(f"  Output: {script_args.output_dir}")
    print()

    # Load dataset using script arguments
    print(f"Loading dataset: {script_args.dataset_name}...")
    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        split=script_args.dataset_split
    )
    print(f"Loaded {len(dataset)} examples")

    # Load tokenizer and model
    print(f"\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        revision=script_args.model_revision,
        trust_remote_code=True
    )

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s)")

    llm = LLM(
        model=script_args.model_name_or_path,
        revision=script_args.model_revision,
        trust_remote_code=True,
        tensor_parallel_size=n_gpus,
    )
    print("Model loaded successfully")

    # Prepare sampling parameters
    print(f"\nSampling parameters:")
    print(f"  Temperature: {script_args.temperature}")
    print(f"  Top-k: {script_args.top_k}")
    print(f"  Top-p: {script_args.top_p}")
    print(f"  Max tokens: {script_args.max_tokens}")
    print(f"  Generations per example: {script_args.num_generations}")

    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_k=script_args.top_k,
        top_p=script_args.top_p,
        max_tokens=script_args.max_tokens,
        n=script_args.num_generations,
    )

    # Select subset of dataset if specified
    if script_args.num_samples > 0 and script_args.num_samples < len(dataset):
        print(f"\nShuffling and selecting {script_args.num_samples} samples (seed={script_args.seed})")
        dataset = dataset.shuffle(seed=script_args.seed).select(range(script_args.num_samples))
    else:
        print(f"\nEvaluating on full dataset ({len(dataset)} samples)")
    # Generate completions
    print(f"\nGenerating completions (batch_size={script_args.batch_size})...")
    dataset = dataset.map(
        batch_generation,
        batched=True,
        batch_size=script_args.batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "llm": llm,
            "sampling_params": sampling_params,
            "script_args": script_args
        },
    )
    print("Generation complete")

    # Evaluate completions
    print(f"\nEvaluating completions (num_proc={script_args.num_proc})...")
    dataset = dataset.map(evaluate, num_proc=script_args.num_proc)
    print("Evaluation complete")
    # Compute aggregate metrics
    print("\nComputing metrics...")
    scores = {"pass@1:4": 0.0, "pass@4:1": 0.0}
    for example in dataset:
        scores["pass@1:4"] += example["pass@1:4"]
        scores["pass@4:1"] += example["pass@4:1"]
    scores["pass@1:4"] /= len(dataset)
    scores["pass@4:1"] /= len(dataset)

    # Print results to console
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"pass@1:4: {scores['pass@1:4']:.4f}")
    print(f"pass@4:1: {scores['pass@4:1']:.4f}")
    print("=" * 80)

    # Save results
    results = {
        "config_general": {
            "model_name": script_args.model_name_or_path,
            "model_revision": script_args.model_revision,
            "dataset_name": script_args.dataset_name,
            "dataset_split": script_args.dataset_split,
            "dataset_config": script_args.dataset_config,
            "num_samples": len(dataset),
            "num_generations": script_args.num_generations,
            "temperature": script_args.temperature,
            "top_k": script_args.top_k,
            "top_p": script_args.top_p,
        },
        "results": {
            "all": {
                "pass@1:4": scores["pass@1:4"],
                "pass@4:1": scores["pass@4:1"],
            }
        },
    }

    results_path = Path(script_args.output_dir) / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    results_file = results_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {results_file}")

    # Save detailed outputs
    details_path = Path(script_args.output_dir) / "details"
    details_path.mkdir(parents=True, exist_ok=True)

    df = dataset.to_pandas()
    details_file = details_path / "details.parquet"
    df.to_parquet(details_file, index=False)
    print(f"Saved detailed outputs to {details_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = TrlParser((EvaluateScriptArguments,))
    script_args = parser.parse_args_and_config()[0]
    # Create output directory if it doesn't exist
    if script_args.output_dir:
        Path(script_args.output_dir).mkdir(parents=True, exist_ok=True)

    main(script_args)