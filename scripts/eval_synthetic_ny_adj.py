#!/usr/bin/env python3
"""
Evaluation script for synthetic New York adjective dataset.

This evaluates the model's ability to track the second (or Nth) mention
of "New York" and extract the adjective preceding it.
"""

import argparse
import json
import os
import sys
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_interface import get_synthetic_ny_adj_data_loader
from src.evaluation.synthetic_metrics import compute_accuracy_with_details, print_examples_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BitNet on synthetic New York adjective task"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/bitnet-b1.58-2B-4T",
        help="Model name or path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models",
        help="Directory to cache models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of synthetic examples",
    )
    parser.add_argument(
        "--n-mention",
        type=int,
        default=2,
        help="Which mention to extract (2=second, 3=third, etc.)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to evaluate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("Synthetic New York Adjective Task Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Extracting {args.n_mention}{'th' if args.n_mention > 3 else ['', 'st', 'nd', 'rd'][args.n_mention]} mention")
    print()
    
    # Determine dtype
    if args.dtype == "auto":
        if device.type == "mps":
            model_dtype = torch.float16
        elif device.type == "cuda":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model_dtype = dtype_map[args.dtype]
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    print(f"✓ Tokenizer loaded")
    
    print(f"\nLoading model with dtype {model_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=model_dtype,
    )
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded and moved to {device}")
    
    # Load dataset
    print(f"\nGenerating synthetic dataset...")
    dataloader = get_synthetic_ny_adj_data_loader(
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        n_mention=args.n_mention,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=42,
    )
    print(f"✓ Dataset loaded ({len(dataloader)} batches)")
    
    # Evaluate accuracy with detailed examples
    print(f"\nEvaluating accuracy...")
    print("-" * 80)
    accuracy, correct, total, examples = compute_accuracy_with_details(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        max_batches=args.max_batches,
        verbose=True,
    )
    
    # Print detailed examples (show all examples)
    print_examples_summary(examples, num_examples=len(examples))
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("=" * 80)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"results/baseline_synthetic_ny_adj_{args.n_samples}samples_{timestamp}.json"
    
    results = {
        "model": args.model_name,
        "device": str(device),
        "dtype": str(model_dtype),
        "n_samples": args.n_samples,
        "n_mention": args.n_mention,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "timestamp": timestamp,
        "examples": [
            {
                "prompt": ex["prompt"],
                "expected": ex["expected"],
                "generated": ex["generated"],
                "match": ex["match"],
            }
            for ex in examples
        ],
    }
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_file}")
    
    return accuracy


if __name__ == "__main__":
    main()

