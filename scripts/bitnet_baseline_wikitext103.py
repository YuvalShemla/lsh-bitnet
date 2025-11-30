#!/usr/bin/env python3
"""
Baseline evaluation script for BitNet on WikiText-103 dataset.

This script evaluates the perplexity of BitNet model on WikiText-103 dataset
without any LSH modifications - just the baseline model performance.
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

# Add parent directory to path to import lsh_bitnet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_interface import get_wikitext103_data_loader
from src.evaluation.lm_metrics import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BitNet baseline on WikiText-103 dataset"
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
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to evaluate (None = all)",
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
    print("BitNet Baseline Evaluation on WikiText-103")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Max batches: {args.max_batches if args.max_batches else 'all'}")
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
    print(f"✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")
    
    print(f"\nLoading model with dtype {model_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=model_dtype,
    )
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded and moved to {device}")
    
    # Print model info
    config = model.config
    print(f"\nModel configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Number of attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Load dataset
    print(f"\nLoading WikiText-103 {args.split} split...")
    dataloader = get_wikitext103_data_loader(
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
    )
    print(f"✓ Dataset loaded ({len(dataloader)} batches)")
    
    # Evaluate
    print(f"\nEvaluating perplexity...")
    print("-" * 80)
    perplexity = evaluate_perplexity(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
        verbose=True,
    )
    
    print("=" * 80)
    print(f"FINAL RESULT: Perplexity = {perplexity:.3f}")
    print("=" * 80)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_name = args.split
    samples_str = f"{args.max_samples}samples" if args.max_samples else "all"
    batches_str = f"{args.max_batches}batches" if args.max_batches else "all"
    result_file = f"results/baseline_wikitext103_{split_name}_{samples_str}_{batches_str}_{timestamp}.json"
    
    results = {
        "model": args.model_name,
        "device": str(device),
        "dtype": str(model_dtype),
        "split": args.split,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
        "max_batches": args.max_batches,
        "perplexity": perplexity,
        "timestamp": timestamp,
    }
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_file}")
    
    return perplexity


if __name__ == "__main__":
    main()

