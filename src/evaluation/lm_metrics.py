"""
Language modeling evaluation metrics.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


@torch.no_grad()
def evaluate_perplexity(
    model,
    dataloader,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> float:
    """
    Evaluate perplexity on a language modeling dataset.
    
    Args:
        model: Language model (should return loss or logits)
        dataloader: DataLoader with batches containing 'input_ids' and 'labels'
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        verbose: Whether to print progress
    
    Returns:
        Perplexity (exp of average cross-entropy loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        
        # Get loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            # HuggingFace models return loss directly
            loss = outputs.loss
        else:
            # Compute loss from logits if not provided
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # Count tokens (excluding padding if any)
        n_tokens = (labels != -100).sum().item()
        
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        num_batches += 1
        
        if verbose and (i + 1) % 10 == 0:
            current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
            print(f"  Batch {i+1}/{len(dataloader) if max_batches is None else max_batches}: "
                  f"Current PPL = {current_ppl:.3f}")
    
    if total_tokens == 0:
        raise ValueError("No tokens processed during evaluation")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    if verbose:
        print(f"\nEvaluation complete:")
        print(f"  Total batches: {num_batches}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.3f}")
    
    return perplexity

