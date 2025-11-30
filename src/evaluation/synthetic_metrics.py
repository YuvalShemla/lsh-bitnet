"""
Evaluation metrics for synthetic tasks.
"""

import torch
from typing import List, Dict, Tuple, Optional


def compute_accuracy_with_details(
    model,
    dataloader,
    tokenizer,
    device,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    max_new_tokens: int = 20,
) -> Tuple[float, int, int, List[Dict]]:
    """
    Compute accuracy on synthetic tasks with detailed logging.
    
    Args:
        model: Language model
        dataloader: DataLoader with batches containing 'prompt', 'answer', etc.
        tokenizer: Tokenizer for decoding
        device: Device to run evaluation on
        max_batches: Maximum number of batches to evaluate (None = all)
        verbose: Whether to print detailed examples
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Tuple of (accuracy, correct_count, total_count, examples_list)
        examples_list contains dicts with 'prompt', 'expected', 'generated', 'match'
    """
    model.eval()
    correct = 0
    total = 0
    examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            prompts = batch["prompt"]  # List of prompt strings
            answers = batch["answer"]  # List of expected answers
            
            for batch_idx in range(len(prompts)):
                prompt_text = prompts[batch_idx]
                expected_answer = answers[batch_idx]
                
                # Tokenize prompt
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
                prompt_tokens = prompt_tokens.to(device)
                
                # Generate
                try:
                    generated = model.generate(
                        prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=1.0,
                        # Stop generation after common sentence endings (period, newline, etc.)
                        # This helps stop after the answer word
                    )
                    
                    # Decode full generation
                    full_generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
                    
                    # Remove the prompt part to get only the generated text
                    # Find where prompt ends in the generated sequence
                    prompt_length = prompt_tokens.shape[1]
                    generated_only_tokens = generated[0][prompt_length:]
                    generated_only_text = tokenizer.decode(generated_only_tokens, skip_special_tokens=True).strip()
                    
                    # Extract just the first word/token (before any punctuation or new sentence)
                    # Split by common sentence separators and take first part
                    first_word = generated_only_text.split('.')[0].split('\n')[0].split(',')[0].split(';')[0].strip()
                    # Remove quotes and any trailing punctuation
                    first_word = first_word.strip('"\'`').rstrip('.,;:!?').strip()
                    
                    # Check if expected answer appears in the first word or anywhere in generation
                    expected_clean = expected_answer.lower().strip()
                    first_word_lower = first_word.lower()
                    generated_lower = generated_only_text.lower()
                    
                    # Match if answer is in first word OR anywhere in generation
                    matched = (expected_clean == first_word_lower) or (expected_clean in generated_lower)
                    
                    # Store the cleaned first word for display
                    generated_only_text = first_word if first_word else generated_only_text
                    
                    if matched:
                        correct += 1
                    total += 1
                    
                    # Store example details
                    example = {
                        'prompt': prompt_text,
                        'expected': expected_answer,
                        'generated': generated_only_text,
                        'full_generated': full_generated_text,
                        'match': matched,
                    }
                    examples.append(example)
                    
                    if verbose and total <= 10:  # Print first 10 examples
                        print(f"\n  Example {total}:")
                        print(f"    Prompt: {prompt_text[:150]}...")
                        print(f"    Expected: '{expected_answer}'")
                        print(f"    Generated (only): '{generated_only_text}'")
                        print(f"    Match: {matched}")
                        
                except Exception as e:
                    total += 1
                    example = {
                        'prompt': prompt_text,
                        'expected': expected_answer,
                        'generated': f'ERROR: {str(e)}',
                        'full_generated': '',
                        'match': False,
                    }
                    examples.append(example)
                    if verbose and total <= 10:
                        print(f"  Example {total}: Generation error: {e}")
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total, examples


def print_examples_summary(examples: List[Dict], num_examples: int = 5):
    """
    Print a summary of examples with full input/output.
    
    Args:
        examples: List of example dictionaries from compute_accuracy_with_details
        num_examples: Number of examples to print
    """
    print("\n" + "=" * 80)
    print("DETAILED EXAMPLES: Prompt and Model Generation")
    print("=" * 80)
    
    for i, ex in enumerate(examples[:num_examples]):
        print(f"\n--- Example {i+1} ---")
        print(f"PROMPT:")
        print(f"  {ex['prompt']}")
        print(f"\nEXPECTED ANSWER: '{ex['expected']}'")
        print(f"\nMODEL GENERATED (without prompt):")
        print(f"  {ex['generated']}")
        print(f"\nFULL GENERATED SEQUENCE (with prompt):")
        print(f"  {ex['full_generated'][:300]}...")
        print(f"\nMATCH: {ex['match']}")
        print("-" * 80)

