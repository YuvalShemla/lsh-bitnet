"""
Synthetic "New York adjective" dataset for testing attention mechanisms.

This dataset creates sequences where the model must track the second mention
of "New York" and extract the adjective preceding it. This is a perfect test
for attention mechanisms and LSH-based attention.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict


ADJECTIVES = [
    "busy", "quiet", "ancient", "modern", "noisy",
    "peaceful", "crowded", "famous", "historic", "vibrant",
    "bustling", "serene", "chaotic", "tranquil", "energetic",
]

DECOY_CITIES = ["Paris", "London", "Berlin", "Tokyo", "Rome", "Chicago", "Barcelona", "Madrid"]

QUESTION_TEMPLATE = (
    "What was the adjective before the {ordinal} time the city New York was mentioned?"
)

ORDINALS = {
    2: "second",
    3: "third",
    4: "fourth",
}


def generate_example(
    n_mention: int = 2,
    min_sentences: int = 3,
    max_sentences: int = 6,
    include_decoy: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """
    Generate a single example.
    
    Args:
        n_mention: Which mention to extract (2 = second, 3 = third, etc.)
        min_sentences: Minimum number of sentences
        max_sentences: Maximum number of sentences
        include_decoy: Whether to include distractor cities
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'context', 'question', 'answer', 'all_adjectives'
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample adjectives for New York mentions
    available_adjs = ADJECTIVES.copy()
    ny_adjectives = random.sample(available_adjs, n_mention)
    target_adj = ny_adjectives[n_mention - 1]  # 0-indexed, so n_mention-1
    
    # Remove used adjectives from pool
    for adj in ny_adjectives:
        if adj in available_adjs:
            available_adjs.remove(adj)
    
    # Generate sentences
    sentences = []
    ny_mention_count = 0
    
    # First sentence always has first NY mention
    s1 = f"Last year I visited the {ny_adjectives[0]} New York with some friends."
    sentences.append(s1)
    ny_mention_count += 1
    
    # Add filler sentences and remaining NY mentions
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences_added = 1
    
    while ny_mention_count < n_mention or sentences_added < num_sentences:
        if ny_mention_count < n_mention:
            # Add next NY mention
            adj = ny_adjectives[ny_mention_count]
            time_phrase = random.choice([
                "Later in life",
                "A few years later",
                "After some time",
                "Eventually",
                "Years later",
            ])
            action = random.choice([
                "I decided to move to",
                "I returned to",
                "I went back to",
                "I found myself in",
            ])
            s = f"{time_phrase}, {action} the {adj} New York to start a new chapter."
            sentences.append(s)
            ny_mention_count += 1
        else:
            # Add filler sentence
            filler = random.choice([
                "We spent a lot of time exploring museums and trying new food.",
                "The experience was unforgettable and changed my perspective.",
                "I took hundreds of photos and met many interesting people.",
                "The trip opened my eyes to new cultures and ways of life.",
                "I learned a lot about history and art during my visit.",
            ])
            sentences.append(filler)
        
        sentences_added += 1
    
    # Optionally add decoy city
    if include_decoy and available_adjs:
        decoy_adj = random.choice(available_adjs)
        decoy_city = random.choice(DECOY_CITIES)
        decoy_sentence = f"My cousin still prefers the {decoy_adj} city of {decoy_city}."
        sentences.append(decoy_sentence)
    
    # Shuffle sentences (but keep first one first to ensure order)
    if len(sentences) > 1:
        first = sentences[0]
        rest = sentences[1:]
        random.shuffle(rest)
        sentences = [first] + rest
    
    context = " ".join(sentences)
    
    # Generate question
    ordinal = ORDINALS.get(n_mention, f"{n_mention}th")
    question = QUESTION_TEMPLATE.format(ordinal=ordinal)
    
    return {
        "context": context,
        "question": question,
        "answer": target_adj,
        "all_adjectives": ny_adjectives,  # For debugging
        "n_mention": n_mention,
    }


class NewYorkAdjectiveDataset(Dataset):
    """Dataset for synthetic New York adjective task."""
    
    def __init__(
        self,
        tokenizer,
        n_samples: int = 1000,
        n_mention: int = 2,
        seq_len: Optional[int] = None,
        min_sentences: int = 3,
        max_sentences: int = 6,
        include_decoy: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            n_samples: Number of examples to generate
            n_mention: Which mention to extract (2 = second, 3 = third, etc.)
            seq_len: Maximum sequence length (None = no truncation)
            min_sentences: Minimum sentences per example
            max_sentences: Maximum sentences per example
            include_decoy: Whether to include distractor cities
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.n_mention = n_mention
        self.seq_len = seq_len
        
        print(f"Generating {n_samples} synthetic examples (n_mention={n_mention})...")
        random.seed(seed)
        
        self.examples = []
        for i in range(n_samples):
            ex = generate_example(
                n_mention=n_mention,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                include_decoy=include_decoy,
                seed=seed + i if seed is not None else None,
            )
            self.examples.append(ex)
        
        print(f"✓ Generated {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Format prompt: "Context: ... Question: ... Answer: (one word only)"
        prompt = f"Context: {ex['context']} Question: {ex['question']} Answer with one word only:"
        answer = ex["answer"]
        
        # Return prompt and answer separately - no need to tokenize here
        return {
            'prompt': prompt,
            'answer': answer,
            'all_adjectives': ex['all_adjectives'],  # For debugging
        }


def get_synthetic_ny_adj_data_loader(
    tokenizer,
    n_samples: int = 1000,
    n_mention: int = 2,
    batch_size: int = 4,
    seq_len: Optional[int] = 512,
    min_sentences: int = 3,
    max_sentences: int = 6,
    include_decoy: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Get DataLoader for synthetic New York adjective dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        n_samples: Number of examples to generate
        n_mention: Which mention to extract (2 = second, 3 = third, etc.)
        batch_size: Batch size
        seq_len: Maximum sequence length
        min_sentences: Minimum sentences per example
        max_sentences: Maximum sentences per example
        include_decoy: Whether to include distractor cities
        seed: Random seed
    
    Returns:
        DataLoader for synthetic dataset
    """
    dataset = NewYorkAdjectiveDataset(
        tokenizer=tokenizer,
        n_samples=n_samples,
        n_mention=n_mention,
        seq_len=seq_len,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        include_decoy=include_decoy,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=0,
        pin_memory=False,
    )

