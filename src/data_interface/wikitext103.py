"""
WikiText-103 dataset loader using HuggingFace datasets.

WikiText-103 is a large language modeling dataset derived from Wikipedia.
Caches the dataset locally in the data folder.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from datasets import load_dataset


class WikiText103Dataset(Dataset):
    """Dataset for WikiText-103."""
    
    def __init__(
        self,
        tokenizer,
        split: str = "train",
        seq_len: int = 256,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            split: Dataset split ('train', 'validation', 'test')
            seq_len: Sequence length for language modeling
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        
        # Load dataset from HuggingFace (with local caching)
        data_dir = os.path.join("data", "wikitext103")
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"Loading WikiText-103 {split} split from HuggingFace...")
        print(f"  (Caching in {data_dir})")
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
            cache_dir=data_dir,
        )
        
        # Tokenize all texts
        print(f"Tokenizing {len(dataset)} examples...")
        all_tokens = []
        for example in dataset:
            text = example["text"]
            if text.strip():  # Skip empty texts
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        
        print(f"Total tokens: {len(all_tokens):,}")
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - seq_len, seq_len):
            if max_samples and len(self.sequences) >= max_samples:
                break
            self.sequences.append(all_tokens[i:i + seq_len + 1])  # +1 for labels
        
        print(f"Created {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # HuggingFace models expect labels == input_ids and handle shifting internally
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = input_ids.clone()  # Same as input_ids, model will shift internally
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def get_wikitext103_data_loader(
    tokenizer,
    split: str = "validation",
    batch_size: int = 4,
    seq_len: int = 256,
    max_samples: Optional[int] = None,
    data_dir: str = "data",
) -> DataLoader:
    """
    Get DataLoader for WikiText-103 dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        seq_len: Sequence length
        max_samples: Maximum number of samples (None = use all)
    
    Returns:
        DataLoader for WikiText-103
    """
    dataset = WikiText103Dataset(
        tokenizer=tokenizer,
        split=split,
        seq_len=seq_len,
        max_samples=max_samples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable for CPU/MPS compatibility
    )

