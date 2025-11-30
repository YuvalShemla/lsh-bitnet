"""
EnWik8 dataset loader for language modeling evaluation.

EnWik8 is a character-level dataset derived from Wikipedia.
For transformer models, we tokenize it using the model's tokenizer.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import requests
from tqdm import tqdm


class EnWik8Dataset(Dataset):
    """Dataset for EnWik8 (character-level Wikipedia text)."""
    
    def __init__(
        self,
        tokenizer,
        data_path: str,
        seq_len: int = 256,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            data_path: Path to enwik8 data file
            seq_len: Sequence length for language modeling
            max_samples: Maximum number of samples to use (None = use all)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        
        # Load and tokenize data
        print(f"Loading EnWik8 from {data_path}...")
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        print(f"Tokenizing {len(text)} characters...")
        # Tokenize the entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            if max_samples and len(self.sequences) >= max_samples:
                break
            self.sequences.append(tokens[i:i + seq_len + 1])  # +1 for labels
        
        print(f"Created {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def download_enwik8(data_dir: str = "data") -> str:
    """
    Download EnWik8 dataset if not already present.
    
    Returns:
        Path to the enwik8 data file
    """
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "enwik8")
    
    if os.path.exists(data_path):
        print(f"EnWik8 already exists at {data_path}")
        return data_path
    
    url = "http://mattmahoney.net/dc/enwik8.zip"
    zip_path = os.path.join(data_dir, "enwik8.zip")
    
    print(f"Downloading EnWik8 from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Extract (enwik8.zip contains just enwik8 file)
    import zipfile
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files
        zip_ref.extractall(data_dir)
        # Check if enwik8 was extracted to a subdirectory
        extracted_files = zip_ref.namelist()
        if len(extracted_files) == 1 and extracted_files[0] != "enwik8":
            # If extracted to a subdirectory, move it
            extracted_path = os.path.join(data_dir, extracted_files[0])
            if os.path.exists(extracted_path) and os.path.exists(data_path):
                # Already in right place
                pass
            elif os.path.exists(extracted_path):
                os.rename(extracted_path, data_path)
    
    # Clean up zip file
    os.remove(zip_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"EnWik8 file not found after extraction. "
            f"Expected at {data_path}"
        )
    
    print(f"EnWik8 downloaded and extracted to {data_path}")
    return data_path


def get_enwik8_data_loader(
    tokenizer,
    split: str = "train",
    batch_size: int = 4,
    seq_len: int = 256,
    max_samples: Optional[int] = None,
    data_dir: str = "data",
    download: bool = True,
) -> DataLoader:
    """
    Get DataLoader for EnWik8 dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split (for enwik8, we use the same file for all splits)
        batch_size: Batch size
        seq_len: Sequence length
        max_samples: Maximum number of samples (None = use all)
        data_dir: Directory to store/load data
        download: Whether to download if not present
    
    Returns:
        DataLoader for EnWik8
    """
    data_path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(data_path):
        if download:
            data_path = download_enwik8(data_dir)
        else:
            raise FileNotFoundError(
                f"EnWik8 data not found at {data_path}. "
                "Set download=True to download automatically."
            )
    
    dataset = EnWik8Dataset(
        tokenizer=tokenizer,
        data_path=data_path,
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

