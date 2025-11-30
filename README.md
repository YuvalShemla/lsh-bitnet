# LSH-BitNet: Locality-Sensitive Hashing Attention for BitNet Models

A modular framework for experimenting with LSH-based attention mechanisms on BitNet models, enabling efficient attention computation through locality-sensitive hashing.

## Overview

This repository provides a flexible framework to:
- Load BitNet models from HuggingFace
- Replace standard attention with LSH-based attention variants
- Experiment with different hashing strategies (bit-sampling, SimHash, etc.)
- Evaluate performance on various language modeling datasets
- Compare efficiency (attention comparisons) vs accuracy trade-offs

---

## Repository Structure

```
lsh-bitnet/
├── src/                           # Main source code
│   ├── config.py                  # Configuration dataclasses
│   ├── attention/                 # Attention backends
│   │   ├── wrapper.py            # AttentionWrapper (main integration)
│   │   ├── full_attention.py    # Baseline full attention
│   │   ├── base_lsh.py           # LSH attention base class
│   │   ├── bit_sampling_lsh.py   # Bit-sampling LSH attention
│   │   └── simhash_lsh.py         # SimHash LSH attention (future)
│   ├── hash_functions/           # Hashing strategies
│   │   ├── base.py               # HashingStrategy interface
│   │   ├── bit_sampling.py       # Bit-sampling hasher
│   │   └── simhash.py            # SimHash hasher (future)
│   ├── indexing/                 # Indexing functions (future)
│   │   └── __init__.py
│   ├── data_interface/           # Dataset loaders
│   │   ├── enwik8.py             # EnWik8 dataset
│   │   ├── wikitext103.py        # WikiText-103 dataset
│   │   └── synthetic_ny_adj.py   # Synthetic New York adjective task
│   ├── evaluation/                # Evaluation metrics
│   │   ├── lm_metrics.py         # Perplexity evaluation
│   │   ├── synthetic_metrics.py  # Accuracy evaluation for synthetic tasks
│   │   └── comparison_logger.py  # Comparison tracking
│   ├── model_adapters/            # Model loading & adapters
│   │   ├── adapters.py           # Model adapter pattern
│   │   ├── registry.py           # Model registry
│   │   └── bitnet_hf.py          # BitNet-specific utilities
│   └── utils/                     # Utilities
│       ├── logging.py
│       └── seed.py
├── scripts/                       # Evaluation scripts
│   ├── bitnet_baseline_wikitext103.py  # WikiText-103 perplexity baseline
│   └── eval_synthetic_ny_adj.py        # Synthetic task accuracy baseline
├── configs/                       # Experiment configurations
│   ├── wt2_bitnet_full.yaml
│   ├── wt2_bitnet_lsh_bitsampling.yaml
│   └── enwik8_bitnet_lsh_bitsampling.yaml
├── models/                        # HuggingFace model cache (data)
│   └── models--microsoft--bitnet-b1.58-2B-4T/
├── data/                          # Dataset cache
│   ├── enwik8                     # EnWik8 raw data
│   └── wikitext103/               # WikiText-103 cache
├── results/                       # Evaluation results
│   ├── baseline_wikitext103_*.json
│   └── baseline_synthetic_ny_adj_*.json
└── tests/                         # Unit tests
```

---

## Current Status

### ✅ Baseline Evaluations Completed

We have established baseline performance on two dataset types:

#### 1. **WikiText-103 Perplexity Baseline**
- **Dataset**: WikiText-103 validation split
- **Evaluation**: Language modeling perplexity
- **Coverage**: 252,672 tokens (full validation set)
- **Configuration**: 987 batches × 2 sequences × 128 tokens
- **Result**: **Perplexity = 33.12**
- **File**: `results/baseline_wikitext103_validation_all_all_*.json`

#### 2. **Synthetic New York Adjective Task Baseline**
- **Dataset**: Synthetic task requiring tracking 2nd mention of "New York"
- **Evaluation**: Accuracy (extracting correct adjective)
- **Samples**: 50 examples
- **Result**: **Accuracy = 40.0%** (20/50 correct)
- **File**: `results/baseline_synthetic_ny_adj_50samples_*.json`

**Note**: The synthetic task accuracy is low because the model is not fine-tuned for this specific task. This baseline will serve as a comparison point for LSH attention implementations.

### 📊 Datasets

#### WikiText-103
- **Source**: HuggingFace `Salesforce/wikitext` (wikitext-103-raw-v1)
- **Type**: Wikipedia articles (raw text)
- **Size**: 
  - Train: 1,801,350 examples
  - Validation: 3,760 examples (~252,725 tokens)
  - Test: 4,358 examples
- **Purpose**: Standard language modeling benchmark
- **Evaluation**: Perplexity (lower is better)

#### Synthetic New York Adjective Task (toy example)
- **Type**: Synthetically generated examples
- **Task**: Extract the adjective before the Nth mention of "New York"
- **Purpose**: Test attention mechanisms (requires long-range dependency tracking)
- **Evaluation**: Accuracy (higher is better)
- **Why useful**: 
  - Explicitly tests attention capabilities
  - Easy to control difficulty (distance between mentions)
  - Interpretable results

#### Future Datasets
We plan to expand to additional datasets in future steps:
- WikiText-2 (smaller, faster evaluation)
- EnWik8 (character-level, different tokenization)
- Other language modeling benchmarks

---

## BitNet Model Architecture

### Model: `microsoft/bitnet-b1.58-2B-4T`

#### Top-Level Architecture

```
BitNetForCausalLM
└── BitNetModel (attribute: 'model')
    ├── embed_tokens: Embedding layer
    ├── layers: ModuleList (30 decoder layers)
    └── norm: BitNetRMSNorm (final layer norm)
```

#### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Hidden Size** | 2560 | Model dimension (d_model) |
| **Number of Layers** | 30 | Transformer decoder layers |
| **Attention Heads** | 20 | Query attention heads |
| **Key-Value Heads** | 5 | Key/Value attention heads (GQA) |
| **Head Dimension** | 128 | Dimension per attention head |
| **Intermediate Size** | 6912 | FFN hidden dimension |
| **Max Position Embeddings** | 4096 | Maximum sequence length |
| **Vocab Size** | 128256 | Vocabulary size |
| **Activation** | relu2 | ReLU² activation function |
| **RMS Norm Eps** | 1e-05 | Layer normalization epsilon |

**Key Feature: Grouped Query Attention (GQA)**
- Query projection: 20 heads × 128 dims = 2560 dims
- Key/Value projections: 5 heads × 128 dims = 640 dims each
- GQA ratio: 4:1 (each KV head serves 4 Q heads)

#### Decoder Layer Structure

Each `BitNetDecoderLayer` contains:

```
BitNetDecoderLayer
├── input_layernorm: BitNetRMSNorm
├── self_attn: BitNetAttention  ⬅️ TARGET FOR LSH WRAPPER
│   ├── q_proj: AutoBitLinear (2560 → 2560)  [20 heads]
│   ├── k_proj: AutoBitLinear (2560 → 640)   [5 heads]
│   ├── v_proj: AutoBitLinear (2560 → 640)   [5 heads]
│   ├── o_proj: AutoBitLinear (2560 → 2560)  [output projection]
│   └── attn_sub_norm: BitNetRMSNorm
├── post_attention_layernorm: BitNetRMSNorm
└── mlp: BitNetMLP
    ├── gate_proj: AutoBitLinear
    ├── up_proj: AutoBitLinear
    ├── down_proj: AutoBitLinear
    └── ffn_sub_norm: BitNetRMSNorm
```

#### Attention Module Details

**Class:** `BitNetAttention`

**Forward Signature:**
```python
forward(
    hidden_states: torch.Tensor,                    # [B, T, 2560]
    position_embeddings: tuple,                      # RoPE embeddings
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> tuple
```

**Tensor Shapes:**
- Input: `[batch_size, seq_len, 2560]`
- Q projection output: `[batch_size, seq_len, 2560]` → reshaped to `[B, 20, T, 128]`
- K projection output: `[batch_size, seq_len, 640]` → reshaped to `[B, 5, T, 128]`
- V projection output: `[batch_size, seq_len, 640]` → reshaped to `[B, 5, T, 128]`

**Attention Computation (Standard):**
1. Q, K, V are projected from hidden states
2. Q is split into 20 heads, K/V into 5 heads each
3. For GQA, each KV head is broadcast to 4 Q heads
4. Attention scores computed: `Q @ K^T / sqrt(128)`
5. Causal masking applied (lower triangular)
6. Softmax and attention-weighted sum: `softmax(scores) @ V`
7. Output projection: `[B, T, 2560]`

**Full Attention Complexity**: O(T²) comparisons per head

---

## Implementation Plan

### Phase 1: Simple Bit-Sampling LSH Attention (Current Focus)

**Goal**: Implement basic LSH-based attention using bit-sampling to partition tokens into buckets.

#### Approach:
1. **Attention Wrapper**: Replace `BitNetAttention` modules with `AttentionWrapper`
   - Wrapper reuses Q/K/V/O projections from original module
   - Intercepts attention computation
   - Delegates to LSH backend when enabled

2. **Bit-Sampling LSH**:
   - Binarize Q, K tensors (threshold at 0.0)
   - Sample random bit positions for hashing
   - Compute hash buckets for each token position
   - Group queries and keys by bucket ID
   - Compute attention only within matching buckets

3. **Metrics to Track**:
   - **Accuracy**: Perplexity on WikiText-103, accuracy on synthetic task
   - **Efficiency**: Number of query-key comparisons made
   - **Comparison Ratio**: LSH comparisons / Full attention comparisons

#### Expected Outcomes:
- Reduced attention comparisons (from O(T²) to O(T·B) where B << T)
- Some accuracy degradation (to be measured)
- Clear efficiency vs accuracy trade-off curve

### Phase 2: LSH Forest for Dynamic Candidate Selection

**Goal**: Implement LSH Forest to allow dynamic candidate selection without pre-specifying number of buckets.

#### Approach:
1. **Multiple Hash Tables**: Use multiple independent hash functions
2. **Union of Matches**: For each query, find matching keys across all tables
3. **Dynamic Bucket Selection**: Automatically determine relevant buckets per query
4. **Adaptive Thresholding**: Adjust candidate set size based on query characteristics

#### Advantages over Simple Bucketing:
- No need to pre-specify bucket count
- Better recall (finds more relevant keys)
- More robust to hash collisions
- Can adapt to different query types

### Phase 3: Comprehensive Comparison

**Evaluation Framework**:
1. **Full Attention** (baseline)
   - Complexity: O(T²) comparisons
   - Accuracy: Baseline (current results)

2. **Bucket Attention** (Phase 1)
   - Complexity: O(T·B) comparisons (B = average bucket size)
   - Accuracy: To be measured
   - Trade-off: Speedup vs accuracy loss

3. **LSH Forest Attention** (Phase 2)
   - Complexity: O(T·C) comparisons (C = dynamic candidate set)
   - Accuracy: To be measured
   - Trade-off: Better accuracy than buckets, more comparisons

**Metrics to Compare**:
- **Perplexity** on WikiText-103
- **Accuracy** on synthetic task
- **Number of comparisons** per forward pass
- **Speedup factor** vs full attention
- **Memory usage** (if applicable)

---

## Usage

### Running Baseline Evaluations

**WikiText-103 Perplexity:**
```bash
python scripts/bitnet_baseline_wikitext103.py \
    --max-batches 500 \
    --batch-size 2 \
    --seq-len 128 \
    --split validation \
    --device cpu
```

**Synthetic Task:**
```bash
python scripts/eval_synthetic_ny_adj.py \
    --n-samples 50 \
    --batch-size 1 \
    --device cpu
```

### Results

Results are automatically saved to `results/` folder with descriptive filenames:
- `baseline_wikitext103_validation_all_500batches_<timestamp>.json`
- `baseline_synthetic_ny_adj_50samples_<timestamp>.json`

---

## Next Steps

1. ✅ **Baseline evaluations** - Completed
2. ⏳ **Implement attention wrapper** - Replace BitNetAttention with wrapper
3. ⏳ **Implement bit-sampling LSH** - Basic bucket-based attention
4. ⏳ **Compare efficiency vs accuracy** - Measure attention comparisons
5. ⏳ **Implement LSH Forest** - Dynamic candidate selection
6. ⏳ **Comprehensive comparison** - Full vs Bucket vs Forest attention

---

## Notes

- **Model Cache**: The `models/` directory contains HuggingFace cached model files (data)
- **Code**: The `src/model_adapters/` directory contains Python code for model adapters
- **GQA**: BitNet uses Grouped Query Attention, which our wrapper will handle correctly
- **Quantization**: BitNet uses 1.58-bit quantization (`AutoBitLinear`), which we preserve
