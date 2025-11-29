# Gated DeltaNet Quick Reference

## Key Concepts

### What is Gated DeltaNet?

Gated DeltaNet is a linear attention mechanism that achieves O(n) complexity instead of O(n²) like standard Transformers, while maintaining competitive performance.

**Key Components:**
1. **Delta Rule**: Efficient state update mechanism
2. **Gating**: Output gating similar to Mamba2
3. **Short Convolutions**: 1D convolutions for local context
4. **Negative Eigenvalues** (optional): Improved state tracking

### Architecture Parameters

```python
GatedDeltaNet(
    hidden_size=2048,        # Model dimension
    expand_v=2.0,            # Value expansion ratio
    head_dim=256,            # Dimension per head
    num_heads=8,             # Number of heads (num_heads * head_dim = 0.75 * hidden_size)
    num_v_heads=None,        # Value heads (None = same as num_heads)
    mode='chunk',            # 'chunk' (training) or 'fused_recurrent' (inference)
    use_gate=True,           # Use output gating (recommended)
    use_short_conv=True,     # Use short convolutions (crucial!)
    allow_neg_eigval=False,  # Allow negative eigenvalues
    conv_size=4,             # Convolution kernel size
    conv_bias=False,         # Convolution bias
    norm_eps=1e-5,           # RMSNorm epsilon
)
```

### Parameter Allocation

When `use_gate=True` (~6M parameters per hidden_size²):
- q_proj, k_proj: 0.75M each
- v_proj, g_proj, o_proj: 1.5M each

When `use_gate=False` (~6M parameters per hidden_size²):
- q_proj, k_proj: 1M each
- v_proj, o_proj: 2M each

## Usage Examples

### 1. Single Layer

```python
import torch
from fla.layers import GatedDeltaNet

layer = GatedDeltaNet(hidden_size=512, num_heads=8, head_dim=64)
x = torch.randn(2, 128, 512)  # (batch, seq_len, hidden)
output, _, _ = layer(x)
```

### 2. Full Model

```python
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

config = GatedDeltaNetConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_heads=12,
    head_dim=64,
    vocab_size=50257,  # GPT-2 vocab size
)

model = GatedDeltaNetForCausalLM(config)
```

### 3. With Caching (Inference)

```python
from fla.models.utils import Cache

# Initialize cache
past_key_values = Cache()

# First forward pass
input_ids = torch.randint(0, 50257, (1, 32))
outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)

# Next token prediction with cache
next_token_id = torch.randint(0, 50257, (1, 1))
outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
```

## Implementation Modes

### Chunk Mode (Training)
- Best for parallel training on long sequences
- Divides sequence into chunks
- Used automatically during `model.train()`

### Fused Recurrent Mode (Inference)
- Best for autoregressive generation
- Processes one token at a time efficiently
- Automatically used for sequences ≤64 tokens during inference

## Important Notes

⚠️ **Short Convolutions are Crucial**
Never set `use_short_conv=False` unless you know what you're doing. The paper shows this significantly impacts performance.

⚠️ **Head Dimension Constraint**
`num_heads * head_dim = 0.75 * hidden_size` when `use_gate=True`

⚠️ **Training vs Inference**
- Training: Always uses 'chunk' mode
- Inference: Auto-switches to 'fused_recurrent' for short sequences

## CUDA Kernels

The implementation includes custom CUDA kernels:

```python
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,           # For training
    fused_recurrent_gated_delta_rule, # For inference
)
```

These kernels implement:
- L2 normalization of q and k
- Gated delta rule updates
- Efficient state management

## Research Questions

Some interesting directions to explore:

1. **Scaling**: How does it scale to 1B, 7B, 13B parameters?
2. **Long Context**: Performance on 32k, 100k, 1M token sequences?
3. **Domain Adaptation**: Code, math, scientific text?
4. **Hybrid Models**: Combining with local attention or MoE?
5. **Training Efficiency**: Memory and speed vs Transformers?
6. **Ablations**: Impact of each component (gating, conv, beta, etc.)?

## Key Files

- **Layer**: `fla/layers/gated_deltanet.py`
- **Model**: `fla/models/gated_deltanet/modeling_gated_deltanet.py`
- **Config**: `fla/models/gated_deltanet/configuration_gated_deltanet.py`
- **Kernels**: `fla/ops/gated_delta_rule/`

## Paper Reference

```
Gated Delta Networks: Improving Mamba2 with Delta Rule
arXiv:2412.06464
https://arxiv.org/abs/2412.06464
```

## Common Configurations

### Tiny (for testing)
```python
config = GatedDeltaNetConfig(
    hidden_size=256,
    num_hidden_layers=6,
    num_heads=4,
    head_dim=64,
)
```

### Small (~124M params, GPT-2 small size)
```python
config = GatedDeltaNetConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_heads=12,
    head_dim=64,
)
```

### Medium (~350M params, GPT-2 medium size)
```python
config = GatedDeltaNetConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_heads=16,
    head_dim=64,
)
```

### Large (~774M params, GPT-2 large size)
```python
config = GatedDeltaNetConfig(
    hidden_size=1280,
    num_hidden_layers=36,
    num_heads=20,
    head_dim=64,
)
```
