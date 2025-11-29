# Gated DeltaNet Research

This repository contains a focused implementation of **Gated DeltaNet** for research purposes.

## Paper

**Gated Delta Networks: Improving Mamba2 with Delta Rule**  
📄 [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

## What is Gated DeltaNet?

Gated DeltaNet is a linear attention mechanism that combines:
- **Delta Rule**: An efficient update mechanism for recurrent states
- **Gating Mechanisms**: Similar to Mamba2's gating for improved expressiveness
- **Efficient Implementations**: Both chunk-based (training) and fused recurrent (inference) kernels

## Repository Structure

```
flash-linear-attention/
├── fla/
│   ├── layers/
│   │   └── gated_deltanet.py         # Core layer implementation
│   ├── models/
│   │   └── gated_deltanet/           # HuggingFace-compatible model
│   │       ├── configuration_gated_deltanet.py
│   │       └── modeling_gated_deltanet.py
│   ├── ops/
│   │   ├── gated_delta_rule/         # CUDA kernels
│   │   ├── common/                   # Common utilities
│   │   └── utils/                    # Helper functions
│   └── modules/                      # RMSNorm, ShortConv, etc.
├── gated_deltanet_research.ipynb     # Google Colab research notebook
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/sustcsonglin/flash-linear-attention.git
cd flash-linear-attention

# Install
pip install -e .

# Install dependencies
pip install transformers einops torch
```

## Quick Start

### Using the Layer

```python
import torch
from fla.layers import GatedDeltaNet

layer = GatedDeltaNet(
    hidden_size=512,
    expand_v=2.0,
    head_dim=64,
    num_heads=8,
    mode='chunk',
    use_gate=True,
    use_short_conv=True,
)

x = torch.randn(2, 128, 512)  # (batch, seq_len, hidden)
output, _, _ = layer(x)
```

### Using the Model

```python
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

config = GatedDeltaNetConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_heads=12,
    head_dim=64,
    vocab_size=50257,
)

model = GatedDeltaNetForCausalLM(config)
```

## Google Colab Research

Open `gated_deltanet_research.ipynb` in Google Colab to start experimenting:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/flash-linear-attention/blob/main/gated_deltanet_research.ipynb)

## Key Parameters

- **hidden_size**: Model hidden dimension
- **expand_v**: Value dimension expansion ratio (default: 2.0)
- **head_dim**: Dimension per attention head
- **num_heads**: Number of attention heads
- **mode**: 'chunk' (training) or 'fused_recurrent' (inference)
- **use_gate**: Enable output gating (recommended)
- **use_short_conv**: Enable short convolutions (crucial for performance)
- **allow_neg_eigval**: Allow negative eigenvalues for better state tracking

## Research Ideas

Some potential research directions:

1. **Scaling Laws**: How does Gated DeltaNet scale compared to Transformers and Mamba?
2. **Long Context**: Test performance on very long sequences (100k+ tokens)
3. **Hybrid Architectures**: Combine with local attention or other mechanisms
4. **Training Efficiency**: Compare training speed and memory vs Transformers
5. **Task Performance**: Evaluate on various downstream tasks
6. **Ablation Studies**: Impact of gating, convolutions, negative eigenvalues, etc.

## Citations

```bibtex
@article{yang2024gated,
  title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author={Yang, Songlin and others},
  journal={arXiv preprint arXiv:2412.06464},
  year={2024}
}
```

## License

See LICENSE file for details.
