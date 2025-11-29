# Getting Started with Gated DeltaNet Research

Welcome! This repository has been cleaned up to focus exclusively on **Gated DeltaNet** research.

## 📚 Key Documents

1. **README.md** - Overview and installation instructions
2. **QUICK_REFERENCE.md** - Comprehensive API reference and examples
3. **CLEANUP_SUMMARY.md** - What was kept/removed from the original repo
4. **gated_deltanet_research.ipynb** - Google Colab research notebook

## 🚀 Quick Start (Google Colab)

### Option 1: Direct Upload
1. Upload `gated_deltanet_research.ipynb` to Google Colab
2. Run the setup cells to install dependencies
3. Start experimenting!

### Option 2: From GitHub
```python
# In a Colab cell:
!git clone <your-repo-url>
%cd flash-linear-attention
!pip install -e .
```

## 📋 Repository Structure

```
flash-linear-attention/
├── README.md                          # Main documentation
├── QUICK_REFERENCE.md                 # API reference
├── CLEANUP_SUMMARY.md                 # Cleanup details
├── gated_deltanet_research.ipynb      # Research notebook
│
├── fla/                               # Main package
│   ├── __init__.py                    # Package exports
│   │
│   ├── layers/                        # Core implementations
│   │   ├── gated_deltanet.py         # ⭐ Main layer
│   │   └── utils.py
│   │
│   ├── models/                        # HuggingFace models
│   │   ├── gated_deltanet/
│   │   │   ├── configuration_gated_deltanet.py
│   │   │   └── modeling_gated_deltanet.py
│   │   ├── modeling_layers.py
│   │   └── utils.py
│   │
│   ├── ops/                           # CUDA kernels
│   │   ├── gated_delta_rule/         # ⭐ Core operations
│   │   ├── common/                    # Shared utilities
│   │   └── utils/
│   │
│   └── modules/                       # Building blocks
│       ├── fused_norm_gate.py        # RMSNorm + gating
│       ├── convolution.py            # Short convolutions
│       └── ...
│
├── setup.py                           # Installation
├── pyproject.toml                     # Project config
└── tests/                             # Tests
```

## 💡 First Steps

### 1. Understand the Layer
Read `fla/layers/gated_deltanet.py` to understand:
- How the gated delta rule works
- The role of beta, gamma parameters
- Short convolution integration
- Chunk vs fused recurrent modes

### 2. Explore the Operations
Check `fla/ops/gated_delta_rule/` for:
- CUDA kernel implementations
- Performance optimizations
- L2 normalization details

### 3. Study the Model
Look at `fla/models/gated_deltanet/modeling_gated_deltanet.py`:
- How layers are stacked
- Position embeddings
- Caching mechanism for generation

## 🔬 Research Ideas

### Beginner-Friendly
1. **Toy Tasks**: Train on simple sequence tasks (copy, reverse, etc.)
2. **Scaling Analysis**: How do parameters affect performance?
3. **Visualization**: Plot attention patterns, state dynamics

### Intermediate
1. **Benchmark**: Compare to Transformer on standard tasks
2. **Long Context**: Test on 8k, 16k, 32k token sequences
3. **Ablation Studies**: Remove components one by one

### Advanced
1. **Novel Architectures**: Hybrid Gated DeltaNet + other mechanisms
2. **Training at Scale**: 1B+ parameter models
3. **Theoretical Analysis**: Expressiveness, approximation bounds

## 📖 Paper to Read

**Gated Delta Networks: Improving Mamba2 with Delta Rule**
- arXiv: https://arxiv.org/abs/2412.06464
- Focus on sections 2-3 for the algorithm
- Section 4 for experimental setup ideas

## 🛠️ Development Tips

### Testing Changes
```bash
# Run module tests
python -m pytest tests/modules/

# Quick sanity check
python -c "from fla.layers import GatedDeltaNet; print('OK')"
```

### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tensor shapes
layer = GatedDeltaNet(hidden_size=256, num_heads=4, head_dim=64)
print(layer)  # Shows architecture
```

### Performance Profiling
```python
import torch
from torch.profiler import profile, ProfilerActivity

layer = GatedDeltaNet(hidden_size=512, num_heads=8, head_dim=64)
x = torch.randn(2, 1024, 512, device='cuda')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output, _, _ = layer(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🤝 Contributing Back

If you make improvements or discoveries:
1. Document your findings
2. Add tests for new features
3. Update the relevant markdown files
4. Consider contributing back to the original repo

## 📬 Getting Help

- Read the QUICK_REFERENCE.md for API details
- Check CLEANUP_SUMMARY.md to understand what's available
- Review the Jupyter notebook for usage examples
- Refer to the paper for algorithmic details

## ✅ Checklist for Starting Research

- [ ] Read the Gated DeltaNet paper
- [ ] Understand the QUICK_REFERENCE.md
- [ ] Run gated_deltanet_research.ipynb successfully
- [ ] Verify installation: `pip install -e .`
- [ ] Test basic functionality with a toy example
- [ ] Decide on your research question
- [ ] Set up experiment tracking (wandb, tensorboard, etc.)
- [ ] Start small, then scale up!

Good luck with your research! 🚀
