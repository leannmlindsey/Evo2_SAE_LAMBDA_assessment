# Evo2 Profiling Experiment Summary

## Overview

This document summarizes the profiling and optimization analysis conducted on the Evo2 7B genomic language model running on an NVIDIA H200 GPU.

---

## 1. Model Architecture

### StripedHyena Configuration (from model loading logs)

| Parameter | Value |
|-----------|-------|
| Model | `shc-evo2-7b-8k-2T-v2` |
| Parameters | ~7B (6.58B across 32 blocks) |
| Hidden Size | 4096 |
| Num Layers | 32 |
| Vocab Size | 512 |
| Max Sequence Length | 1,048,576 (1M tokens) |

### Layer Distribution (32 total blocks)

| Layer Type | Indices | Count | Description |
|------------|---------|-------|-------------|
| Attention | [3, 10, 17, 24, 31] | 5 | Flash Attention with RoPE |
| Hyena Short (HCS) | [0, 4, 7, 11, 14, 18, 21, 25, 28] | 9 | Short filter (kernel=7) |
| Hyena Medium (HCM) | [1, 5, 8, 12, 15, 19, 22, 26, 29] | 9 | Medium filter (length=128) |
| Hyena Long (HCL) | [2, 6, 9, 13, 16, 20, 23, 27, 30] | 9 | Long FFT-based convolution |

### Key Architecture Features
- **Tokenization**: CharLevelTokenizer with ASCII encoding (A=65, T=84, C=67, G=71)
- **FP8 Quantization**: Already enabled via Transformer Engine (`use_fp8_input_projections: True`)
- **Flash Attention**: Enabled (`use_flash_attn: True`)
- **FlashFFTConv**: Currently disabled (`use_flashfft: False`)

---

## 2. Profiling Tools & Results

### Tools Used

| Tool | Purpose | Status |
|------|---------|--------|
| `torch.profiler` | High-level kernel timing | Working |
| `nsys` (Nsight Systems) | Timeline and memory analysis | Working |
| `ncu` (Nsight Compute) | Roofline/detailed kernel analysis | Blocked by permissions |

### nsys Profiling Results (5000 bp sequence)

**Top Kernels by Time:**

| Kernel | Time (ms) | % of Total | Role |
|--------|-----------|------------|------|
| `nvjet_tst_256x160` (GEMM) | 245.2 | 35.5% | Matrix multiplications |
| `nvjet_tst_128x304` (GEMM) | 127.2 | 18.4% | Matrix multiplications |
| `conv_depthwise2d_forward_kernel_generic` | 124.8 | 18.1% | Hyena short filter |
| `elementwise_kernel` | 93.4 | 13.5% | Gating, residuals |
| `vector_fft` | 45.8 | 6.6% | Hyena FFT convolution |
| `flash_fwd_kernel` | 36.1 | 5.2% | Attention |

**Memory Operations:**
- HtoD (Host to Device): 2.1 sec, 13.7 GB - **occurs during warmup/model loading only**
- DtoH (Device to Host): minimal during inference
- Inference after warmup: ~444ms (compute-bound)

### Key Finding: HtoD Transfers are NOT the Bottleneck

Initial analysis suggested 84% of CUDA time was memory copies. However:
- Memory optimization tests (pinned memory, non-blocking transfers) showed **zero speedup**
- The 13.7 GB HtoD transfer happens during **model loading**, not inference
- After warmup, inference is **compute-bound**, not memory-bound

---

## 3. Kernel Analysis for Fusion Opportunities

### Available Kernels (from ncu)

```
1. CatArrayBatchedCopy
2. _fft_conjugate_copy_kernel
3. conv_depthwise2d_forward_kernel_generic
4. distribution_elementwise_grid_stride_kernel
5. elementwise_kernel                          <- FUSION TARGET
6. elementwise_kernel_with_index
7. flash_fwd_kernel
8. indexSelectLargeIndex
9-14. nvjet_* (various GEMM configurations)
15. postprocess_kernel
16. preprocess_kernel
17. reduce_kernel
18. rotary_kernel
19. unary_kernel
20. unrolled_elementwise_kernel
21. vector_fft
22. vectorized_elementwise_kernel
```

### Data Flow Between Layers (Potential Fusion Points)

```
Attention Block Output (u)
    | [Write to HBM]           <- Interface write
    v
Hyena Block Input
    | Normalization + Projection
    | [Write to HBM]           <- Fusion candidate #1
    v
Short Filter (FIR)
    | [Write to HBM]           <- Fusion candidate #2
    v
Long Filter (FFT -> Multiply -> IFFT)
    | [Write to HBM x3]        <- Fusion candidate #3 (FlashFFTConv)
    v
Residual + MLP
    | [Write to HBM]           <- Fusion candidate #4
    v
Hyena Block Output
    | [Write to HBM]           <- Interface write
    v
Next Layer...
```

### Fusion Opportunities Identified

| Priority | Fusion | Current | Potential Gain |
|----------|--------|---------|----------------|
| 1 | FlashFFTConv (FFT->Mul->IFFT) | Disabled | 30-50% on Hyena FFT ops |
| 2 | Elementwise ops at layer interfaces | Separate kernels | Reduce HBM round-trips |
| 3 | Pre-norm + Projection | Separate | Eliminate intermediate write |
| 4 | Residual + Post-norm + MLP | Separate | Eliminate 2+ intermediate writes |

---

## 4. Hypothesis: Attention/Hyena Interface Optimization

The working hypothesis is that the **interface between attention and Hyena layers** presents optimization opportunities:

1. Each layer writes its output to HBM
2. The next layer reads from HBM
3. If operations at the interface (residuals, normalizations) could be fused with the preceding compute kernel, HBM writes could be eliminated

**To validate this hypothesis**, we need:
- Roofline analysis showing `elementwise_kernel` is memory-bound
- Memory bandwidth measurements at layer interfaces
- This requires `ncu` with proper permissions

---

## 5. Blocking Issue: ncu Permissions

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

**Solutions (requires sudo):**

```bash
# Option 1: Temporary fix
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# Option 2: Permanent fix
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiler.conf'
```

---

## 6. Scripts Created

| Script | Purpose |
|--------|---------|
| `test_inference.py` | Minimal inference for ncu profiling |
| `roofline_analysis.py` | Theoretical roofline plot for H200 |
| `cuda_graphs_optimization.py` | CUDA graphs compatibility testing |
| `optimize_memory_transfer.py` | Memory transfer optimization tests |
| `find_nvidia_tools.sh` | Locate nsys/ncu on system |

---

## 7. Recommendations for LBNL Interview

### Talking Points

1. **Architecture Understanding**
   - StripedHyena hybrid: 5 attention + 27 Hyena layers
   - Three Hyena types for different context lengths
   - Already using FP8 via Transformer Engine

2. **Profiling Methodology**
   - Layered approach: torch.profiler -> nsys -> ncu
   - Importance of distinguishing warmup from inference
   - Initial memory bottleneck hypothesis was disproven through testing

3. **Optimization Opportunities**
   - Enable FlashFFTConv (currently disabled in config)
   - Kernel fusion at layer interfaces
   - Custom Triton kernels for elementwise ops

4. **What Was Learned**
   - Evo2 inference is compute-bound after warmup
   - GEMM and depthwise conv dominate runtime
   - Memory transfers (HtoD) are model loading, not inference

### Next Steps (if ncu permissions resolved)

1. Run roofline analysis on `elementwise_kernel`
2. Measure arithmetic intensity at layer interfaces
3. Quantify potential gains from fusion
4. Profile with FlashFFTConv enabled

---

## 8. H200 Specifications (Reference)

| Metric | Value |
|--------|-------|
| Memory Bandwidth | 4.8 TB/s (HBM3e) |
| FP16/BF16 Tensor Core | 1,979 TFLOPS |
| FP32 | 67 TFLOPS |
| FP8 | 3,958 TFLOPS |
| HBM Capacity | 80 GB |

---

## Appendix: Key Config Flags to Investigate

From the model config, these flags could enable additional optimizations:

```python
'use_flash_rmsnorm': False,      # Could fuse RMSNorm
'use_flash_depthwise': False,    # Could optimize depthwise conv
'use_flashfft': False,           # FlashFFTConv - should enable!
'use_laughing_hyena': False,     # Alternative Hyena implementation
```

Enabling `use_flashfft: True` would be the lowest-hanging fruit for optimization.
