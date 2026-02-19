#!/usr/bin/env python3
"""
Minimal Evo2 inference for profiling with ncu.

Usage:
    # Standard run
    python test_inference.py

    # Profile elementwise kernels (LAYER INTERFACE - best fusion candidate)
    ncu --set roofline --kernel-name regex:"elementwise_kernel$" --launch-count 5 -o roofline_elem python test_inference.py

    # Profile depthwise conv (Hyena short filter)
    ncu --set roofline --kernel-name "conv_depthwise2d_forward_kernel_generic" --launch-count 3 -o roofline_conv python test_inference.py

    # Profile FFT (Hyena long convolution)
    ncu --set roofline --kernel-name "vector_fft" --launch-count 3 -o roofline_fft python test_inference.py

    # Profile Flash Attention
    ncu --set roofline --kernel-name "flash_fwd_kernel" --launch-count 3 -o roofline_attn python test_inference.py

    # Profile GEMM
    ncu --set roofline --kernel-name regex:"nvjet_tst" --launch-count 3 -o roofline_gemm python test_inference.py

Available kernels from Evo2:
    elementwise_kernel                      - Gating/residuals (FUSION TARGET)
    conv_depthwise2d_forward_kernel_generic - Hyena short filter
    vector_fft                              - Hyena FFT convolution
    flash_fwd_kernel                        - Attention
    nvjet_tst_*                             - GEMM operations
    rotary_kernel                           - RoPE embeddings
"""

import torch
import numpy as np

# Configuration
SEQ_LEN = 5000
NUM_WARMUP = 2


def generate_random_dna(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def main():
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Warmup iterations: {NUM_WARMUP}")

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    # Generate sequence
    seq = generate_random_dna(SEQ_LEN)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    print(f"Input shape: {input_ids.shape}")

    # Warmup
    print(f"\nWarmup ({NUM_WARMUP} iterations)...")
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            _ = model.model(input_ids)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # Profiled inference
    print("\nRunning inference (this is what gets profiled)...")
    torch.cuda.synchronize()

    with torch.no_grad():
        output = model.model(input_ids)

    torch.cuda.synchronize()
    print("Inference complete.")

    # Output info
    if isinstance(output, tuple):
        print(f"Output shape: {output[0].shape}")
    else:
        print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
