#!/usr/bin/env python3
"""
Minimal Evo2 inference for profiling with ncu.

Usage:
    # Standard run
    python test_inference.py

    # Profile depthwise conv
    ncu --set roofline --kernel-name "conv_depthwise2d" --launch-count 3 -o roofline_conv python test_inference.py

    # Profile elementwise kernels
    ncu --set roofline --kernel-name "elementwise" --launch-count 5 -o roofline_elem python test_inference.py

    # Profile GEMM
    ncu --set roofline --kernel-name "nvjet" --launch-count 3 -o roofline_gemm python test_inference.py
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
