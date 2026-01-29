#!/usr/bin/env python3
"""
Check if Evo2/StripedHyena already uses fused Triton kernels.

Before claiming "you could fuse X" in an interview, verify it's not already done!
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict


def generate_random_dna(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def check_triton_usage():
    """Check if Triton is being used and how."""
    print("="*60)
    print("Checking Triton Usage in Evo2")
    print("="*60)

    # Check if triton is installed
    try:
        import triton
        print(f"\nTriton version: {triton.__version__}")
    except ImportError:
        print("\nTriton not installed")
        return

    # Check vortex/evo2 for triton imports
    print("\nSearching for Triton usage in evo2/vortex...")

    import evo2
    import vortex
    import inspect
    import os

    evo2_path = os.path.dirname(evo2.__file__)
    vortex_path = os.path.dirname(vortex.__file__)

    print(f"  evo2 path: {evo2_path}")
    print(f"  vortex path: {vortex_path}")

    # Search for triton imports in source files
    triton_files = []

    for base_path in [evo2_path, vortex_path]:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if 'triton' in content.lower() or '@triton.jit' in content:
                                triton_files.append(filepath)
                                # Find specific triton decorators
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if '@triton.jit' in line or 'triton.jit' in line:
                                        print(f"\n  Found @triton.jit in {filepath}:")
                                        # Print context
                                        start = max(0, i-1)
                                        end = min(len(lines), i+10)
                                        for j in range(start, end):
                                            print(f"    {j+1}: {lines[j][:80]}")
                    except:
                        pass

    if triton_files:
        print(f"\n  Files with Triton references: {len(triton_files)}")
        for f in triton_files[:10]:
            print(f"    - {f}")
    else:
        print("\n  No Triton kernels found in source")


def profile_kernel_names(model, seq_len=10000):
    """Profile and look for Triton kernel signatures."""
    print("\n" + "="*60)
    print("Profiling Kernel Names")
    print("="*60)

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model.model(input_ids)
        torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            output = model.model(input_ids)
        torch.cuda.synchronize()

    # Categorize kernels
    triton_kernels = []
    flash_attn_kernels = []
    cudnn_kernels = []
    cublas_kernels = []
    fft_kernels = []
    elementwise_kernels = []
    other_kernels = []

    for event in prof.key_averages():
        if event.cuda_time_total <= 0:
            continue

        name = event.key.lower()
        info = {
            'name': event.key,
            'cuda_time_ms': event.cuda_time_total / 1000,
            'count': event.count,
        }

        if 'triton' in name:
            triton_kernels.append(info)
        elif 'flash' in name or 'fmha' in name:
            flash_attn_kernels.append(info)
        elif 'cudnn' in name:
            cudnn_kernels.append(info)
        elif 'cublas' in name or 'gemm' in name or 'cutlass' in name:
            cublas_kernels.append(info)
        elif 'fft' in name or 'cufft' in name:
            fft_kernels.append(info)
        elif any(op in name for op in ['elementwise', 'vectorized', 'mul_', 'add_', 'sigmoid', 'silu', 'gelu', 'relu']):
            elementwise_kernels.append(info)
        else:
            other_kernels.append(info)

    # Report
    def print_category(name, kernels, show_all=False):
        if not kernels:
            print(f"\n{name}: None found")
            return

        total_time = sum(k['cuda_time_ms'] for k in kernels)
        total_calls = sum(k['count'] for k in kernels)
        print(f"\n{name}:")
        print(f"  Total: {total_time:.2f}ms across {total_calls} calls ({len(kernels)} unique kernels)")

        # Sort by time
        kernels_sorted = sorted(kernels, key=lambda x: x['cuda_time_ms'], reverse=True)

        limit = len(kernels_sorted) if show_all else min(5, len(kernels_sorted))
        for k in kernels_sorted[:limit]:
            print(f"    {k['name'][:60]}: {k['cuda_time_ms']:.2f}ms ({k['count']} calls)")

    print_category("TRITON KERNELS (custom fused)", triton_kernels, show_all=True)
    print_category("FLASH ATTENTION KERNELS", flash_attn_kernels, show_all=True)
    print_category("cuDNN KERNELS", cudnn_kernels)
    print_category("cuBLAS/GEMM KERNELS", cublas_kernels)
    print_category("FFT KERNELS", fft_kernels, show_all=True)
    print_category("ELEMENTWISE KERNELS (fusion candidates?)", elementwise_kernels, show_all=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: What's Already Optimized?")
    print("="*60)

    total_time = sum(k['cuda_time_ms'] for cat in [triton_kernels, flash_attn_kernels, cudnn_kernels,
                                                     cublas_kernels, fft_kernels, elementwise_kernels, other_kernels]
                     for k in cat)

    triton_time = sum(k['cuda_time_ms'] for k in triton_kernels)
    flash_time = sum(k['cuda_time_ms'] for k in flash_attn_kernels)
    ewise_time = sum(k['cuda_time_ms'] for k in elementwise_kernels)
    fft_time = sum(k['cuda_time_ms'] for k in fft_kernels)

    print(f"""
Total inference time: {total_time:.2f}ms

Already optimized:
  - Triton kernels:     {triton_time:>8.2f}ms ({100*triton_time/total_time:.1f}%) {"<-- ALREADY FUSED" if triton_time > 0 else ""}
  - Flash Attention:    {flash_time:>8.2f}ms ({100*flash_time/total_time:.1f}%) {"<-- ALREADY FUSED" if flash_time > 0 else ""}

Potential optimization targets:
  - Elementwise ops:    {ewise_time:>8.2f}ms ({100*ewise_time/total_time:.1f}%) {"<-- Check if fuseable" if ewise_time > total_time*0.05 else "<-- Negligible"}
  - FFT operations:     {fft_time:>8.2f}ms ({100*fft_time/total_time:.1f}%) {"<-- FlashFFTConv candidate" if fft_time > total_time*0.05 else "<-- Negligible"}
""")

    if triton_time > total_time * 0.1:
        print("NOTE: Significant Triton usage detected - they ARE using custom fused kernels!")
        print("      Need to find what's NOT yet fused.")

    if ewise_time < total_time * 0.05:
        print("NOTE: Elementwise ops are <5% of time - fusion here won't help much.")

    return {
        'triton_kernels': triton_kernels,
        'flash_attn_kernels': flash_attn_kernels,
        'elementwise_kernels': elementwise_kernels,
        'fft_kernels': fft_kernels,
        'total_time_ms': total_time,
    }


def check_flash_fft_conv_usage(model):
    """Check if FlashFFTConv is already being used."""
    print("\n" + "="*60)
    print("Checking for FlashFFTConv Usage")
    print("="*60)

    # Check imports
    try:
        import vortex
        import inspect
        source = inspect.getsourcefile(vortex)
        print(f"  vortex source: {source}")

        # Look for flashfftconv in vortex
        import os
        vortex_path = os.path.dirname(vortex.__file__)

        found_flash_fft = False
        for root, dirs, files in os.walk(vortex_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if 'flashfftconv' in content.lower() or 'flash_fft' in content.lower():
                                print(f"\n  Found FlashFFTConv reference in: {filepath}")
                                found_flash_fft = True
                    except:
                        pass

        if not found_flash_fft:
            print("\n  No FlashFFTConv usage found - this is an optimization opportunity!")

    except Exception as e:
        print(f"  Error checking: {e}")


def main():
    print("="*60)
    print("Checking Existing Kernel Fusion in Evo2")
    print("="*60)
    print("\nThis checks what's ALREADY optimized before suggesting improvements.\n")

    # Check Triton usage in source
    check_triton_usage()

    # Load model
    print("\n\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    # Check FlashFFTConv
    check_flash_fft_conv_usage(model)

    # Profile actual kernel usage
    results = profile_kernel_names(model, seq_len=10000)

    # Final recommendations
    print("\n" + "="*60)
    print("INTERVIEW PREPARATION")
    print("="*60)

    if results['triton_kernels']:
        print("""
They ARE using Triton. Your talking points should be:

1. "I see you're already using Triton for X - what specific operations
    did you find most beneficial to fuse?"

2. "Have you considered FlashFFTConv for the Hyena convolutions?
    I noticed the FFT operations take X% of inference time."

3. "What's the bottleneck now - is it memory bandwidth or compute?"
""")
    else:
        print("""
They're NOT using Triton extensively. Your talking points:

1. "I profiled Evo2 and found X% of time in unfused elementwise ops.
    Triton could reduce this significantly."

2. "The gating operations in Hyena are a prime fusion target -
    3 separate kernels could become 1."

3. "FlashFFTConv could optimize the long convolutions."
""")


if __name__ == "__main__":
    main()
