#!/usr/bin/env python3
"""
Investigate and optimize Host-to-Device memory transfers in Evo2.

The nsys profile showed 99% of memory op time is HtoD transfers.
This script investigates why and tests optimizations.
"""

import torch
import time
import numpy as np
from pathlib import Path


def generate_random_dna(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def check_model_device_placement(model):
    """Check where model parameters actually live."""
    print("\n" + "="*60)
    print("Model Device Placement Analysis")
    print("="*60)

    device_counts = {}
    dtype_counts = {}
    total_params = 0
    total_bytes = 0

    for name, param in model.model.named_parameters():
        device = str(param.device)
        dtype = str(param.dtype)

        device_counts[device] = device_counts.get(device, 0) + 1
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    print(f"\nTotal parameters: {total_params:,} ({total_bytes / 1e9:.2f} GB)")

    print(f"\nParameters by device:")
    for device, count in sorted(device_counts.items()):
        print(f"  {device}: {count} tensors")

    print(f"\nParameters by dtype:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count} tensors")

    # Check for any CPU tensors
    cpu_params = [(name, p) for name, p in model.model.named_parameters()
                  if 'cpu' in str(p.device)]
    if cpu_params:
        print(f"\n WARNING: {len(cpu_params)} parameters still on CPU!")
        for name, p in cpu_params[:5]:
            print(f"  {name}: {p.shape} on {p.device}")

    return device_counts, dtype_counts


def check_buffer_placement(model):
    """Check model buffers (non-parameter tensors)."""
    print("\n" + "="*60)
    print("Model Buffer Analysis")
    print("="*60)

    buffer_info = []
    for name, buf in model.model.named_buffers():
        buffer_info.append({
            'name': name,
            'device': str(buf.device),
            'dtype': str(buf.dtype),
            'shape': tuple(buf.shape),
            'bytes': buf.numel() * buf.element_size()
        })

    print(f"Total buffers: {len(buffer_info)}")

    # Group by device
    by_device = {}
    for b in buffer_info:
        dev = b['device']
        if dev not in by_device:
            by_device[dev] = []
        by_device[dev].append(b)

    for device, bufs in by_device.items():
        total_bytes = sum(b['bytes'] for b in bufs)
        print(f"\n{device}: {len(bufs)} buffers, {total_bytes / 1e6:.2f} MB")
        for b in bufs[:3]:
            print(f"  {b['name']}: {b['shape']}")


def profile_single_inference(model, seq_len=10000, use_pinned=False, use_non_blocking=False):
    """Profile a single inference with different memory settings."""

    seq = generate_random_dna(seq_len)

    # Tokenize on CPU
    tokens = model.tokenizer.tokenize(seq)

    if use_pinned:
        # Use pinned memory
        input_ids = torch.tensor(tokens, dtype=torch.int).unsqueeze(0)
        input_ids = input_ids.pin_memory()
        input_ids = input_ids.cuda(non_blocking=use_non_blocking)
    else:
        # Standard approach
        input_ids = torch.tensor(tokens, dtype=torch.int).unsqueeze(0).cuda()

    torch.cuda.synchronize()

    # Time the inference
    start = time.perf_counter()
    with torch.no_grad():
        output = model.model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed


def test_memory_optimizations(model, seq_len=10000, num_runs=5):
    """Test different memory transfer optimizations."""
    print("\n" + "="*60)
    print("Memory Transfer Optimization Tests")
    print("="*60)

    configs = [
        ("Standard", False, False),
        ("Pinned Memory", True, False),
        ("Pinned + Non-blocking", True, True),
    ]

    results = {}

    for name, use_pinned, use_non_blocking in configs:
        print(f"\nTesting: {name}")

        # Warmup
        for _ in range(2):
            profile_single_inference(model, seq_len, use_pinned, use_non_blocking)

        # Timed runs
        times = []
        for i in range(num_runs):
            t = profile_single_inference(model, seq_len, use_pinned, use_non_blocking)
            times.append(t)
            print(f"  Run {i+1}: {t*1000:.2f} ms")

        results[name] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
        }

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    baseline = results["Standard"]['mean']
    for name, r in results.items():
        speedup = baseline / r['mean']
        print(f"{name:25s}: {r['mean']:.2f} ms (±{r['std']:.2f}) - {speedup:.2f}x")


def investigate_htod_source(model, seq_len=10000):
    """Try to identify the source of HtoD transfers."""
    print("\n" + "="*60)
    print("Investigating HtoD Transfer Sources")
    print("="*60)

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    mem_before = torch.cuda.memory_allocated()

    # Run inference with memory tracking
    with torch.no_grad():
        output = model.model(input_ids)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    print(f"Memory before: {mem_before / 1e9:.2f} GB")
    print(f"Memory after:  {mem_after / 1e9:.2f} GB")
    print(f"Memory peak:   {mem_peak / 1e9:.2f} GB")
    print(f"Delta:         {(mem_after - mem_before) / 1e9:.2f} GB")

    # Check if there are any hooks or custom forward passes
    print("\nChecking for forward hooks...")
    for name, module in model.model.named_modules():
        if hasattr(module, '_forward_hooks') and module._forward_hooks:
            print(f"  {name}: has forward hooks")
        if hasattr(module, '_forward_pre_hooks') and module._forward_pre_hooks:
            print(f"  {name}: has forward pre-hooks")


def check_cuda_streams():
    """Check current CUDA stream configuration."""
    print("\n" + "="*60)
    print("CUDA Stream Analysis")
    print("="*60)

    current_stream = torch.cuda.current_stream()
    default_stream = torch.cuda.default_stream()

    print(f"Current stream: {current_stream}")
    print(f"Default stream: {default_stream}")
    print(f"Are they the same? {current_stream == default_stream}")

    # Check number of available streams
    print(f"\nDevice count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")


def suggest_optimizations(model):
    """Based on analysis, suggest optimizations."""
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)

    print("""
Based on the profiling data showing 99% of memory time in HtoD transfers:

1. INVESTIGATE: Why 838 HtoD transfers?
   - Check if model uses lazy loading
   - Check for dtype conversions (bf16 -> fp32 -> bf16)
   - Check for non-contiguous tensors

2. QUICK WINS:
   - Use pinned memory for input data
   - Use non_blocking=True for .cuda() calls
   - Ensure model is fully on GPU after loading

3. MEDIUM EFFORT:
   - Pre-allocate output buffers
   - Use CUDA graphs for repeated inference
   - Profile with torch.cuda.memory._record_memory_history()

4. ADVANCED:
   - Custom CUDA kernel for tokenization (keep on GPU)
   - Memory-mapped model loading
   - Multi-stream pipeline for batch processing

5. WHAT TO CHECK IN EVO2 CODE:
   - Look for .cpu() calls in forward pass
   - Look for .to() or .float() calls
   - Check Hyena implementation for CPU fallbacks
""")


def main():
    print("="*60)
    print("Evo2 Memory Transfer Optimization Analysis")
    print("="*60)

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    # Analysis
    check_model_device_placement(model)
    check_buffer_placement(model)
    check_cuda_streams()
    investigate_htod_source(model)

    # Test optimizations
    test_memory_optimizations(model, seq_len=10000, num_runs=3)

    # Recommendations
    suggest_optimizations(model)


if __name__ == "__main__":
    main()
