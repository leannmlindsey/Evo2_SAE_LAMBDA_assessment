#!/usr/bin/env python3
"""
Evo2 Profiling Script - Day 1

This script profiles Evo2 inference to identify bottlenecks.
Run on H200 to get accurate GPU timings.

Outputs:
- Console summary of where time is spent
- profile_results.json with detailed timings
- Chrome trace file for visualization (profile_trace.json)
"""

import json
import time
import torch
import numpy as np
from pathlib import Path

# Profiling configuration
SEQUENCE_LENGTHS = [1000, 10000, 50000, 100000]  # Test different lengths
NUM_WARMUP = 2
NUM_RUNS = 5
OUTPUT_DIR = Path("./profiling_results")


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def generate_random_dna(length):
    """Generate random DNA sequence."""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def profile_basic_inference(model, sequence_lengths):
    """Profile basic inference at different sequence lengths."""
    results = []

    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Profiling sequence length: {seq_len:,}")
        print('='*60)

        # Generate test sequence
        seq = generate_random_dna(seq_len)
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).cuda()

        # Warmup
        print("Warming up...")
        for _ in range(NUM_WARMUP):
            with torch.no_grad():
                _ = model.model(input_ids)
            torch.cuda.synchronize()

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure memory before
        mem_before = get_gpu_memory()

        # Timed runs
        times = []
        print(f"Running {NUM_RUNS} timed iterations...")
        for i in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output = model.model(input_ids)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.2f} ms")

        # Memory stats
        mem_after = get_gpu_memory()
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result = {
            'seq_length': seq_len,
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'tokens_per_second': seq_len / np.mean(times),
            'memory_before_gb': mem_before,
            'memory_after_gb': mem_after,
            'peak_memory_gb': peak_mem,
        }
        results.append(result)

        print(f"\nResults for {seq_len:,} tokens:")
        print(f"  Mean time: {result['mean_time_ms']:.2f} ms")
        print(f"  Throughput: {result['tokens_per_second']:,.0f} tokens/sec")
        print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")

    return results


def profile_with_torch_profiler(model, seq_len=10000):
    """Detailed profiling with torch.profiler to identify bottlenecks."""
    print(f"\n{'='*60}")
    print(f"Detailed torch.profiler analysis (seq_len={seq_len:,})")
    print('='*60)

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
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            output = model.model(input_ids)
        torch.cuda.synchronize()

    # Print summary
    print("\nTop 20 CUDA operations by time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    # Save trace for Chrome visualization
    trace_path = OUTPUT_DIR / "profile_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nSaved Chrome trace to: {trace_path}")
    print("Open chrome://tracing and load this file to visualize")

    # Extract key metrics
    key_ops = []
    for item in prof.key_averages():
        if item.cuda_time_total > 0:
            key_ops.append({
                'name': item.key,
                'cuda_time_ms': item.cuda_time_total / 1000,
                'cpu_time_ms': item.cpu_time_total / 1000,
                'calls': item.count,
                'cuda_memory_mb': item.cuda_memory_usage / 1e6 if item.cuda_memory_usage else 0,
            })

    # Sort by CUDA time
    key_ops.sort(key=lambda x: x['cuda_time_ms'], reverse=True)

    return key_ops[:50]  # Top 50 operations


def profile_layer_by_layer(model, seq_len=10000):
    """Profile each layer/block to find slowest components."""
    print(f"\n{'='*60}")
    print(f"Layer-by-layer profiling (seq_len={seq_len:,})")
    print('='*60)

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    layer_times = {}

    def make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            layer_times[name] = time.perf_counter()
        return hook

    # Register hooks on major components
    hooks = []
    for name, module in model.model.named_modules():
        # Only hook top-level blocks to avoid too much overhead
        if name.count('.') <= 1 and name:  # Top-level or one level deep
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run inference
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.model(input_ids)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate layer durations
    sorted_layers = sorted(layer_times.items(), key=lambda x: x[1])
    layer_durations = []

    for i, (name, end_time) in enumerate(sorted_layers):
        if i == 0:
            duration = end_time - start_time
        else:
            duration = end_time - sorted_layers[i-1][1]

        layer_durations.append({
            'layer': name,
            'time_ms': duration * 1000,
            'percent': (duration / total_time) * 100
        })

    # Sort by time
    layer_durations.sort(key=lambda x: x['time_ms'], reverse=True)

    print("\nTop layers by time:")
    for item in layer_durations[:20]:
        print(f"  {item['layer']}: {item['time_ms']:.2f} ms ({item['percent']:.1f}%)")

    return layer_durations


def analyze_model_architecture(model):
    """Analyze model architecture and parameter distribution."""
    print(f"\n{'='*60}")
    print("Model Architecture Analysis")
    print('='*60)

    total_params = 0
    layer_params = {}

    for name, param in model.model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Group by top-level module
        top_level = name.split('.')[0]
        if top_level not in layer_params:
            layer_params[top_level] = 0
        layer_params[top_level] += num_params

    print(f"\nTotal parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"\nParameter distribution:")

    for name, count in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_params) * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Check dtype
    sample_param = next(model.model.parameters())
    print(f"\nModel dtype: {sample_param.dtype}")
    print(f"Model device: {sample_param.device}")

    return {
        'total_params': total_params,
        'layer_params': layer_params,
        'dtype': str(sample_param.dtype),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("Evo2 Profiling - Day 1")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    results = {}

    # 1. Architecture analysis
    results['architecture'] = analyze_model_architecture(model)

    # 2. Basic inference profiling at different sequence lengths
    results['basic_profiling'] = profile_basic_inference(model, SEQUENCE_LENGTHS)

    # 3. Detailed torch.profiler analysis
    results['detailed_ops'] = profile_with_torch_profiler(model, seq_len=10000)

    # 4. Layer-by-layer profiling
    results['layer_profiling'] = profile_layer_by_layer(model, seq_len=10000)

    # Save results
    results_path = OUTPUT_DIR / "profile_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nThroughput by sequence length:")
    for r in results['basic_profiling']:
        print(f"  {r['seq_length']:>7,} tokens: {r['tokens_per_second']:>10,.0f} tok/s, {r['mean_time_ms']:>8.1f} ms, {r['peak_memory_gb']:.1f} GB")

    print("\nTop 5 CUDA operations:")
    for op in results['detailed_ops'][:5]:
        print(f"  {op['name'][:50]:50s}: {op['cuda_time_ms']:>8.2f} ms")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Look at profile_trace.json in chrome://tracing")
    print("  2. Identify if bottleneck is compute or memory bound")
    print("  3. Run Day 2 INT8 quantization experiments")
    print("="*60)


if __name__ == "__main__":
    main()
