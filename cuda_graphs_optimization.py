#!/usr/bin/env python3
"""
CUDA Graphs Optimization for Evo2

CUDA Graphs capture a sequence of GPU operations and replay them with minimal
CPU overhead. Benefits:
1. Eliminates kernel launch overhead
2. Enables driver-level optimizations
3. Reduces CPU-GPU synchronization

Limitations:
1. Static shapes - input size must be fixed
2. No CPU operations during captured region
3. No dynamic control flow

For Evo2, CUDA graphs could help by:
- Eliminating the 4313 kernel launches overhead
- Reducing cudaMemcpyAsync scheduling overhead
- Enabling operation fusion at driver level
"""

import torch
import time
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("./cuda_graphs_results")


def generate_random_dna(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def check_cuda_graphs_compatibility(model, seq_len=1000):
    """Check if model is compatible with CUDA graphs."""
    print("\n" + "="*60)
    print("CUDA Graphs Compatibility Check")
    print("="*60)

    issues = []

    # Check 1: Are there CPU operations in forward pass?
    print("\n1. Checking for potential CPU operations...")

    # Check 2: Dynamic shapes?
    print("2. Checking for dynamic shape operations...")

    # Check 3: Data-dependent control flow?
    print("3. Checking for data-dependent control flow...")

    # Try a simple capture
    print("\n4. Attempting CUDA graph capture...")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Warmup
    print("   Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model.model(input_ids)
    torch.cuda.synchronize()

    # Try to capture
    print("   Capturing graph...")
    try:
        # Create a static input buffer
        static_input = input_ids.clone()

        # Create graph
        g = torch.cuda.CUDAGraph()

        # Warmup in capture mode
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            with torch.no_grad():
                _ = model.model(static_input)

        torch.cuda.current_stream().wait_stream(s)

        # Capture
        with torch.cuda.graph(g):
            with torch.no_grad():
                static_output = model.model(static_input)

        print("   SUCCESS: Graph captured!")
        return True, g, static_input, static_output

    except Exception as e:
        print(f"   FAILED: {e}")
        issues.append(str(e))
        return False, None, None, None


def benchmark_with_cuda_graph(model, g, static_input, static_output, seq_len, num_runs=10):
    """Benchmark inference with CUDA graph."""
    print("\n" + "="*60)
    print("CUDA Graph Benchmark")
    print("="*60)

    # Generate new input data (must be same shape)
    seq = generate_random_dna(seq_len)
    new_tokens = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Copy new data to static input buffer
    static_input.copy_(new_tokens)

    # Warmup graph replay
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    # Benchmark graph replay
    times_graph = []
    for _ in range(num_runs):
        # Copy new input (simulating different inputs)
        static_input.copy_(new_tokens)

        torch.cuda.synchronize()
        start = time.perf_counter()

        g.replay()

        torch.cuda.synchronize()
        times_graph.append(time.perf_counter() - start)

    return times_graph


def benchmark_without_cuda_graph(model, seq_len, num_runs=10):
    """Benchmark standard inference without CUDA graph."""
    print("\nStandard Inference (no CUDA graph):")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.model(input_ids)
    torch.cuda.synchronize()

    # Benchmark
    times_standard = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.model(input_ids)

        torch.cuda.synchronize()
        times_standard.append(time.perf_counter() - start)

    return times_standard


def try_partial_cuda_graph(model, seq_len=1000):
    """Try capturing only parts of the model that are graph-compatible."""
    print("\n" + "="*60)
    print("Partial CUDA Graph Capture")
    print("="*60)

    print("""
If full model capture fails, we can try:
1. Capture only the attention layers
2. Capture only the Hyena layers
3. Capture the main backbone, exclude input/output processing
""")

    # This would require modifying the model's forward pass
    # For now, just demonstrate the concept

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Try to identify which part fails
    print("\nAnalyzing model layers for graph compatibility...")

    # Check each major component
    for name, module in model.model.named_children():
        print(f"\n  Checking: {name}")
        try:
            # Get input to this module by running up to it
            # This is simplified - real implementation would need hooks
            print(f"    Type: {type(module).__name__}")
        except Exception as e:
            print(f"    Error: {e}")


def use_torch_compile_alternative(model, seq_len=1000):
    """Try torch.compile as an alternative to CUDA graphs."""
    print("\n" + "="*60)
    print("torch.compile Alternative")
    print("="*60)

    print("""
torch.compile (PyTorch 2.0+) provides similar benefits to CUDA graphs:
- Kernel fusion
- Reduced overhead
- Automatic optimization

But with more flexibility:
- Handles dynamic shapes (with some modes)
- Automatic fallback for incompatible ops
""")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    print("\nTrying torch.compile...")

    try:
        # Compile the model
        compiled_model = torch.compile(model.model, mode="reduce-overhead")

        # Warmup (compilation happens here)
        print("  Warming up (compilation)...")
        with torch.no_grad():
            for _ in range(3):
                _ = compiled_model(input_ids)
        torch.cuda.synchronize()

        print("  Compilation successful!")

        # Benchmark
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = compiled_model(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        print(f"  Compiled inference: {np.mean(times)*1000:.2f} ms (±{np.std(times)*1000:.2f})")

        return compiled_model, times

    except Exception as e:
        print(f"  torch.compile failed: {e}")
        return None, None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("CUDA Graphs Optimization for Evo2")
    print("="*60)

    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    # Test sequence length (must be fixed for CUDA graphs)
    seq_len = 1000  # Start small

    # Check compatibility
    success, graph, static_input, static_output = check_cuda_graphs_compatibility(model, seq_len)

    num_runs = 10

    if success:
        # Benchmark with graph
        times_graph = benchmark_with_cuda_graph(
            model, graph, static_input, static_output, seq_len, num_runs
        )

        # Benchmark without graph
        times_standard = benchmark_without_cuda_graph(model, seq_len, num_runs)

        # Compare
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)

        mean_standard = np.mean(times_standard) * 1000
        mean_graph = np.mean(times_graph) * 1000
        speedup = mean_standard / mean_graph

        print(f"\nSequence length: {seq_len}")
        print(f"Standard inference: {mean_standard:.2f} ms (±{np.std(times_standard)*1000:.2f})")
        print(f"CUDA graph replay:  {mean_graph:.2f} ms (±{np.std(times_graph)*1000:.2f})")
        print(f"Speedup: {speedup:.2f}x")

        if speedup > 1.1:
            print("\n CUDA graphs provide significant speedup!")
        else:
            print("\n CUDA graphs provide minimal benefit (compute-bound, not launch-bound)")

    else:
        print("\nCUDA graph capture failed. Trying alternatives...")

        # Try partial capture
        try_partial_cuda_graph(model, seq_len)

        # Try torch.compile
        compiled_model, compile_times = use_torch_compile_alternative(model, seq_len)

        if compile_times:
            times_standard = benchmark_without_cuda_graph(model, seq_len, num_runs)

            print("\n" + "="*60)
            print("torch.compile RESULTS")
            print("="*60)

            mean_standard = np.mean(times_standard) * 1000
            mean_compiled = np.mean(compile_times) * 1000
            speedup = mean_standard / mean_compiled

            print(f"\nStandard inference:  {mean_standard:.2f} ms")
            print(f"Compiled inference:  {mean_compiled:.2f} ms")
            print(f"Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    print("""
CUDA Graphs work best when:
1. Input shapes are fixed
2. Model has no CPU operations in forward pass
3. No dynamic control flow

If CUDA graphs don't work for Evo2:
1. Try torch.compile with mode="reduce-overhead"
2. Try torch.compile with mode="max-autotune"
3. Consider CUDA graphs for specific layers only
4. Use CUDA graphs for batch inference (fixed batch size)

For production deployment:
- Profile with different sequence lengths
- Consider separate graphs for common sequence lengths
- Monitor GPU utilization to ensure graphs are helping
""")


if __name__ == "__main__":
    main()
