#!/usr/bin/env python3
"""
Profiling Comparison: torch.profiler vs Nsight Systems vs Nsight Compute

This script demonstrates the different levels of profiling and what each reveals.

Usage:
    # 1. Python profiler (run directly)
    python profile_comparison.py --method torch

    # 2. Nsight Systems (wrap with nsys)
    nsys profile -o evo2_nsys python profile_comparison.py --method nsys
    nsys stats evo2_nsys.nsys-rep

    # 3. Nsight Compute (wrap with ncu, profiles specific kernels)
    ncu --set full -o evo2_ncu python profile_comparison.py --method ncu
    ncu -i evo2_ncu.ncu-rep

Comparison of what each tool reveals:
─────────────────────────────────────────────────────────────────────────────
torch.profiler:
  - Python-level view
  - Op names (aten::mm, aten::add, etc.)
  - Total time per op type
  - Easy call stacks
  - Good for: "Which operations are slow?"

Nsight Systems (nsys):
  - System-level timeline
  - CPU-GPU synchronization
  - Kernel launch gaps
  - Memory transfers (HtoD, DtoH)
  - API call overhead
  - Good for: "Why is GPU utilization low?"

Nsight Compute (ncu):
  - Single kernel deep-dive
  - Occupancy analysis
  - Memory bandwidth achieved vs theoretical
  - Compute throughput
  - Roofline model position
  - Good for: "Why is this specific kernel slow?"
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("./profile_comparison_results")


def generate_random_dna(length):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def run_inference(model, seq_len=10000, num_runs=3):
    """Run inference multiple times for profiling."""
    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Warmup
    print("Warming up...")
    for _ in range(2):
        with torch.no_grad():
            _ = model.model(input_ids)
        torch.cuda.synchronize()

    # Timed runs
    print(f"Running {num_runs} inference passes...")
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.model(input_ids)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")

    return times, input_ids


def profile_with_torch(model, seq_len=10000):
    """Profile using torch.profiler."""
    print("\n" + "="*60)
    print("METHOD 1: torch.profiler")
    print("="*60)
    print("""
What it shows:
  - Operation names (aten::mm, aten::linear, etc.)
  - Time per operation type
  - CUDA kernel time vs CPU time
  - Memory allocation

What it misses:
  - Detailed kernel metrics (occupancy, bandwidth)
  - CPU-GPU synchronization gaps
  - Timeline visualization
""")

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

    # Print results
    print("\nTop 15 CUDA operations by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Save trace
    trace_path = OUTPUT_DIR / "torch_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nSaved Chrome trace to: {trace_path}")
    print("View in: chrome://tracing")

    # Memory summary
    print("\nMemory Summary:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


def profile_with_nsys(model, seq_len=10000):
    """Setup for Nsight Systems profiling."""
    print("\n" + "="*60)
    print("METHOD 2: Nsight Systems (nsys)")
    print("="*60)
    print("""
What it shows:
  - Full system timeline
  - CPU activity vs GPU activity
  - Kernel launch latency
  - Memory transfer (HtoD, DtoH)
  - CUDA API call duration
  - Gaps where GPU is idle

What it misses:
  - Inside-kernel details
  - Why a specific kernel is slow

How to run:
  nsys profile -o evo2_profile python profile_comparison.py --method nsys

How to analyze:
  nsys stats evo2_profile.nsys-rep        # Command line summary
  nsys-ui evo2_profile.nsys-rep           # GUI timeline
""")

    # Add NVTX markers for better visualization
    try:
        import nvtx
        has_nvtx = True
        print("\nNVTX available - adding range markers...")
    except ImportError:
        has_nvtx = False
        print("\nNVTX not available (pip install nvtx for better markers)")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Run with NVTX markers if available
    if has_nvtx:
        with nvtx.annotate("warmup", color="red"):
            for _ in range(2):
                with torch.no_grad():
                    _ = model.model(input_ids)
                torch.cuda.synchronize()

        with nvtx.annotate("inference", color="green"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model.model(input_ids)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

        print(f"\nInference time: {elapsed*1000:.2f} ms")
    else:
        # Run without markers
        run_inference(model, seq_len, num_runs=3)

    print("""
After running with nsys, look for:
  1. GPU idle time (gaps in kernel execution)
  2. Long CUDA API calls
  3. Unexpected memory transfers
  4. Kernel launch overhead (many small kernels = bad)
""")


def profile_with_ncu(model, seq_len=5000):  # Shorter sequence for ncu
    """Setup for Nsight Compute profiling."""
    print("\n" + "="*60)
    print("METHOD 3: Nsight Compute (ncu)")
    print("="*60)
    print("""
What it shows:
  - Per-kernel detailed metrics
  - Achieved occupancy vs theoretical
  - Memory throughput (GB/s)
  - Compute throughput (FLOPS)
  - Roofline model position
  - Warp stall reasons
  - Register usage

What it misses:
  - System-level view
  - Overall application performance
  - CPU-side issues

How to run (profiles ALL kernels - slow!):
  ncu --set full -o evo2_ncu python profile_comparison.py --method ncu

How to run (specific kernel pattern):
  ncu --set full --kernel-name ".*gemm.*" -o evo2_gemm python profile_comparison.py --method ncu
  ncu --set full --kernel-name ".*fft.*" -o evo2_fft python profile_comparison.py --method ncu

How to analyze:
  ncu -i evo2_ncu.ncu-rep                 # Command line
  ncu-ui evo2_ncu.ncu-rep                 # GUI
""")

    # Use shorter sequence because ncu is very slow
    print(f"\nUsing shorter sequence ({seq_len}) because ncu profiles every kernel launch")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    # Single inference for ncu
    torch.cuda.synchronize()
    print("Running single inference pass for ncu profiling...")

    with torch.no_grad():
        output = model.model(input_ids)

    torch.cuda.synchronize()
    print("Done. Analyze with: ncu -i <output>.ncu-rep")

    print("""
Key metrics to examine in ncu:
  1. Occupancy: Are SMs fully utilized?
     - Low occupancy = register pressure or shared memory limit

  2. Memory throughput:
     - Compare achieved GB/s to peak (H200: ~4.8 TB/s HBM)
     - If close to peak = memory bound

  3. Compute throughput:
     - Compare achieved TFLOPS to peak
     - If close to peak = compute bound

  4. Roofline:
     - Shows if kernel is memory or compute bound
     - Indicates optimization direction
""")


def print_profiling_strategy():
    """Print recommended profiling strategy."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    RECOMMENDED PROFILING STRATEGY                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  Step 1: torch.profiler (5 min)                                          ║
║    → Identify which operations take the most time                        ║
║    → Get list of hot kernels                                             ║
║                                                                           ║
║  Step 2: Nsight Systems (15 min)                                         ║
║    → Check GPU utilization over time                                     ║
║    → Find CPU-GPU sync issues                                            ║
║    → Identify kernel launch overhead                                     ║
║                                                                           ║
║  Step 3: Nsight Compute (30+ min)                                        ║
║    → Deep dive into specific slow kernels                                ║
║    → Determine if memory or compute bound                                ║
║    → Guide optimization (fusion, algorithm change, etc.)                 ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  For Evo2 specifically, focus on:                                        ║
║    1. FFT kernels (Hyena convolutions)                                   ║
║    2. GEMM kernels (linear layers)                                       ║
║    3. Attention kernels (Flash Attention)                                ║
║    4. Elementwise kernels (gating, activations)                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Profile Evo2 with different tools")
    parser.add_argument("--method", type=str, default="torch",
                        choices=["torch", "nsys", "ncu", "all", "info"],
                        help="Profiling method to use")
    parser.add_argument("--seq_len", type=int, default=10000,
                        help="Sequence length for profiling")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.method == "info":
        print_profiling_strategy()
        return

    print("="*60)
    print("Evo2 Profiling Comparison")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"SMs: {props.multi_processor_count}")

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    if args.method == "torch" or args.method == "all":
        profile_with_torch(model, args.seq_len)

    if args.method == "nsys" or args.method == "all":
        profile_with_nsys(model, args.seq_len)

    if args.method == "ncu" or args.method == "all":
        profile_with_ncu(model, min(args.seq_len, 5000))

    print_profiling_strategy()


if __name__ == "__main__":
    main()
