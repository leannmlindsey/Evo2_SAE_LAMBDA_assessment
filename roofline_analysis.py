#!/usr/bin/env python3
"""
Roofline Analysis for Evo2

A roofline plot shows whether kernels are:
- Memory-bound (below the diagonal)
- Compute-bound (below the horizontal ceiling)

H200 Specs:
- Memory bandwidth: ~4.8 TB/s (HBM3e)
- FP16 Tensor Core: ~1,979 TFLOPS
- FP32: ~67 TFLOPS
- BF16: ~1,979 TFLOPS

This script:
1. Runs ncu to collect roofline metrics
2. Parses the results
3. Creates a roofline plot
"""

import subprocess
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("./roofline_results")

# H200 specifications
H200_SPECS = {
    'memory_bandwidth_tb_s': 4.8,  # TB/s
    'memory_bandwidth_gb_s': 4800,  # GB/s
    'fp32_tflops': 67,
    'fp16_tflops': 1979,
    'bf16_tflops': 1979,
    'fp8_tflops': 3958,
    'int8_tops': 3958,
}


def create_ncu_command(output_file, seq_len=5000):
    """Create ncu command for roofline analysis."""

    # Metrics needed for roofline
    metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM utilization
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",  # Memory throughput %
        "dram__bytes.sum",  # Total DRAM bytes
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",  # FP32 add ops
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",  # FP32 mul ops
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",  # FP32 FMA ops
        "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",  # FP16 add ops
        "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",  # FP16 mul ops
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",  # FP16 FMA ops
        "gpu__time_duration.sum",  # Kernel duration
        "launch__grid_size",  # Grid size
        "launch__block_size",  # Block size
    ]

    metrics_str = ",".join(metrics)

    cmd = f"""ncu --set roofline \\
    --target-processes all \\
    --export {output_file} \\
    --force-overwrite \\
    python -c "
import torch
import numpy as np
from evo2 import Evo2

# Generate sequence
seq = ''.join(np.random.choice(['A','T','C','G'], {seq_len}))

# Load model
model = Evo2('evo2_7b')

# Warmup
input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).cuda()
with torch.no_grad():
    for _ in range(2):
        _ = model.model(input_ids)
torch.cuda.synchronize()

# Profile this inference
with torch.no_grad():
    output = model.model(input_ids)
torch.cuda.synchronize()
"
"""
    return cmd


def create_simple_ncu_command(output_file, seq_len=1000):
    """Simpler ncu command - just collect basic metrics."""

    cmd = [
        "ncu",
        "--set", "full",
        "--export", output_file,
        "--force-overwrite",
        "python", "-c",
        f"""
import torch
import numpy as np
from evo2 import Evo2

seq = ''.join(np.random.choice(['A','T','C','G'], {seq_len}))
model = Evo2('evo2_7b')
input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).cuda()

# Warmup
with torch.no_grad():
    for _ in range(2):
        _ = model.model(input_ids)
torch.cuda.synchronize()

# Profile
with torch.no_grad():
    output = model.model(input_ids)
torch.cuda.synchronize()
"""
    ]
    return cmd


def estimate_roofline_from_nsys():
    """Estimate roofline position from nsys data we already have."""
    print("\n" + "="*60)
    print("Estimating Roofline from Existing Data")
    print("="*60)

    # From nsys stats, we have:
    # - Total kernel time: ~442ms (Self CUDA time)
    # - Memory transfers: 13.7 GB HtoD, but this is warmup

    # From the kernel summary, top kernels:
    kernels = [
        # (name, time_ns, estimated_bytes, estimated_flops)
        ("nvjet_tst_256x160 (GEMM)", 245156341, None, None),
        ("nvjet_tst_128x304 (GEMM)", 127211233, None, None),
        ("conv_depthwise2d", 124831613, None, None),
        ("elementwise_kernel", 93372987, None, None),
        ("vector_fft", 45760919, None, None),
        ("flash_fwd_kernel", 36121318, None, None),
    ]

    total_time_ns = sum(k[1] for k in kernels)
    total_time_s = total_time_ns / 1e9

    print(f"\nTop kernels total time: {total_time_s*1000:.2f} ms")

    # For GEMM kernels, we can estimate:
    # For a GEMM of size MxNxK:
    #   FLOPs = 2*M*N*K (multiply-add)
    #   Bytes = (M*K + K*N + M*N) * sizeof(dtype)
    #   Arithmetic Intensity = FLOPs / Bytes

    print("""
To get actual roofline data, run ncu:

    ncu --set roofline -o evo2_roofline python your_script.py

Then analyze with:

    ncu -i evo2_roofline.ncu-rep --page raw

Or use ncu-ui for graphical roofline.
""")


def create_theoretical_roofline_plot():
    """Create theoretical roofline plot for H200."""
    print("\n" + "="*60)
    print("Creating Theoretical Roofline Plot")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Arithmetic intensity range (FLOP/Byte)
    ai = np.logspace(-2, 4, 1000)

    # Memory bandwidth ceiling (diagonal)
    mem_bw = H200_SPECS['memory_bandwidth_gb_s']  # GB/s
    mem_ceiling = ai * mem_bw  # GFLOPS

    # Compute ceilings (horizontal)
    fp32_ceiling = H200_SPECS['fp32_tflops'] * 1000  # GFLOPS
    fp16_ceiling = H200_SPECS['fp16_tflops'] * 1000  # GFLOPS
    bf16_ceiling = H200_SPECS['bf16_tflops'] * 1000  # GFLOPS

    # Plot rooflines
    ax.loglog(ai, np.minimum(mem_ceiling, fp32_ceiling), 'b-', linewidth=2, label=f'FP32 ({H200_SPECS["fp32_tflops"]} TFLOPS)')
    ax.loglog(ai, np.minimum(mem_ceiling, fp16_ceiling), 'g-', linewidth=2, label=f'FP16/BF16 ({H200_SPECS["fp16_tflops"]} TFLOPS)')

    # Ridge points (where memory meets compute)
    ridge_fp32 = fp32_ceiling / mem_bw
    ridge_fp16 = fp16_ceiling / mem_bw

    ax.axvline(x=ridge_fp32, color='b', linestyle='--', alpha=0.5, label=f'FP32 ridge point ({ridge_fp32:.1f})')
    ax.axvline(x=ridge_fp16, color='g', linestyle='--', alpha=0.5, label=f'FP16 ridge point ({ridge_fp16:.1f})')

    # Estimated kernel positions (these are rough estimates)
    # For typical transformer operations:
    estimated_kernels = [
        # (name, arithmetic_intensity, achieved_gflops, marker)
        ("GEMM (large)", 100, 800000, 'o'),  # High AI, near compute ceiling
        ("GEMM (small)", 20, 200000, 's'),   # Medium AI
        ("Attention", 50, 500000, '^'),       # Medium-high AI
        ("FFT", 5, 24000, 'D'),               # Low AI, memory bound
        ("Elementwise", 0.5, 2400, 'v'),      # Very low AI, memory bound
        ("Depthwise Conv", 2, 9600, 'p'),     # Low AI, memory bound
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(estimated_kernels)))

    for (name, ai_val, gflops, marker), color in zip(estimated_kernels, colors):
        ax.scatter([ai_val], [gflops], s=200, marker=marker, c=[color],
                   label=f'{name} (AI≈{ai_val})', zorder=5, edgecolors='black')

    # Annotations
    ax.fill_between(ai[ai < ridge_fp16], 0, np.minimum(mem_ceiling, fp16_ceiling)[ai < ridge_fp16],
                    alpha=0.1, color='red', label='Memory Bound Region')
    ax.fill_between(ai[ai > ridge_fp16], 0, np.minimum(mem_ceiling, fp16_ceiling)[ai > ridge_fp16],
                    alpha=0.1, color='blue', label='Compute Bound Region')

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('H200 Roofline Model with Estimated Evo2 Kernel Positions', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(100, 3000000)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "h200_roofline_estimated.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print analysis
    print("""
ROOFLINE INTERPRETATION:
========================

Memory Bound (left of ridge point):
  - Elementwise ops (AI ~0.5): Severely memory bound
  - FFT ops (AI ~5): Memory bound
  - Depthwise conv (AI ~2): Memory bound

  → These benefit from: memory bandwidth optimization, fusion

Compute Bound (right of ridge point):
  - Large GEMM (AI ~100): Compute bound
  - Attention (AI ~50): Compute bound

  → These benefit from: better algorithms, tensor cores

For Evo2:
  - The many elementwise kernels are likely memory bound
  - GEMM/attention are likely compute bound
  - Fusion would help the memory-bound kernels most
""")


def run_ncu_roofline(seq_len=1000):
    """Run ncu with roofline metrics."""
    print("\n" + "="*60)
    print("Running Nsight Compute for Roofline Data")
    print("="*60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "evo2_roofline"

    print(f"Output: {output_file}.ncu-rep")
    print(f"Sequence length: {seq_len}")
    print("\nThis will take several minutes...")
    print("(ncu profiles each kernel individually)\n")

    # Create inline script
    script = f"""
import torch
import numpy as np
from evo2 import Evo2

seq = ''.join(np.random.choice(['A','T','C','G'], {seq_len}))
model = Evo2('evo2_7b')
input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).cuda()

# Warmup
with torch.no_grad():
    for _ in range(2):
        _ = model.model(input_ids)
torch.cuda.synchronize()

# Profile
with torch.no_grad():
    output = model.model(input_ids)
torch.cuda.synchronize()
print("Done!")
"""

    # Write script to file
    script_path = OUTPUT_DIR / "ncu_roofline_script.py"
    with open(script_path, 'w') as f:
        f.write(script)

    # Run ncu
    cmd = [
        "ncu",
        "--set", "roofline",
        "--export", str(output_file),
        "--force-overwrite",
        "python", str(script_path)
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"\nRoofline data saved to: {output_file}.ncu-rep")
        print("\nTo visualize:")
        print(f"  ncu-ui {output_file}.ncu-rep")
        print("\nOr export roofline chart:")
        print(f"  ncu --import {output_file}.ncu-rep --page roofline --csv > roofline.csv")

    except subprocess.TimeoutExpired:
        print("ncu timed out after 30 minutes")
    except FileNotFoundError:
        print("ncu not found. Make sure Nsight Compute is in PATH.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("Roofline Analysis for Evo2")
    print("="*60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ncu", action="store_true", help="Run ncu to collect roofline data (slow!)")
    parser.add_argument("--seq-len", type=int, default=1000, help="Sequence length for ncu profiling")
    args = parser.parse_args()

    # Always create theoretical roofline
    create_theoretical_roofline_plot()

    # Estimate from existing data
    estimate_roofline_from_nsys()

    if args.run_ncu:
        run_ncu_roofline(args.seq_len)
    else:
        print("\n" + "="*60)
        print("To collect actual roofline data, run:")
        print("  python roofline_analysis.py --run-ncu --seq-len 1000")
        print("="*60)


if __name__ == "__main__":
    main()
