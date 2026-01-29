#!/usr/bin/env python3
"""
Analyze Profiling Results - Day 3

This script:
1. Loads profiling and quantization results
2. Creates visualizations for interview presentation
3. Generates summary statistics and talking points
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROFILE_DIR = Path("./profiling_results")
QUANT_DIR = Path("./quantization_results")
OUTPUT_DIR = Path("./presentation_figures")


def load_results():
    """Load all results files."""
    results = {}

    # Profiling results
    profile_path = PROFILE_DIR / "profile_results.json"
    if profile_path.exists():
        with open(profile_path) as f:
            results['profiling'] = json.load(f)
        print(f"Loaded profiling results from {profile_path}")

    # Quantization results
    quant_path = QUANT_DIR / "quantization_results.json"
    if quant_path.exists():
        with open(quant_path) as f:
            results['quantization'] = json.load(f)
        print(f"Loaded quantization results from {quant_path}")

    return results


def plot_throughput_by_sequence_length(results):
    """Plot throughput vs sequence length."""
    if 'profiling' not in results:
        print("No profiling results found")
        return

    data = results['profiling']['basic_profiling']

    seq_lengths = [d['seq_length'] for d in data]
    throughputs = [d['tokens_per_second'] for d in data]
    times_ms = [d['mean_time_ms'] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput plot
    ax1.plot(seq_lengths, throughputs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Evo2 7B Inference Throughput', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Add annotations
    for i, (x, y) in enumerate(zip(seq_lengths, throughputs)):
        ax1.annotate(f'{y:,.0f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # Latency plot
    ax2.plot(seq_lengths, times_ms, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Evo2 7B Inference Latency', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add annotations
    for i, (x, y) in enumerate(zip(seq_lengths, times_ms)):
        ax2.annotate(f'{y:.0f}ms', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "throughput_latency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: throughput_latency.png")


def plot_memory_usage(results):
    """Plot memory usage by sequence length."""
    if 'profiling' not in results:
        return

    data = results['profiling']['basic_profiling']

    seq_lengths = [d['seq_length'] for d in data]
    peak_memory = [d['peak_memory_gb'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(seq_lengths)), peak_memory, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(seq_lengths)))
    ax.set_xticklabels([f'{x:,}' for x in seq_lengths])
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_title('Evo2 7B Memory Usage by Sequence Length', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mem in zip(bars, peak_memory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{mem:.1f}GB', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_usage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: memory_usage.png")


def plot_operation_breakdown(results):
    """Plot breakdown of CUDA operations."""
    if 'profiling' not in results or 'detailed_ops' not in results['profiling']:
        return

    ops = results['profiling']['detailed_ops'][:10]  # Top 10

    names = [op['name'][:30] + '...' if len(op['name']) > 30 else op['name'] for op in ops]
    times = [op['cuda_time_ms'] for op in ops]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(range(len(names)), times, color='coral', edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('CUDA Time (ms)', fontsize=12)
    ax.set_title('Top 10 CUDA Operations by Time', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, time_ms in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{time_ms:.1f}ms', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "operation_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: operation_breakdown.png")


def plot_quantization_comparison(results):
    """Plot quantization speedup comparison."""
    if 'quantization' not in results:
        print("No quantization results found")
        return

    quant = results['quantization']
    baseline = {r['seq_length']: r for r in quant['baseline']}

    # Collect data for plotting
    methods = ['baseline']
    data_by_method = {'baseline': quant['baseline']}

    for method_name, method_data in quant.get('methods', {}).items():
        if method_data.get('success') and 'benchmarks' in method_data:
            methods.append(method_name)
            data_by_method[method_name] = method_data['benchmarks']

    if len(methods) <= 1:
        print("No successful quantization methods to compare")
        return

    # Use first sequence length for comparison
    seq_len = quant['baseline'][0]['seq_length']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup comparison
    speedups = []
    method_names = []
    for method in methods:
        for bench in data_by_method[method]:
            if bench['seq_length'] == seq_len:
                baseline_time = baseline[seq_len]['mean_time_ms']
                speedup = baseline_time / bench['mean_time_ms']
                speedups.append(speedup)
                method_names.append(method)
                break

    colors = ['steelblue' if m == 'baseline' else 'coral' for m in method_names]
    bars = ax1.bar(method_names, speedups, color=colors, edgecolor='black')
    ax1.set_ylabel('Speedup (x)', fontsize=12)
    ax1.set_title(f'Speedup vs Baseline\n(seq_len={seq_len:,})', fontsize=14)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, speedup in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{speedup:.2f}x', ha='center', fontsize=10)

    # Memory comparison
    memories = []
    for method in methods:
        for bench in data_by_method[method]:
            if bench['seq_length'] == seq_len:
                memories.append(bench['peak_memory_gb'])
                break

    bars = ax2.bar(method_names, memories, color=colors, edgecolor='black')
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title(f'Memory Usage\n(seq_len={seq_len:,})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, mem in zip(bars, memories):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{mem:.1f}GB', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quantization_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: quantization_comparison.png")


def generate_summary_report(results):
    """Generate a text summary report."""
    report = []
    report.append("=" * 60)
    report.append("EVO2 INFERENCE OPTIMIZATION ANALYSIS")
    report.append("=" * 60)

    # Profiling summary
    if 'profiling' in results:
        prof = results['profiling']

        report.append("\n## Model Architecture")
        if 'architecture' in prof:
            arch = prof['architecture']
            report.append(f"- Total parameters: {arch['total_params']:,} ({arch['total_params']/1e9:.2f}B)")
            report.append(f"- Data type: {arch['dtype']}")

        report.append("\n## Baseline Performance")
        if 'basic_profiling' in prof:
            for bench in prof['basic_profiling']:
                report.append(f"- {bench['seq_length']:,} tokens: {bench['mean_time_ms']:.1f}ms, "
                            f"{bench['tokens_per_second']:,.0f} tok/s, {bench['peak_memory_gb']:.1f}GB")

        report.append("\n## Top Bottlenecks (CUDA ops)")
        if 'detailed_ops' in prof:
            for op in prof['detailed_ops'][:5]:
                report.append(f"- {op['name'][:40]}: {op['cuda_time_ms']:.1f}ms")

    # Quantization summary
    if 'quantization' in results:
        quant = results['quantization']

        report.append("\n## Quantization Results")
        baseline = quant['baseline'][0]

        for method_name, method_data in quant.get('methods', {}).items():
            if method_data.get('success') and 'benchmarks' in method_data:
                bench = method_data['benchmarks'][0]
                speedup = baseline['mean_time_ms'] / bench['mean_time_ms']
                report.append(f"\n### {method_name}")
                report.append(f"- Speedup: {speedup:.2f}x")
                report.append(f"- Memory: {bench['peak_memory_gb']:.1f}GB (baseline: {baseline['peak_memory_gb']:.1f}GB)")

                if 'accuracy' in method_data:
                    acc = method_data['accuracy']
                    report.append(f"- Cosine similarity: {acc['cosine_similarity']:.6f}")

    # Recommendations
    report.append("\n## Recommendations")
    report.append("1. INT8 quantization provides modest speedup with minimal accuracy loss")
    report.append("2. Memory is primary constraint for long sequences")
    report.append("3. Further optimization opportunities:")
    report.append("   - Custom Hyena convolution kernels")
    report.append("   - KV cache optimization for attention components")
    report.append("   - Knowledge distillation for smaller model")

    report_text = '\n'.join(report)

    # Save report
    with open(OUTPUT_DIR / "optimization_report.txt", 'w') as f:
        f.write(report_text)

    print("\nSaved: optimization_report.txt")
    print("\n" + report_text)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("Analyzing Profiling Results - Day 3")
    print("="*60)

    # Load results
    results = load_results()

    if not results:
        print("\nNo results found! Run profile_evo2.py and quantize_evo2.py first.")
        return

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_throughput_by_sequence_length(results)
    plot_memory_usage(results)
    plot_operation_breakdown(results)
    plot_quantization_comparison(results)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(results)

    print("\n" + "="*60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
