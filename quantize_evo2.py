#!/usr/bin/env python3
"""
Evo2 INT8 Quantization Script - Day 2

This script applies INT8 quantization to Evo2 and measures:
- Speedup vs baseline
- Memory reduction
- Accuracy impact (on a sample task)

Methods tried:
1. Dynamic quantization (torch.ao)
2. bitsandbytes 8-bit
3. Static quantization (if calibration data available)
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("./quantization_results")
NUM_WARMUP = 2
NUM_RUNS = 5


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0


def generate_random_dna(length):
    """Generate random DNA sequence."""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def benchmark_model(model, model_fn, seq_lengths, name="model"):
    """Benchmark a model at different sequence lengths."""
    results = []

    for seq_len in seq_lengths:
        print(f"  Benchmarking {name} at {seq_len:,} tokens...")

        seq = generate_random_dna(seq_len)
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).cuda()

        # Warmup
        for _ in range(NUM_WARMUP):
            with torch.no_grad():
                _ = model_fn(input_ids)
            torch.cuda.synchronize()

        # Clear cache and reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Timed runs
        times = []
        for _ in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model_fn(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        results.append({
            'seq_length': seq_len,
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'tokens_per_second': seq_len / np.mean(times),
            'peak_memory_gb': peak_mem,
        })

    return results


def try_dynamic_quantization(model):
    """Try PyTorch dynamic quantization."""
    print("\n" + "="*60)
    print("Method 1: PyTorch Dynamic Quantization")
    print("="*60)

    try:
        # Dynamic quantization for linear layers
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        # Count quantized layers
        num_quantized = 0
        for name, module in quantized_model.named_modules():
            if 'DynamicQuantizedLinear' in str(type(module)):
                num_quantized += 1

        print(f"  Quantized {num_quantized} linear layers to INT8")

        return quantized_model, True

    except Exception as e:
        print(f"  Failed: {e}")
        return None, False


def try_bitsandbytes_8bit(model_name="evo2_7b"):
    """Try loading model with bitsandbytes 8-bit quantization."""
    print("\n" + "="*60)
    print("Method 2: bitsandbytes 8-bit Loading")
    print("="*60)

    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes version: {bnb.__version__}")

        # Check if Evo2 supports load_in_8bit
        from evo2 import Evo2

        # Try to load with 8-bit - this may or may not work depending on Evo2's implementation
        # First check if there's a quantization option
        import inspect
        sig = inspect.signature(Evo2.__init__)
        print(f"  Evo2.__init__ parameters: {list(sig.parameters.keys())}")

        # If not directly supported, we can try manual quantization
        print("  Attempting manual bitsandbytes quantization...")

        return None, False

    except ImportError:
        print("  bitsandbytes not installed. Install with: pip install bitsandbytes")
        return None, False
    except Exception as e:
        print(f"  Failed: {e}")
        return None, False


def try_manual_int8_linear(model):
    """Manually replace Linear layers with INT8 versions."""
    print("\n" + "="*60)
    print("Method 3: Manual INT8 Linear Layer Replacement")
    print("="*60)

    try:
        import copy

        # Count original linear layers
        num_linear = sum(1 for m in model.model.modules() if isinstance(m, torch.nn.Linear))
        print(f"  Found {num_linear} Linear layers")

        # Create a copy and quantize
        # This is a simplified approach - real production would need more care
        quantized_model = copy.deepcopy(model.model)

        replaced = 0
        for name, module in quantized_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = quantized_model.get_submodule(parent_name)
                else:
                    parent = quantized_model
                    attr_name = name

                # Create quantized version
                try:
                    quantized_linear = torch.ao.nn.quantized.dynamic.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        dtype=torch.qint8
                    )
                    # Copy weights (quantized)
                    quantized_linear.set_weight_bias(
                        torch.quantize_per_tensor(
                            module.weight.float().cpu(),
                            scale=module.weight.abs().max() / 127,
                            zero_point=0,
                            dtype=torch.qint8
                        ),
                        module.bias.cpu() if module.bias is not None else None
                    )
                    setattr(parent, attr_name, quantized_linear)
                    replaced += 1
                except Exception as e:
                    pass  # Skip layers that can't be quantized

        print(f"  Replaced {replaced}/{num_linear} layers with INT8")

        return quantized_model if replaced > 0 else None, replaced > 0

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def try_half_precision(model):
    """Convert to FP16 as a simpler optimization."""
    print("\n" + "="*60)
    print("Method 4: FP16 Half Precision (baseline comparison)")
    print("="*60)

    try:
        # Check current dtype
        sample_param = next(model.model.parameters())
        print(f"  Current dtype: {sample_param.dtype}")

        if sample_param.dtype == torch.float16 or sample_param.dtype == torch.bfloat16:
            print("  Model already in half precision")
            return model.model, True

        # Convert to FP16
        model.model.half()
        new_dtype = next(model.model.parameters()).dtype
        print(f"  Converted to: {new_dtype}")

        return model.model, True

    except Exception as e:
        print(f"  Failed: {e}")
        return None, False


def compare_outputs(model, baseline_fn, quantized_fn, seq_len=1000):
    """Compare outputs between baseline and quantized model."""
    print("\n  Comparing output accuracy...")

    seq = generate_random_dna(seq_len)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).cuda()

    with torch.no_grad():
        baseline_out = baseline_fn(input_ids)
        quantized_out = quantized_fn(input_ids)

    # Handle tuple outputs
    if isinstance(baseline_out, tuple):
        baseline_out = baseline_out[0]
    if isinstance(quantized_out, tuple):
        quantized_out = quantized_out[0]

    # Convert to same dtype for comparison
    baseline_out = baseline_out.float()
    quantized_out = quantized_out.float()

    # Calculate metrics
    mse = torch.mean((baseline_out - quantized_out) ** 2).item()
    mae = torch.mean(torch.abs(baseline_out - quantized_out)).item()
    max_diff = torch.max(torch.abs(baseline_out - quantized_out)).item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        baseline_out.flatten().unsqueeze(0),
        quantized_out.flatten().unsqueeze(0)
    ).item()

    print(f"    MSE: {mse:.6f}")
    print(f"    MAE: {mae:.6f}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Cosine similarity: {cos_sim:.6f}")

    return {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'cosine_similarity': cos_sim,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("Evo2 INT8 Quantization - Day 2")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {total_mem:.1f} GB")

    # Load baseline model
    print("\nLoading baseline Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    # Check baseline dtype
    sample_param = next(model.model.parameters())
    print(f"Baseline dtype: {sample_param.dtype}")
    print(f"Baseline device: {sample_param.device}")

    # Test sequence lengths
    seq_lengths = [1000, 10000, 50000]

    results = {
        'baseline': {},
        'methods': {}
    }

    # Baseline benchmark
    print("\n" + "="*60)
    print("Baseline Benchmarks")
    print("="*60)
    results['baseline'] = benchmark_model(
        model, lambda x: model.model(x), seq_lengths, "baseline"
    )

    # Try different quantization methods
    methods = [
        ("dynamic_quantization", try_dynamic_quantization),
        ("fp16", try_half_precision),
    ]

    for method_name, method_fn in methods:
        if method_name == "fp16":
            # Reload model for FP16 test (since previous methods may have modified it)
            print("\nReloading model for FP16 test...")
            model = Evo2("evo2_7b")

        quantized, success = method_fn(model)

        if success and quantized is not None:
            print(f"\n  Benchmarking {method_name}...")

            # Create inference function
            if hasattr(quantized, '__call__'):
                quant_fn = lambda x, q=quantized: q(x)
            else:
                quant_fn = lambda x, q=quantized: q(x)

            # Benchmark
            try:
                quant_results = benchmark_model(
                    model, quant_fn, seq_lengths, method_name
                )
                results['methods'][method_name] = {
                    'benchmarks': quant_results,
                    'success': True
                }

                # Compare accuracy
                accuracy = compare_outputs(
                    model,
                    lambda x: model.model(x),
                    quant_fn,
                    seq_len=1000
                )
                results['methods'][method_name]['accuracy'] = accuracy

            except Exception as e:
                print(f"  Benchmark failed: {e}")
                results['methods'][method_name] = {'success': False, 'error': str(e)}
        else:
            results['methods'][method_name] = {'success': False}

    # Save results
    results_path = OUTPUT_DIR / "quantization_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved results to: {results_path}")

    # Print comparison summary
    print("\n" + "="*60)
    print("SUMMARY: Speedup Comparison")
    print("="*60)

    baseline_by_len = {r['seq_length']: r for r in results['baseline']}

    print(f"\n{'Method':<25} {'Seq Len':>10} {'Time (ms)':>12} {'Speedup':>10} {'Memory':>10}")
    print("-"*70)

    for seq_len in seq_lengths:
        baseline = baseline_by_len[seq_len]
        print(f"{'baseline':<25} {seq_len:>10,} {baseline['mean_time_ms']:>12.1f} {'1.00x':>10} {baseline['peak_memory_gb']:>9.1f}G")

        for method_name, method_data in results['methods'].items():
            if method_data.get('success') and 'benchmarks' in method_data:
                for bench in method_data['benchmarks']:
                    if bench['seq_length'] == seq_len:
                        speedup = baseline['mean_time_ms'] / bench['mean_time_ms']
                        mem_reduction = baseline['peak_memory_gb'] / bench['peak_memory_gb']
                        print(f"{method_name:<25} {seq_len:>10,} {bench['mean_time_ms']:>12.1f} {speedup:>9.2f}x {bench['peak_memory_gb']:>9.1f}G")

    # Accuracy summary
    print("\n" + "="*60)
    print("SUMMARY: Accuracy Impact")
    print("="*60)

    for method_name, method_data in results['methods'].items():
        if method_data.get('success') and 'accuracy' in method_data:
            acc = method_data['accuracy']
            print(f"\n{method_name}:")
            print(f"  Cosine similarity: {acc['cosine_similarity']:.6f}")
            print(f"  MSE: {acc['mse']:.6f}")

    print("\n" + "="*60)
    print("Next: Review results and prepare presentation")
    print("="*60)


if __name__ == "__main__":
    main()
