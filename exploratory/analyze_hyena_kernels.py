#!/usr/bin/env python3
"""
Analyze Hyena Kernel Fusion Opportunities

This script:
1. Profiles the Hyena layers specifically
2. Identifies kernel launch patterns
3. Finds fusion opportunities
4. Tests FlashFFTConv if available
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("./hyena_analysis")


def generate_random_dna(length):
    """Generate random DNA sequence."""
    bases = ['A', 'T', 'C', 'G']
    return ''.join(np.random.choice(bases, length))


def analyze_model_structure(model):
    """Analyze the model structure to find Hyena components."""
    print("\n" + "="*60)
    print("Model Structure Analysis")
    print("="*60)

    hyena_modules = []
    attention_modules = []
    other_modules = []

    for name, module in model.model.named_modules():
        module_type = type(module).__name__

        if 'hyena' in name.lower() or 'hyena' in module_type.lower():
            hyena_modules.append((name, module_type))
        elif 'attention' in name.lower() or 'attention' in module_type.lower():
            attention_modules.append((name, module_type))
        elif name.count('.') <= 2:  # Top-level modules
            other_modules.append((name, module_type))

    print(f"\nFound {len(hyena_modules)} Hyena-related modules:")
    for name, mtype in hyena_modules[:20]:
        print(f"  {name}: {mtype}")

    print(f"\nFound {len(attention_modules)} Attention-related modules:")
    for name, mtype in attention_modules[:20]:
        print(f"  {name}: {mtype}")

    return {
        'hyena_modules': hyena_modules,
        'attention_modules': attention_modules,
    }


def profile_kernel_launches(model, seq_len=10000):
    """Profile individual kernel launches to find fusion opportunities."""
    print("\n" + "="*60)
    print(f"Kernel Launch Analysis (seq_len={seq_len:,})")
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

    # Detailed profiling with stack traces
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            output = model.model(input_ids)
        torch.cuda.synchronize()

    # Analyze kernel patterns
    kernel_stats = defaultdict(lambda: {'count': 0, 'cuda_time': 0, 'shapes': set()})

    for event in prof.key_averages():
        if event.cuda_time_total > 0:
            name = event.key
            kernel_stats[name]['count'] += event.count
            kernel_stats[name]['cuda_time'] += event.cuda_time_total / 1000  # Convert to ms

            if event.input_shapes:
                kernel_stats[name]['shapes'].add(str(event.input_shapes))

    # Sort by total CUDA time
    sorted_kernels = sorted(
        kernel_stats.items(),
        key=lambda x: x[1]['cuda_time'],
        reverse=True
    )

    # Identify kernel categories
    fft_kernels = []
    gemm_kernels = []
    elementwise_kernels = []
    other_kernels = []

    for name, stats in sorted_kernels:
        name_lower = name.lower()
        if 'fft' in name_lower or 'cufft' in name_lower:
            fft_kernels.append((name, stats))
        elif 'gemm' in name_lower or 'mm' in name_lower or 'linear' in name_lower:
            gemm_kernels.append((name, stats))
        elif any(op in name_lower for op in ['mul', 'add', 'relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'exp']):
            elementwise_kernels.append((name, stats))
        else:
            other_kernels.append((name, stats))

    print("\n## FFT Operations (Hyena convolutions)")
    total_fft_time = sum(s['cuda_time'] for _, s in fft_kernels)
    print(f"Total FFT time: {total_fft_time:.2f} ms")
    for name, stats in fft_kernels[:10]:
        print(f"  {name[:50]}: {stats['cuda_time']:.2f}ms ({stats['count']} calls)")

    print("\n## GEMM/Linear Operations")
    total_gemm_time = sum(s['cuda_time'] for _, s in gemm_kernels)
    print(f"Total GEMM time: {total_gemm_time:.2f} ms")
    for name, stats in gemm_kernels[:10]:
        print(f"  {name[:50]}: {stats['cuda_time']:.2f}ms ({stats['count']} calls)")

    print("\n## Elementwise Operations (fusion candidates)")
    total_ewise_time = sum(s['cuda_time'] for _, s in elementwise_kernels)
    print(f"Total elementwise time: {total_ewise_time:.2f} ms")
    for name, stats in elementwise_kernels[:15]:
        print(f"  {name[:50]}: {stats['cuda_time']:.2f}ms ({stats['count']} calls)")

    # Calculate fusion potential
    print("\n" + "="*60)
    print("FUSION OPPORTUNITY ANALYSIS")
    print("="*60)

    total_time = sum(s['cuda_time'] for _, s in sorted_kernels)

    print(f"\nTime breakdown:")
    print(f"  FFT operations:        {total_fft_time:>8.2f} ms ({100*total_fft_time/total_time:.1f}%)")
    print(f"  GEMM operations:       {total_gemm_time:>8.2f} ms ({100*total_gemm_time/total_time:.1f}%)")
    print(f"  Elementwise ops:       {total_ewise_time:>8.2f} ms ({100*total_ewise_time/total_time:.1f}%)")
    print(f"  Other:                 {total_time - total_fft_time - total_gemm_time - total_ewise_time:>8.2f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  Total:                 {total_time:>8.2f} ms")

    # Fusion recommendations
    print("\n## Fusion Recommendations:")

    if total_ewise_time > total_time * 0.1:
        print(f"  1. HIGH PRIORITY: Fuse elementwise operations ({total_ewise_time:.1f}ms, {len(elementwise_kernels)} kernels)")
        print(f"     - Multiple small kernels → single fused kernel")
        print(f"     - Potential savings: 30-50% of elementwise time")

    if total_fft_time > total_time * 0.1:
        print(f"  2. HIGH PRIORITY: Use FlashFFTConv for Hyena ({total_fft_time:.1f}ms)")
        print(f"     - Fuses FFT → multiply → IFFT")
        print(f"     - Reduces memory traffic")
        print(f"     - Potential savings: 30-50% of FFT time")

    if len(gemm_kernels) > 10:
        print(f"  3. MEDIUM PRIORITY: Fuse consecutive linear layers")
        print(f"     - LayerNorm + Linear → Fused kernel")
        print(f"     - Bias + Activation → Fused kernel")

    return {
        'total_time_ms': total_time,
        'fft_time_ms': total_fft_time,
        'gemm_time_ms': total_gemm_time,
        'elementwise_time_ms': total_ewise_time,
        'num_fft_kernels': len(fft_kernels),
        'num_gemm_kernels': len(gemm_kernels),
        'num_elementwise_kernels': len(elementwise_kernels),
        'top_kernels': [(name, stats['cuda_time']) for name, stats in sorted_kernels[:30]],
    }


def check_flashfftconv():
    """Check if FlashFFTConv is available."""
    print("\n" + "="*60)
    print("FlashFFTConv Availability Check")
    print("="*60)

    try:
        from flashfftconv import FlashFFTConv
        print("  FlashFFTConv is available!")

        # Test basic functionality
        conv = FlashFFTConv(128, dtype=torch.bfloat16).cuda()
        x = torch.randn(1, 128, 1024, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(128, 1024, dtype=torch.bfloat16, device='cuda')

        out = conv(x, k)
        print(f"  Test passed: input {x.shape} → output {out.shape}")

        return True
    except ImportError:
        print("  FlashFFTConv not installed")
        print("  Install with: pip install flashfftconv")
        return False
    except Exception as e:
        print(f"  FlashFFTConv error: {e}")
        return False


def analyze_hyena_layer(model):
    """Deep dive into a single Hyena layer."""
    print("\n" + "="*60)
    print("Hyena Layer Deep Dive")
    print("="*60)

    # Find first Hyena layer
    hyena_layer = None
    hyena_name = None

    for name, module in model.model.named_modules():
        module_type = type(module).__name__
        if 'hyena' in module_type.lower():
            hyena_layer = module
            hyena_name = name
            break

    if hyena_layer is None:
        print("  No Hyena layer found - checking for similar SSM layers...")
        for name, module in model.model.named_modules():
            module_type = type(module).__name__
            if any(x in module_type.lower() for x in ['ssm', 'mamba', 'conv', 'mixer']):
                print(f"  Found: {name}: {module_type}")

        return None

    print(f"\nAnalyzing: {hyena_name}")
    print(f"Type: {type(hyena_layer).__name__}")

    # Print layer structure
    print("\nLayer components:")
    for name, child in hyena_layer.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, 'weight'):
            print(f"       weight shape: {child.weight.shape}")

    # Print forward method signature if available
    import inspect
    try:
        sig = inspect.signature(hyena_layer.forward)
        print(f"\nForward signature: {sig}")
    except:
        pass

    return hyena_layer


def create_fused_kernel_template():
    """Create a template for a fused CUDA kernel."""
    print("\n" + "="*60)
    print("Fused Kernel Template")
    print("="*60)

    template = '''
// Fused Hyena Gating Kernel
// Fuses: multiply + silu activation + multiply
// Before: 3 kernel launches, 3x memory read/write
// After: 1 kernel launch, 1x memory read/write

__global__ void fused_hyena_gate_kernel(
    const float* __restrict__ x,      // Input [B, L, D]
    const float* __restrict__ gate1,  // Gate 1 [B, L, D]
    const float* __restrict__ gate2,  // Gate 2 [B, L, D]
    float* __restrict__ output,       // Output [B, L, D]
    int B, int L, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L * D;

    if (idx < total) {
        float x_val = x[idx];
        float g1 = gate1[idx];
        float g2 = gate2[idx];

        // Fused: x * silu(g1) * g2
        // silu(x) = x * sigmoid(x)
        float silu_g1 = g1 * (1.0f / (1.0f + expf(-g1)));
        output[idx] = x_val * silu_g1 * g2;
    }
}

// Python wrapper using torch.utils.cpp_extension
// or use Triton for easier development:

import triton
import triton.language as tl

@triton.jit
def fused_hyena_gate_triton(
    x_ptr, gate1_ptr, gate2_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    g1 = tl.load(gate1_ptr + offsets, mask=mask)
    g2 = tl.load(gate2_ptr + offsets, mask=mask)

    # Fused silu gating
    silu_g1 = g1 * tl.sigmoid(g1)
    output = x * silu_g1 * g2

    tl.store(output_ptr + offsets, output, mask=mask)
'''

    print(template)

    # Save template
    with open(OUTPUT_DIR / "fused_kernel_template.txt", 'w') as f:
        f.write(template)

    print(f"\nSaved template to: {OUTPUT_DIR}/fused_kernel_template.txt")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*60)
    print("Hyena Kernel Fusion Analysis")
    print("="*60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading Evo2 model...")
    from evo2 import Evo2
    model = Evo2("evo2_7b")

    results = {}

    # 1. Analyze model structure
    results['structure'] = analyze_model_structure(model)

    # 2. Check FlashFFTConv
    results['flashfftconv_available'] = check_flashfftconv()

    # 3. Profile kernel launches
    results['kernel_analysis'] = profile_kernel_launches(model, seq_len=10000)

    # 4. Analyze Hyena layer
    analyze_hyena_layer(model)

    # 5. Create fused kernel template
    create_fused_kernel_template()

    # Save results
    # Convert sets to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        return obj

    results = convert_sets(results)

    results_path = OUTPUT_DIR / "hyena_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nSaved analysis to: {results_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Kernel Fusion Roadmap")
    print("="*60)

    ka = results['kernel_analysis']
    print(f"""
1. QUICK WIN - Elementwise Fusion:
   - Current: {ka['num_elementwise_kernels']} separate kernels, {ka['elementwise_time_ms']:.1f}ms
   - Use Triton to fuse gating operations
   - Expected savings: 30-50% of elementwise time

2. MEDIUM EFFORT - FlashFFTConv:
   - Current FFT time: {ka['fft_time_ms']:.1f}ms
   - Install: pip install flashfftconv
   - Replace standard FFT conv with FlashFFTConv
   - Expected savings: 30-50% of FFT time

3. ADVANCED - Custom Hyena Kernel:
   - Fuse entire Hyena operator into single kernel
   - Requires deep understanding of Hyena math
   - Maximum potential but highest effort
""")


if __name__ == "__main__":
    main()
