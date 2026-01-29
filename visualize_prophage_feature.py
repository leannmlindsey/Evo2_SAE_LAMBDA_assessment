#!/usr/bin/env python3
"""
Visualize Evo2 SAE prophage feature f/19746
============================================
This script exactly replicates the notebook approach:
1. Load Evo2 and SAE
2. Extract feature activations
3. Plot them

No detection logic - just visualization like the paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from math import prod
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from evo2 import Evo2

torch.set_grad_enabled(False)

# ============================================================
# ModelScope - exactly as in notebook
# ============================================================

class ModelScope:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self._build_module_dict()

    def _build_module_dict(self):
        self._module_dict = {}
        def recurse(module, prefix=''):
            for name, child in module.named_children():
                self._module_dict[prefix + name] = child
                recurse(child, prefix=prefix + name + '-')
        recurse(self.model)

    def add_hook(self, hook_fn, module_str, hook_name):
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle

    def remove_all_hooks(self):
        for hook_name in list(self.hooks.keys()):
            self.hooks[hook_name].remove()
            del self.hooks[hook_name]

    def clear_all_caches(self):
        for key in self.activations_cache:
            self.activations_cache[key] = []


# ============================================================
# BatchTopKTiedSAE - exactly as in notebook
# ============================================================

class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, k, device, dtype, tiebreaker_epsilon=1e-6):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(device, dtype)

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)

    def _batch_topk(self, f, k, tiebreak=False):
        if tiebreak:
            f = f + self.tiebreaker.to(f.device).broadcast_to(f.shape)
        *input_shape, _ = f.shape
        numel = k * prod(input_shape)
        f_flat = f.flatten().float()
        f_topk = torch.topk(f_flat, numel, dim=-1)
        result = torch.zeros_like(f_flat).scatter(-1, f_topk.indices, f_topk.values)
        return result.reshape(f.shape).to(f.dtype)

    def decode(self, f):
        return f @ self.W.T + self.b_dec


def load_topk_sae(sae_path, d_hidden, device, dtype, expansion_factor=8):
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")
    new_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sae_dict.items()}
    cached_sae = BatchTopKTiedSAE(d_hidden, d_hidden * expansion_factor, 64, device, dtype)
    cached_sae.load_state_dict(new_dict)
    return cached_sae


# ============================================================
# ObservableEvo2 - exactly as in notebook
# ============================================================

class ObservableEvo2:
    def __init__(self, model_name):
        self.evo_model = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self):
        return next(self.evo_model.model.parameters()).device

    def forward(self, toks, cache_activations_at=None):
        if not cache_activations_at:
            cache_activations_at = []
        output_cache = {}

        for layer in cache_activations_at:
            def _intervene(model, input, output, layer=layer):
                acts = output[0] if isinstance(output, tuple) else output
                output_cache[layer] = acts.detach()
                return (acts, output[1]) if isinstance(output, tuple) else acts
            self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        try:
            model_outputs = self.model(toks)
            cached_activations = {layer: act.clone() for layer, act in output_cache.items()}
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()

        return model_outputs[0], cached_activations


# ============================================================
# Feature extraction - exactly as in notebook
# ============================================================

SAE_LAYER_NAME = 'blocks-26'

def get_feature_ts(model, sae, seq):
    """Exactly as in notebook."""
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    logits, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    feats = sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


# ============================================================
# Main - visualize with ground truth comparison
# ============================================================

def load_ground_truth(csv_path, assembly_id):
    """Load ground truth prophage regions for a specific assembly."""
    import csv
    regions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Assembly'] == assembly_id or row['NCBI Id'] == assembly_id:
                regions.append({
                    'start': int(row['start']),
                    'end': int(row['end']),
                    'organism': row['Organism Name']
                })
    return regions


def load_fasta(fasta_path):
    """Load sequence from FASTA file."""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
        seq = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return seq


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize prophage feature f/19746 with ground truth")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA file")
    parser.add_argument("--ground_truth", type=str, default=None, help="Ground truth CSV file")
    parser.add_argument("--assembly", type=str, default=None, help="Assembly ID to match in ground truth")
    parser.add_argument("--start", type=int, default=None, help="Start position (optional, default=full genome)")
    parser.add_argument("--end", type=int, default=None, help="End position (optional)")
    parser.add_argument("--output", type=str, default="prophage_feature_plot.png", help="Output plot")
    parser.add_argument("--feature", type=int, default=19746, help="Feature index to plot")
    parser.add_argument("--window_size", type=int, default=50000, help="Window size for processing long sequences")
    args = parser.parse_args()

    print("Loading Evo2 model...")
    model = ObservableEvo2(model_name="evo2_7b")
    print(f"  Device: {model.device}")

    print("Loading SAE...")
    file_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model"
    )
    sae = load_topk_sae(file_path, d_hidden=model.d_hidden, device=model.device,
                        dtype=torch.bfloat16, expansion_factor=8)
    print(f"  Loaded SAE: d_in={sae.d_in}, d_hidden={sae.d_hidden}")

    # Load ground truth if provided
    gt_regions = []
    if args.ground_truth and args.assembly:
        print(f"\nLoading ground truth from {args.ground_truth}...")
        gt_regions = load_ground_truth(args.ground_truth, args.assembly)
        print(f"  Found {len(gt_regions)} prophage regions for {args.assembly}")
        for i, r in enumerate(gt_regions):
            print(f"    {i+1}. {r['start']:,}-{r['end']:,} ({r['end']-r['start']:,} bp)")

    print(f"\nLoading sequence from {args.fasta}...")
    full_seq = load_fasta(args.fasta)
    print(f"  Full sequence length: {len(full_seq):,} bp")

    # Determine region to analyze
    if args.start is not None and args.end is not None:
        start, end = args.start, args.end
    elif gt_regions:
        # If ground truth provided, analyze region around first prophage
        first_prophage = gt_regions[0]
        padding = 20000  # 20kb padding around prophage
        start = max(0, first_prophage['start'] - padding)
        end = min(len(full_seq), first_prophage['end'] + padding)
        print(f"  Auto-selected region around first prophage: {start:,}-{end:,}")
    else:
        # Default: first 100kb
        start, end = 0, min(100000, len(full_seq))

    seq = full_seq[start:end]
    print(f"  Analyzing region: {start:,}-{end:,} bp ({len(seq):,} bp)")

    # Extract features in windows if sequence is long
    print(f"\nExtracting feature activations...")
    if len(seq) <= args.window_size:
        feature_ts = get_feature_ts(model, sae, seq)
    else:
        # Process in overlapping windows
        overlap = 1000
        all_acts = np.zeros(len(seq))
        counts = np.zeros(len(seq))

        positions = list(range(0, len(seq), args.window_size - overlap))
        for win_start in tqdm(positions, desc="Processing windows"):
            win_end = min(win_start + args.window_size, len(seq))
            win_seq = seq[win_start:win_end]
            if len(win_seq) < 100:
                continue
            win_feats = get_feature_ts(model, sae, win_seq)
            win_acts = win_feats[:, args.feature]
            actual_len = min(len(win_acts), win_end - win_start)
            # Take max for overlapping regions
            all_acts[win_start:win_start+actual_len] = np.maximum(
                all_acts[win_start:win_start+actual_len],
                win_acts[:actual_len]
            )
        prophage_acts = all_acts
        feature_ts = None  # Don't need full feature matrix

    if feature_ts is not None:
        prophage_acts = feature_ts[:, args.feature]

    print(f"  Shape: {prophage_acts.shape}")
    print(f"\nFeature {args.feature} stats:")
    print(f"  max: {prophage_acts.max():.4f}")
    print(f"  mean: {prophage_acts.mean():.6f}")
    print(f"  non-zero: {sum(prophage_acts > 0)} / {len(prophage_acts)}")
    print(f"  positions > 0.5: {sum(prophage_acts > 0.5)}")

    # Plot with ground truth
    print(f"\nGenerating plot...")
    fig, axes = plt.subplots(2, 1, figsize=(20, 5), height_ratios=[3, 1], sharex=True)

    # Top: Feature activation
    ax1 = axes[0]
    x_coords = np.arange(start, start + len(prophage_acts))
    ax1.plot(x_coords, prophage_acts, lw=0.5, alpha=0.9, color='blue')
    ax1.set_ylabel(f'Feature {args.feature}\nactivation')
    ax1.set_title(f'Evo2 SAE Feature f/{args.feature} (prophage) vs Ground Truth')
    ax1.set_ylim(bottom=0)

    # Add ground truth shading to top plot
    for r in gt_regions:
        if r['start'] < end and r['end'] > start:
            ax1.axvspan(max(r['start'], start), min(r['end'], end),
                       alpha=0.2, color='red', label='Ground truth')

    # Bottom: Ground truth blocks
    ax2 = axes[1]
    ax2.set_xlim(start, end)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Ground\nTruth')
    ax2.set_xlabel('Genomic Position (bp)')
    ax2.set_yticks([])

    for r in gt_regions:
        if r['start'] < end and r['end'] > start:
            rect_start = max(r['start'], start)
            rect_end = min(r['end'], end)
            ax2.axvspan(rect_start, rect_end, alpha=0.7, color='red')
            # Label
            mid = (rect_start + rect_end) / 2
            ax2.text(mid, 0.5, f"{r['end']-r['start']:,}bp",
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    if not gt_regions:
        ax2.text(0.5, 0.5, 'No ground truth provided', transform=ax2.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"  Saved: {args.output}")

    # Save raw activations
    np_file = args.output.replace('.png', '_activations.npy')
    np.save(np_file, prophage_acts)
    print(f"  Saved: {np_file}")

    # Print summary
    if gt_regions:
        print(f"\n{'='*60}")
        print("Summary: Feature activation in ground truth regions")
        print(f"{'='*60}")
        for r in gt_regions:
            if r['start'] < end and r['end'] > start:
                r_start = max(r['start'], start) - start
                r_end = min(r['end'], end) - start
                region_acts = prophage_acts[r_start:r_end]
                print(f"  Region {r['start']:,}-{r['end']:,}:")
                print(f"    max activation: {region_acts.max():.4f}")
                print(f"    mean activation: {region_acts.mean():.6f}")
                print(f"    positions > 0.5: {sum(region_acts > 0.5)} / {len(region_acts)}")

    print("\nDone!")
