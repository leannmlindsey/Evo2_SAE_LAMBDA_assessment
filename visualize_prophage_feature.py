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
# Main - visualize like the notebook
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize prophage feature f/19746")
    parser.add_argument("--fasta", type=str, default="./genomes/ecoli_k12.fna", help="FASTA file")
    parser.add_argument("--start", type=int, default=1195000, help="Start position (e14 prophage)")
    parser.add_argument("--end", type=int, default=1220000, help="End position")
    parser.add_argument("--output", type=str, default="prophage_feature_plot.png", help="Output plot")
    parser.add_argument("--feature", type=int, default=19746, help="Feature index to plot")
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

    print(f"\nLoading sequence from {args.fasta}...")
    with open(args.fasta, 'r') as f:
        lines = f.readlines()
        full_seq = ''.join(line.strip() for line in lines if not line.startswith('>'))

    seq = full_seq[args.start:args.end]
    print(f"  Extracted {args.start:,}-{args.end:,} bp ({len(seq):,} bp)")

    print(f"\nExtracting feature activations...")
    feature_ts = get_feature_ts(model, sae, seq)
    print(f"  Shape: {feature_ts.shape}")

    # Get prophage feature
    prophage_acts = feature_ts[:, args.feature]
    print(f"\nFeature {args.feature} stats:")
    print(f"  max: {prophage_acts.max():.4f}")
    print(f"  mean: {prophage_acts.mean():.6f}")
    print(f"  non-zero: {sum(prophage_acts > 0)} / {len(prophage_acts)}")
    print(f"  positions > 0.5: {sum(prophage_acts > 0.5)}")

    # Plot - exactly like notebook
    print(f"\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.plot(prophage_acts, lw=0.5, alpha=0.9, color='blue')
    ax.set_xlim(0, len(prophage_acts))
    ax.set_xlabel(f'Position (bp from {args.start:,})')
    ax.set_ylabel(f'Feature {args.feature} activation')
    ax.set_title(f'Evo2 SAE Feature f/{args.feature} (prophage) - {args.start:,}-{args.end:,} bp')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"  Saved: {args.output}")

    # Also save the raw activations
    np_file = args.output.replace('.png', '_activations.npy')
    np.save(np_file, prophage_acts)
    print(f"  Saved: {np_file}")

    print("\nDone!")
