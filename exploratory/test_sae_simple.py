#!/usr/bin/env python3
"""
Simple test script that exactly replicates the notebook approach.
"""

import torch
from math import prod
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

    def list_modules(self):
        return self._module_dict.keys()

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

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def load_topk_sae(sae_path, d_hidden, device, dtype, expansion_factor=8):
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")
    new_dict = {}
    for key, item in sae_dict.items():
        new_dict[key.replace("_orig_mod.", "").replace("module.", "")] = item

    cached_sae = BatchTopKTiedSAE(
        d_hidden,
        d_hidden * expansion_factor,
        64,
        device,
        dtype,
    )
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
# Main test - exactly as in notebook
# ============================================================

SAE_LAYER_NAME = 'blocks-26'

def get_feature_ts(model, sae, seq):
    """Exactly as in notebook."""
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    logits, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    feats = sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


if __name__ == "__main__":
    print("Loading Evo2 model...")
    model = ObservableEvo2(model_name="evo2_7b")
    print(f"  Device: {model.device}")
    print(f"  d_hidden: {model.d_hidden}")

    print("\nLoading SAE...")
    file_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model"
    )
    topk_sae = load_topk_sae(
        file_path,
        d_hidden=model.d_hidden,
        device=model.device,
        dtype=torch.bfloat16,
        expansion_factor=8
    )
    print(f"  d_in: {topk_sae.d_in}, d_hidden: {topk_sae.d_hidden}")

    # Test with example sequence from notebook (human genome chunk)
    example_seq = 'TCTGAAAGGACAGTTTTATTGTAGGTACACATGGCTGCCATTTCAAATGTAACTCACAGCTTGTCCATCAGTCCTTGGAGGTCTTTCTATGAAAGGAGCTTGGTGGCGTCCAAACACCACCCAATGTCCACTTAGAAGTAAGCACCGTGTCTGCCCTGAGCTGACTCCTTTTCCAAGGAAGGGGTTGGATCGCTGAGTGTTTTTCCAGGTGTCTACTTGTTGTTAATTAATAGCAATGACAAAGCAGAAGGTTCATGCGTAGCTCGGCTTTCTGGTATTTGCTGCCCGTTGACCAATGGAAGATAAACCTTTGCCTCAGGTGGCACCACTAGCTGGTTAAGAGGCACTTTGTCCTTTCACCCAGGAGCAAACGCACATCACCTGTGTCCTCATCTGATGGCCCTGGTGTGGGGCACAGTCGTGTTGGCAGGGAGGGAGGTGGGGTTGGTCCCCTTTGTGGGTTTGTTGCGAGGCCGTGTTCCAGCTGTTTCCACAGGGAGCGATTTTCAGCTCCACAGGACACTGCTCCCCAGTTCCTCCTGAGAACAAAAGGGGGCGCTGGGGAGAGGCCACCGTTCTGAGGGCTCACTGTATGTGTTCCAGAATCTCCCCTGCAGACCCCCACTGAGGACGGATCTGAGGAACCGGGCTCTGAAACCTCTGATGCTAAGAGCACTCCAACAGCGGAAGGTGGGCCCCCCTTCAGACGCCCCCTCCATGCCTCCAGCCTGTGCTTAGCCGTGCTTTGAGCCTCCCTCCTGGCTGCATCTGCTGCTCCCCCTGGCTGAGAGATGTGCTCACTCCTTCGGTGCTTTGCAGGACAGCGTGGTGGGAGCTGAGCCTTGCGTCGATGCCTTGCTTGCTGGTGCTGAGTGTGGGCACCTTCATCCCGTGTGTGCTCTGGAGGCAGCCACCCTTGGACAGTCCCGCGCACAGCTCCACAAAGCCCCGCTCCATACGATTGTCCTCCCACACCCCCTTCAAAAGCCCCCTCCTCTCT'

    print(f"\nTesting with {len(example_seq)} bp sequence...")
    feature_ts = get_feature_ts(model, topk_sae, example_seq)
    print(f"  Feature activations shape: {feature_ts.shape}")
    print(f"  Activation stats:")
    print(f"    min: {feature_ts.min():.4f}")
    print(f"    max: {feature_ts.max():.4f}")
    print(f"    mean: {feature_ts.mean():.4f}")
    print(f"    std: {feature_ts.std():.4f}")
    print(f"    non-zero: {(feature_ts != 0).sum()} / {feature_ts.size}")

    # Check specific features from notebook
    selected_features = [15680, 28339, 1050, 25666]
    print(f"\n  Selected feature activations (from notebook):")
    for feat_idx in selected_features:
        feat_acts = feature_ts[:, feat_idx]
        print(f"    Feature {feat_idx}: max={feat_acts.max():.4f}, mean={feat_acts.mean():.4f}, non-zero={sum(feat_acts > 0)}")

    # Check feature 19746 (our prophage feature)
    print(f"\n  Feature 19746 (prophage):")
    feat_19746 = feature_ts[:, 19746]
    print(f"    max: {feat_19746.max():.4f}")
    print(f"    mean: {feat_19746.mean():.4f}")
    print(f"    non-zero positions: {sum(feat_19746 > 0)}")

    # Load E. coli and test
    print("\n" + "="*60)
    print("Testing with E. coli genome segments...")
    try:
        with open('./genomes/ecoli_k12.fna', 'r') as f:
            lines = f.readlines()
            ecoli_seq = ''.join(line.strip() for line in lines if not line.startswith('>'))

        # Test known prophage regions in E. coli K-12 MG1655
        prophage_regions = [
            ("e14 prophage", 1195000, 1220000),      # e14 at ~1.19-1.22 Mb
            ("Rac prophage", 1410000, 1435000),      # Rac at ~1.41-1.43 Mb
            ("DLP12 prophage", 560000, 585000),      # DLP12 at ~0.56-0.58 Mb
            ("Non-prophage control", 500000, 525000), # Control region
            ("Qin prophage", 1630000, 1655000),      # Qin at ~1.63-1.65 Mb
        ]

        for name, start, end in prophage_regions:
            test_seq = ecoli_seq[start:end]
            print(f"\n  Testing {name}: {start:,}-{end:,} bp ({len(test_seq):,} bp)")

            feature_ts = get_feature_ts(model, topk_sae, test_seq)

            # Check prophage feature
            feat_19746 = feature_ts[:, 19746]
            print(f"    Feature 19746 (prophage):")
            print(f"      max: {feat_19746.max():.4f}")
            print(f"      mean: {feat_19746.mean():.6f}")
            print(f"      non-zero positions: {sum(feat_19746 > 0)} / {len(feat_19746)}")
            print(f"      positions > 0.5: {sum(feat_19746 > 0.5)}")
            print(f"      positions > 0.1: {sum(feat_19746 > 0.1)}")

    except FileNotFoundError:
        print("  E. coli genome not found at ./genomes/ecoli_k12.fna")

    print("\nDone!")
