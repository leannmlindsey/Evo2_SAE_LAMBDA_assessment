#!/usr/bin/env python3
"""
Quick test: load cached pretrained embeddings, extract 32 random embeddings, compare.

Creates a random StripedHyena with Savanna-style initialization and flash_attn
disabled so it can run in float32 for maximum safety. This tests that:
1. Random embeddings contain no NaN/Inf
2. Random embeddings differ across sequences (not collapsed)
3. Random embeddings are not correlated with pretrained embeddings

The key fix: Vortex's default random init causes NaN because IIR log_poles can be
positive (exponential growth). Savanna (the training framework) always uses negative
log_poles (exponential decay). See: github.com/Zymrael/savanna

Usage:
    python scripts/test_random_embeddings.py \
        --csv_dir /path/to/csv/data \
        --pretrained_embeddings /path/to/embeddings_pretrained.npz \
        --model evo2_7b \
        --layer blocks.28.mlp.l3
"""

import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test random vs pretrained embeddings")
    parser.add_argument("--csv_dir", type=str, required=True)
    parser.add_argument("--pretrained_embeddings", type=str, required=True)
    parser.add_argument("--model", type=str, default="evo2_7b")
    parser.add_argument("--layer", type=str, default="blocks.28.mlp.l3")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_savanna_style_init(state_dict, hidden_size=4096, num_layers=32,
                             short_filter_length=3, seed=42):
    """Apply Savanna-style initialization to a Vortex StripedHyena state_dict.

    See: github.com/Zymrael/savanna/savanna/model/init_functions.py
    and: github.com/Zymrael/savanna/savanna/model/operators/hyena/hyena.py
    """
    torch.manual_seed(seed)

    small_init_std = math.sqrt(2.0 / (5.0 * hidden_size))
    wang_init_std = 2.0 / num_layers / math.sqrt(hidden_size)
    short_conv_bound = math.sqrt(1.0 / short_filter_length)

    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        if '_extra_state' in key or not isinstance(tensor, torch.Tensor):
            continue

        if 'log_poles' in key:
            state_dict[key] = -torch.abs(torch.randn_like(tensor)) * 0.5 - 0.1
        elif 'residues' in key:
            state_size = tensor.shape[-1] if tensor.dim() > 1 else 16
            state_dict[key] = torch.randn_like(tensor) * 0.1 / math.sqrt(state_size)
        elif key.endswith('.D'):
            state_dict[key] = torch.zeros_like(tensor)
        elif '.filter.h' in key or (key.endswith('.h') and 'filter' in key):
            filter_length = tensor.shape[-1]
            state_dict[key] = torch.randn_like(tensor) / math.sqrt(filter_length) * 1e-3
        elif 'short_filter_weight' in key:
            state_dict[key] = torch.empty_like(tensor).uniform_(
                -short_conv_bound, short_conv_bound)
        elif 'short_filter_bias' in key:
            state_dict[key] = torch.zeros_like(tensor)
        elif 'norm' in key and 'weight' in key:
            state_dict[key] = torch.ones_like(tensor)
        elif key.endswith('.bias'):
            state_dict[key] = torch.zeros_like(tensor)
        elif ('out_filter_dense' in key or 'out_proj' in key) and 'weight' in key:
            state_dict[key] = torch.randn_like(tensor) * wang_init_std
        elif 'weight' in key and tensor.dim() >= 2:
            state_dict[key] = torch.randn_like(tensor) * small_init_std


def extract_with_hooks(model, tokenizer, sequences, layer_name, pooling, max_length):
    """Extract embeddings using manual hook registration (no Evo2 wrapper needed)."""
    all_embeddings = []
    for seq in tqdm(sequences, desc="Extracting random embeddings"):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        input_ids = torch.tensor(
            tokenizer.tokenize(seq), dtype=torch.int
        ).unsqueeze(0).to(next(model.parameters()).device)

        captured = {}
        def hook_fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            captured["emb"] = output.detach()

        layer = model.get_submodule(layer_name)
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                model.forward(input_ids)
            emb = captured["emb"]
            if pooling == "mean":
                pooled = emb.mean(dim=1)
            elif pooling == "first":
                pooled = emb[:, 0, :]
            elif pooling == "last":
                pooled = emb[:, -1, :]
            elif pooling == "max":
                pooled = emb.max(dim=1)[0]
            all_embeddings.append(pooled.cpu().float().numpy())
        finally:
            handle.remove()

    return np.vstack(all_embeddings)


def main():
    args = parse_args()
    start = time.time()

    # ---------------------------------------------------------------
    # 1. Load cached pretrained embeddings
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Load cached pretrained embeddings")
    print(f"{'='*60}")
    loaded = np.load(args.pretrained_embeddings)
    pretrained_emb = loaded["train_embeddings"][: args.n_samples]
    print(f"Loaded {pretrained_emb.shape[0]} pretrained embeddings")
    print(f"  Shape: {pretrained_emb.shape}")
    print(f"  Sample [0][:5]: {pretrained_emb[0, :5]}")

    # Load matching sequences
    train_path = f"{args.csv_dir}/train.csv"
    df = pd.read_csv(train_path)
    sequences = df["sequence"].tolist()[: args.n_samples]
    print(f"Using first {len(sequences)} sequences from {train_path}")

    # ---------------------------------------------------------------
    # 2. Create random model (flash_attn disabled, float32,
    #    Savanna-style init)
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Create random model (Savanna-style init, float32)")
    print(f"{'='*60}")

    import yaml
    import pkgutil
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict

    # Get tokenizer from pretrained Evo2
    from evo2 import Evo2
    print("Loading pretrained Evo2 (for tokenizer)...")
    pretrained_model = Evo2(args.model)
    tokenizer = pretrained_model.tokenizer

    # Free pretrained model GPU memory
    del pretrained_model
    torch.cuda.empty_cache()
    print("Freed pretrained model memory")

    # Load config and disable flash attention
    config_name = args.model
    if config_name not in CONFIG_MAP:
        config_name = f"{args.model}_base"
    config_path = CONFIG_MAP[config_name]
    print(f"Config: {config_name} -> {config_path}")

    cfg = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    cfg["use_flash_attn"] = False
    cfg["inference_mode"] = False
    cfg = dotdict(cfg)
    print(f"  use_flash_attn: {cfg.use_flash_attn}")
    print(f"  inference_mode: {cfg.inference_mode}")

    hidden_size = cfg.get("hidden_size", 4096)
    num_layers = cfg.get("num_layers", 32)
    short_filter_length = cfg.get("short_filter_length", 3)

    # Create random model
    torch.manual_seed(args.seed + 100)
    print("Creating StripedHyena with random init...")
    random_model = StripedHyena(cfg)

    # Apply Savanna-style initialization BEFORE casting to float32
    print("Applying Savanna-style initialization...")
    random_sd = {k: v.cpu() for k, v in random_model.state_dict().items()}
    apply_savanna_style_init(
        random_sd,
        hidden_size=hidden_size,
        num_layers=num_layers,
        short_filter_length=short_filter_length,
        seed=args.seed + 100,
    )
    random_model.load_state_dict(random_sd, strict=False)

    # Cast to float32 and move to GPU
    random_model = random_model.float().to("cuda:0")
    random_model.eval()

    n_params = sum(p.numel() for p in random_model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  dtype: {next(random_model.parameters()).dtype}")

    # Verify the layer exists
    try:
        random_model.get_submodule(args.layer)
        print(f"  Layer '{args.layer}' found")
    except AttributeError as e:
        print(f"  ERROR: Layer '{args.layer}' not found: {e}")
        for name, _ in random_model.named_modules():
            if "mlp" in name and "blocks.28" in name:
                print(f"    Available: {name}")
        sys.exit(1)

    # Quick smoke test: 1 sequence
    print(f"\n  --- Smoke test (1 sequence) ---")
    test_input = "ATCGATCGATCGATCG"
    test_ids = torch.tensor(
        tokenizer.tokenize(test_input), dtype=torch.int
    ).unsqueeze(0).to("cuda:0")
    captured = {}
    def smoke_hook(_, __, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["emb"] = output.detach()
    layer = random_model.get_submodule(args.layer)
    handle = layer.register_forward_hook(smoke_hook)
    with torch.no_grad():
        random_model.forward(test_ids)
    handle.remove()
    smoke_emb = captured["emb"]
    nan_count = torch.isnan(smoke_emb).sum().item()
    inf_count = torch.isinf(smoke_emb).sum().item()
    print(f"  Smoke test: NaN={nan_count}, Inf={inf_count}, "
          f"mean={smoke_emb[~torch.isnan(smoke_emb)].mean().item():.4f}")
    if nan_count > 0 or inf_count > 0:
        print(f"  ABORT: Smoke test failed — random model produces NaN/Inf!")
        print(f"  Check log_poles initialization (must be negative)")
        # Print log_poles values for debugging
        for name, param in random_model.named_parameters():
            if 'log_poles' in name:
                print(f"    {name}: min={param.min().item():.4f}, "
                      f"max={param.max().item():.4f}, "
                      f"has_positive={bool((param > 0).any())}")
                break
        sys.exit(1)
    print(f"  Smoke test passed!")

    # ---------------------------------------------------------------
    # 3. Extract random embeddings
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Extract random embeddings (float32, Savanna-style init)")
    print(f"{'='*60}")

    random_emb = extract_with_hooks(
        random_model, tokenizer, sequences,
        args.layer, args.pooling, args.max_length
    )
    print(f"Random embeddings shape: {random_emb.shape}")

    # ---------------------------------------------------------------
    # 4. Validation
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 4: VALIDATION")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    # Check 1: No NaN
    nan_count = np.isnan(random_emb).sum()
    nan_rows = np.any(np.isnan(random_emb), axis=1).sum()
    if nan_count == 0:
        print(f"  PASS: No NaN values")
        passed += 1
    else:
        print(f"  FAIL: {nan_count} NaN values in {nan_rows}/{len(random_emb)} rows")
        failed += 1

    # Check 2: No Inf
    inf_count = np.isinf(random_emb).sum()
    if inf_count == 0:
        print(f"  PASS: No Inf values")
        passed += 1
    else:
        print(f"  FAIL: {inf_count} Inf values")
        failed += 1

    # Check 3: Embeddings differ across sequences
    all_same = True
    for i in range(1, len(random_emb)):
        if not np.allclose(random_emb[0], random_emb[i], atol=1e-5):
            all_same = False
            break
    if not all_same:
        print(f"  PASS: Random embeddings differ across sequences")
        passed += 1
    else:
        print(f"  FAIL: All random embeddings are identical (collapsed)")
        failed += 1

    # Check 4: Variance
    var = np.var(random_emb, axis=0).mean()
    if var > 1e-6:
        print(f"  PASS: Mean per-feature variance = {var:.6f}")
        passed += 1
    else:
        print(f"  FAIL: Near-zero variance = {var}")
        failed += 1

    # Check 5: Low correlation with pretrained
    if nan_count == 0:
        n = min(10000, pretrained_emb.size, random_emb.size)
        flat_r = random_emb.flatten()[:n].astype(np.float64)
        flat_p = pretrained_emb.flatten()[:n].astype(np.float64)
        corr = np.corrcoef(flat_r, flat_p)[0, 1]
        if abs(corr) < 0.3:
            print(f"  PASS: Low correlation with pretrained = {corr:.4f}")
            passed += 1
        else:
            print(f"  FAIL: High correlation with pretrained = {corr:.4f}")
            failed += 1
    else:
        print(f"  SKIP: Cannot compute correlation (NaN present)")

    # Check 6: Pairwise distances
    if nan_count == 0:
        dists = np.linalg.norm(
            random_emb[:min(10, len(random_emb))] - random_emb[0], axis=1
        )
        mean_dist = dists[1:].mean()
        if mean_dist > 1e-3:
            print(f"  PASS: Mean L2 distance between samples = {mean_dist:.4f}")
            passed += 1
        else:
            print(f"  FAIL: Samples too close, mean L2 = {mean_dist}")
            failed += 1
    else:
        print(f"  SKIP: Cannot compute distances (NaN present)")

    # Sample comparisons
    print(f"\n  Sample comparison (first 5 dims):")
    for i in range(min(3, len(sequences))):
        print(f"    Seq {i} pretrained: {pretrained_emb[i, :5]}")
        print(f"    Seq {i} random:     {random_emb[i, :5]}")
        print()

    # Summary
    print(f"{'='*60}")
    print(f"RESULT: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    print(f"Total time: {time.time() - start:.1f}s")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
