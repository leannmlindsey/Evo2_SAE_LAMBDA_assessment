#!/usr/bin/env python3
"""
Quick test: load cached pretrained embeddings, extract 32 random embeddings, compare.

Verifies that:
1. Random embeddings contain no NaN/Inf
2. Random embeddings differ across sequences (no collapse)
3. Random embeddings are not close to pretrained embeddings

Usage:
    python scripts/test_random_embeddings.py \
        --csv_dir /path/to/csv/data \
        --pretrained_embeddings /path/to/embeddings_pretrained.npz \
        --model evo2_7b \
        --layer blocks.28.mlp.l3
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test random vs pretrained embeddings")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Path to CSV dir (need train.csv for sequences)")
    parser.add_argument("--pretrained_embeddings", type=str, required=True,
                        help="Path to cached pretrained embeddings .npz file")
    parser.add_argument("--model", type=str, default="evo2_7b")
    parser.add_argument("--layer", type=str, default="blocks.28.mlp.l3")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_batch(model, sequences, layer_name, pooling, max_length):
    """Extract embeddings for a small batch of sequences."""
    all_embeddings = []
    for seq in tqdm(sequences, desc="Extracting random embeddings"):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq), dtype=torch.int
        ).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            _, embeddings = model(
                input_ids, return_embeddings=True, layer_names=[layer_name]
            )
            layer_emb = embeddings[layer_name]
            if pooling == "mean":
                pooled = layer_emb.mean(dim=1)
            elif pooling == "first":
                pooled = layer_emb[:, 0, :]
            elif pooling == "last":
                pooled = layer_emb[:, -1, :]
            elif pooling == "max":
                pooled = layer_emb.max(dim=1)[0]
            all_embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(all_embeddings)


def main():
    args = parse_args()
    start = time.time()

    # ---------------------------------------------------------------
    # 1. Load cached pretrained embeddings (no model needed)
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Load cached pretrained embeddings")
    print(f"{'='*60}")
    loaded = np.load(args.pretrained_embeddings)
    pretrained_emb = loaded["train_embeddings"][: args.n_samples]
    print(f"Loaded {pretrained_emb.shape[0]} pretrained embeddings "
          f"from {args.pretrained_embeddings}")
    print(f"  Shape: {pretrained_emb.shape}")
    print(f"  NaN: {np.isnan(pretrained_emb).any()}, "
          f"Inf: {np.isinf(pretrained_emb).any()}")
    print(f"  Sample [0][:5]: {pretrained_emb[0, :5]}")
    print(f"  Sample [1][:5]: {pretrained_emb[1, :5]}")

    # Load matching sequences from train.csv
    train_path = f"{args.csv_dir}/train.csv"
    df = pd.read_csv(train_path)
    sequences = df["sequence"].tolist()[: args.n_samples]
    print(f"\nUsing first {len(sequences)} sequences from {train_path}")

    # ---------------------------------------------------------------
    # 2. Load model and replace weights with random initialization
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Load model + replace with random weights")
    print(f"{'='*60}")

    from evo2 import Evo2
    print(f"Loading Evo2 model: {args.model}")
    model = Evo2(args.model)
    print(f"Pretrained model dtype: {next(model.model.parameters()).dtype}")

    import yaml
    import pkgutil
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict

    torch.manual_seed(args.seed + 100)

    config_name = args.model
    if config_name not in CONFIG_MAP:
        config_name = f"{args.model}_base"
    config_path = CONFIG_MAP[config_name]
    print(f"Config: {config_name} -> {config_path}")

    cfg = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    cfg = dotdict(cfg)

    # Hybrid approach: randomize PARAMETERS only, keep BUFFERS at pretrained values.
    # Buffers contain structural values (Hyena filter time constants, rotary freqs,
    # etc.) that cause NaN if randomized. Parameters are the learned weights.
    # Using load_state_dict ensures TP-safe replacement (unlike in-place modification).

    print("Creating temp StripedHyena for random parameter values...")
    temp_model = StripedHyena(cfg)
    temp_sd = {k: v.cpu() for k, v in temp_model.state_dict().items()}
    del temp_model
    torch.cuda.empty_cache()

    # Identify which state_dict keys are parameters vs buffers
    param_names = set(name for name, _ in model.model.named_parameters())
    buffer_names = set(name for name, _ in model.model.named_buffers())
    pretrained_sd = {k: v.cpu() for k, v in model.model.state_dict().items()}

    print(f"  State dict: {len(pretrained_sd)} keys")
    print(f"  Parameters: {len(param_names)}")
    print(f"  Buffers: {len(buffer_names)}")

    # Build hybrid state dict: random params + pretrained buffers
    hybrid_sd = {}
    n_randomized = 0
    n_kept = 0
    for k in pretrained_sd:
        if k in param_names and k in temp_sd:
            hybrid_sd[k] = temp_sd[k]  # random parameter from temp model
            n_randomized += 1
        else:
            hybrid_sd[k] = pretrained_sd[k]  # keep pretrained buffer
            n_kept += 1

    print(f"  Randomized {n_randomized} parameters, kept {n_kept} buffers")

    model.model.load_state_dict(hybrid_sd, strict=True)
    print("Loaded hybrid state_dict into backbone")
    print(f"Model dtype: {next(model.model.parameters()).dtype}")

    # ---------------------------------------------------------------
    # 3. Extract random embeddings for the same sequences
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Extract random embeddings (bfloat16)")
    print(f"{'='*60}")

    random_emb = extract_batch(
        model, sequences, args.layer, args.pooling, args.max_length
    )
    print(f"Random embeddings shape: {random_emb.shape}")

    # ---------------------------------------------------------------
    # 4. Run all checks
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

    # Check 3: Embeddings differ across sequences (no collapse)
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

    # Check 4: Random embeddings have variance
    var = np.var(random_emb, axis=0).mean()
    if var > 1e-6:
        print(f"  PASS: Mean per-feature variance = {var:.6f}")
        passed += 1
    else:
        print(f"  FAIL: Near-zero variance = {var:.10f}")
        failed += 1

    # Check 5: Random != pretrained (low correlation)
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

    # Check 6: Pairwise distance between random embeddings is non-trivial
    if nan_count == 0:
        dists = np.linalg.norm(
            random_emb[:min(10, len(random_emb))] - random_emb[0], axis=1
        )
        mean_dist = dists[1:].mean()
        if mean_dist > 1e-3:
            print(f"  PASS: Mean L2 distance between samples = {mean_dist:.4f}")
            passed += 1
        else:
            print(f"  FAIL: Samples too close, mean L2 = {mean_dist:.10f}")
            failed += 1
    else:
        print(f"  SKIP: Cannot compute distances (NaN present)")

    # Print sample comparisons
    print(f"\n  Sample comparison (first 5 dims):")
    for i in range(min(3, len(sequences))):
        print(f"    Seq {i} pretrained: {pretrained_emb[i, :5]}")
        print(f"    Seq {i} random:     {random_emb[i, :5]}")
        print()

    # Summary
    print(f"{'='*60}")
    print(f"RESULT: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
