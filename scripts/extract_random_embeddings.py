#!/usr/bin/env python3
"""
Extract random model embeddings for all splits (train, val, test).

Creates a random StripedHyena with Savanna-style initialization and flash_attn
disabled (float32). Extracts embeddings for all sequences in train.csv, dev/val.csv,
and test.csv, then saves to embeddings_random_model.npz in the same format as the
main embedding analysis script expects.

This script does NOT require pretrained embeddings — it can run independently
while pretrained extraction is still in progress on another GPU.

Output: {output_dir}/embeddings_random_model.npz containing:
    - train_embeddings, val_embeddings, test_embeddings

Usage:
    python scripts/extract_random_embeddings.py \
        --csv_dir /path/to/csv/data \
        --output_dir ./results/embedding_analysis/8k \
        --model evo2_7b \
        --layer blocks.28.mlp.l3
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract random model embeddings for all splits"
    )
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing train.csv, dev.csv/val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings_random_model.npz")
    parser.add_argument("--model", type=str, default="evo2_7b")
    parser.add_argument("--layer", type=str, default="blocks.28.mlp.l3")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "first", "last", "max"])
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of sequences per batch (for progress tracking)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_savanna_style_init(state_dict, hidden_size=4096, num_layers=32,
                             short_filter_length=3, seed=42):
    """Apply Savanna-style initialization to a Vortex StripedHyena state_dict."""
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


def extract_with_hooks(model, tokenizer, sequences, layer_name, pooling, max_length,
                       batch_size=16):
    """Extract embeddings using manual hook registration."""
    all_embeddings = []
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        for seq in batch_seqs:
            if max_length is not None and len(seq) > max_length:
                seq = seq[:max_length]
            input_ids = torch.tensor(
                tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to(device)

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
    # 1. Load CSV data
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Load CSV data")
    print(f"{'='*60}")

    train_path = os.path.join(args.csv_dir, "train.csv")
    test_path = os.path.join(args.csv_dir, "test.csv")

    dev_path = os.path.join(args.csv_dir, "dev.csv")
    val_path = os.path.join(args.csv_dir, "val.csv")
    if os.path.exists(dev_path):
        validation_path = dev_path
    elif os.path.exists(val_path):
        validation_path = val_path
    else:
        raise FileNotFoundError(f"No dev.csv or val.csv found in {args.csv_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print(f"  Train: {len(train_df)} sequences")
    print(f"  Val:   {len(val_df)} sequences")
    print(f"  Test:  {len(test_df)} sequences")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} sequences")

    # ---------------------------------------------------------------
    # 2. Create random model
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
    print("Loading pretrained Evo2 (for tokenizer only)...")
    pretrained_model = Evo2(args.model)
    tokenizer = pretrained_model.tokenizer

    del pretrained_model
    torch.cuda.empty_cache()
    print("Freed pretrained model memory")

    # Load config (keep flash_attn enabled + bfloat16 for speed)
    config_name = args.model
    if config_name not in CONFIG_MAP:
        config_name = f"{args.model}_base"
    config_path = CONFIG_MAP[config_name]
    print(f"Config: {config_name} -> {config_path}")

    cfg = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    cfg = dotdict(cfg)

    hidden_size = cfg.get("hidden_size", 4096)
    num_layers = cfg.get("num_layers", 32)
    short_filter_length = cfg.get("short_filter_length", 3)

    # Create random model with Savanna-style init
    torch.manual_seed(args.seed + 100)
    print("Creating StripedHyena with random init...")
    random_model = StripedHyena(cfg)

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

    random_model.to_bfloat16_except_pr_lc()
    random_model = random_model.to("cuda:0")
    random_model.eval()

    n_params = sum(p.numel() for p in random_model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  dtype: {next(random_model.parameters()).dtype}")

    # Verify layer exists
    try:
        random_model.get_submodule(args.layer)
        print(f"  Layer '{args.layer}' found")
    except AttributeError as e:
        print(f"  ERROR: Layer '{args.layer}' not found: {e}")
        sys.exit(1)

    # Smoke test
    print(f"\n  --- Smoke test ---")
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
    print(f"  NaN={nan_count}, Inf={inf_count}, "
          f"mean={smoke_emb[~torch.isnan(smoke_emb)].mean().item():.4f}")
    if nan_count > 0 or inf_count > 0:
        print(f"  ABORT: Smoke test failed!")
        sys.exit(1)
    print(f"  Smoke test passed!")

    # ---------------------------------------------------------------
    # 3. Extract embeddings for all splits
    # ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{'='*60}")
        print(f"STEP 3: Extract {split_name} random embeddings ({len(split_df)} sequences)")
        print(f"{'='*60}")

        sequences = split_df["sequence"].tolist()
        t0 = time.time()
        emb = extract_with_hooks(
            random_model, tokenizer, sequences,
            args.layer, args.pooling, args.max_length,
            batch_size=args.batch_size
        )
        elapsed = time.time() - t0

        # Validate
        nan_count = np.isnan(emb).sum()
        inf_count = np.isinf(emb).sum()
        var = np.var(emb, axis=0).mean()
        print(f"  Shape: {emb.shape}")
        print(f"  NaN: {nan_count}, Inf: {inf_count}")
        print(f"  Mean per-feature variance: {var:.6f}")
        print(f"  Time: {elapsed:.1f}s ({len(split_df)/elapsed:.1f} seq/s)")

        if nan_count > 0:
            print(f"  WARNING: {split_name} has NaN values!")

        if split_name == "train":
            train_emb = emb
        elif split_name == "val":
            val_emb = emb
        else:
            test_emb = emb

    # ---------------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 4: Save embeddings")
    print(f"{'='*60}")

    output_path = os.path.join(args.output_dir, "embeddings_random_model.npz")
    np.savez(
        output_path,
        train_embeddings=train_emb,
        val_embeddings=val_emb,
        test_embeddings=test_emb,
    )
    print(f"Saved to: {output_path}")
    print(f"  train: {train_emb.shape}")
    print(f"  val:   {val_emb.shape}")
    print(f"  test:  {test_emb.shape}")

    total_time = time.time() - start
    total_seqs = len(train_df) + len(val_df) + len(test_df)
    print(f"\nTotal time: {total_time:.1f}s ({total_seqs/total_time:.1f} seq/s)")


if __name__ == "__main__":
    main()
