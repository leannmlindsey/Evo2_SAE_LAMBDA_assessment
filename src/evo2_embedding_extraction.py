#!/usr/bin/env python3
"""
Embedding Extraction Script for Evo2

This script extracts embeddings from Evo2 models using the official return_embeddings API.
It processes sequences from CSV files and saves embeddings as numpy arrays.

Usage:
    python evo2_embedding_extraction.py \
        --input_csv /path/to/sequences.csv \
        --output_dir ./embeddings \
        --model evo2_7b \
        --layer blocks.28.mlp.l3

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional)

Output:
    - embeddings.npz: Numpy file containing embeddings and labels
"""

import argparse
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from Evo2 model"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="evo2_7b",
        choices=["evo2_7b", "evo2_40b"],
        help="Evo2 model to use",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="blocks.28.mlp.l3",
        help="Layer name for embedding extraction (e.g., 'blocks.28.mlp.l3')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1 for long sequences)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (truncate longer sequences)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "first", "last", "max"],
        help="Pooling strategy for sequence embeddings",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="embeddings",
        help="Name for output file (without extension)",
    )
    return parser.parse_args()


def pool_embeddings(
    embeddings: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Pool sequence embeddings to a single vector.

    Args:
        embeddings: Tensor of shape (batch, seq_len, hidden_dim)
        pooling: Pooling strategy ('mean', 'first', 'last', 'max')

    Returns:
        Pooled tensor of shape (batch, hidden_dim)
    """
    if pooling == "mean":
        return embeddings.mean(dim=1)
    elif pooling == "first":
        return embeddings[:, 0, :]
    elif pooling == "last":
        return embeddings[:, -1, :]
    elif pooling == "max":
        return embeddings.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")


def extract_embeddings(
    model,
    sequences: List[str],
    layer_name: str,
    batch_size: int,
    max_length: Optional[int],
    pooling: str,
) -> np.ndarray:
    """
    Extract embeddings from Evo2 model for given sequences.

    Args:
        model: The Evo2 model
        sequences: List of DNA sequences
        layer_name: Layer name for embedding extraction
        batch_size: Batch size for processing
        max_length: Maximum sequence length (optional)
        pooling: Pooling strategy

    Returns:
        Numpy array of embeddings, shape (n_sequences, hidden_dim)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]

        for seq in batch_seqs:
            # Truncate if needed
            if max_length is not None and len(seq) > max_length:
                seq = seq[:max_length]

            # Tokenize
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')

            # Forward pass with embedding extraction
            with torch.no_grad():
                outputs, embeddings = model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[layer_name]
                )

                # Get embeddings for the specified layer
                layer_embeddings = embeddings[layer_name]  # (batch, seq_len, hidden_dim)

                # Pool to single vector
                pooled = pool_embeddings(layer_embeddings, pooling)

                all_embeddings.append(pooled.cpu().float().numpy())

    return np.vstack(all_embeddings)


def main():
    """Main function to extract embeddings."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Evo2 Embedding Extraction")
    print("=" * 60)

    start_time = time.time()

    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Print sequence length stats
    seq_lengths = df["sequence"].str.len()
    print(f"  Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.0f}")

    # Load Evo2 model
    print(f"\nLoading Evo2 model: {args.model}")
    from evo2 import Evo2
    model = Evo2(args.model)
    print(f"  Model loaded on: {next(model.model.parameters()).device}")

    # Extract embeddings
    print(f"\nExtracting embeddings from layer: {args.layer}")
    print(f"  Pooling: {args.pooling}")
    if args.max_length:
        print(f"  Max length: {args.max_length}")

    sequences = df["sequence"].tolist()
    embeddings = extract_embeddings(
        model,
        sequences,
        args.layer,
        args.batch_size,
        args.max_length,
        args.pooling,
    )

    print(f"\nEmbedding shape: {embeddings.shape}")

    # Prepare output
    output_data = {
        "embeddings": embeddings,
    }

    if has_labels:
        output_data["labels"] = df["label"].values

    # Save embeddings
    output_path = os.path.join(args.output_dir, f"{args.output_name}.npz")
    np.savez(output_path, **output_data)
    print(f"\nSaved embeddings to: {output_path}")

    # Also save metadata
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "input_csv": args.input_csv,
        "num_samples": len(df),
        "embedding_dim": embeddings.shape[1],
    }

    import json
    metadata_path = os.path.join(args.output_dir, f"{args.output_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.2f} sequences/second")


if __name__ == "__main__":
    main()
