#!/usr/bin/env python3
"""
Inference Script for Evo2

This script performs inference using Evo2 embeddings with a trained classifier.
It can use either:
1. A pre-trained 3-layer NN classifier (from evo2_embedding_analysis.py)
2. A linear probe classifier (trained on the fly)

Usage:
    # Using pre-trained NN classifier
    python evo2_inference.py \
        --input_csv /path/to/test.csv \
        --classifier_path ./results/three_layer_nn.pt \
        --embeddings_path ./results/embeddings.npz \
        --output_csv /path/to/predictions.csv

    # Train new linear probe on the fly
    python evo2_inference.py \
        --input_csv /path/to/test.csv \
        --train_embeddings_path ./results/train_embeddings.npz \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with Evo2 embeddings"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="evo2_7b",
        choices=["evo2_7b", "evo2_40b"],
        help="Evo2 model to use for embedding extraction",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="blocks.28.mlp.l3",
        help="Layer name for embedding extraction",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "first", "last", "max"],
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    # Classifier options
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="Path to pre-trained 3-layer NN classifier (.pt file)",
    )
    parser.add_argument(
        "--train_embeddings_path",
        type=str,
        default=None,
        help="Path to training embeddings for linear probe (if no classifier_path)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save extracted embeddings to file",
    )
    return parser.parse_args()


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def pool_embeddings(embeddings: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool sequence embeddings to a single vector."""
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
    max_length: Optional[int],
    pooling: str,
) -> np.ndarray:
    """Extract embeddings from Evo2 model for given sequences."""
    all_embeddings = []

    for seq in tqdm(sequences, desc="Extracting embeddings"):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]

        input_ids = torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        ).unsqueeze(0).to('cuda:0')

        with torch.no_grad():
            outputs, embeddings = model(
                input_ids,
                return_embeddings=True,
                layer_names=[layer_name]
            )

            layer_embeddings = embeddings[layer_name]
            pooled = pool_embeddings(layer_embeddings, pooling)
            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Evo2 Inference")
    print("=" * 60)

    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Extract embeddings
    print(f"\nLoading Evo2 model: {args.model}")
    from evo2 import Evo2
    evo2_model = Evo2(args.model)

    print(f"Extracting embeddings from layer: {args.layer}")
    sequences = df["sequence"].tolist()
    embeddings = extract_embeddings(
        evo2_model,
        sequences,
        args.layer,
        args.max_length,
        args.pooling,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # Save embeddings if requested
    if args.save_embeddings:
        emb_path = args.input_csv.replace(".csv", "_embeddings.npz")
        np.savez(emb_path, embeddings=embeddings)
        print(f"  Saved embeddings to: {emb_path}")

    # Free Evo2 model memory
    del evo2_model
    torch.cuda.empty_cache()

    # Load or train classifier
    if args.classifier_path:
        # Load pre-trained 3-layer NN
        print(f"\nLoading classifier from: {args.classifier_path}")
        checkpoint = torch.load(args.classifier_path, map_location=device)

        input_dim = checkpoint.get("input_dim", embeddings.shape[1])
        hidden_dim = checkpoint.get("hidden_dim", 256)

        classifier = ThreeLayerNN(input_dim, hidden_dim).to(device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval()

        # Standardize embeddings (need to use same scaler as training)
        # If scaler was saved, load it; otherwise use StandardScaler
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Predict
        with torch.no_grad():
            X = torch.FloatTensor(embeddings_scaled).to(device)
            outputs = classifier(X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= args.threshold).astype(int)

    elif args.train_embeddings_path:
        # Train linear probe on the fly
        print(f"\nTraining linear probe using: {args.train_embeddings_path}")
        train_data = np.load(args.train_embeddings_path)
        train_embeddings = train_data["train_embeddings"]
        train_labels = train_data["train_labels"]

        # Standardize
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_embeddings)
        embeddings_scaled = scaler.transform(embeddings)

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_scaled, train_labels)

        # Predict
        probs = clf.predict_proba(embeddings_scaled)[:, 1]
        preds = (probs >= args.threshold).astype(int)

    else:
        raise ValueError("Must provide either --classifier_path or --train_embeddings_path")

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_1"] = probs
    output_df["pred_label"] = preds

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds, probs)

        # Add metadata
        metrics["model"] = args.model
        metrics["layer"] = args.layer
        metrics["pooling"] = args.pooling
        metrics["threshold"] = args.threshold
        metrics["num_samples"] = len(df)
        metrics["classifier_path"] = args.classifier_path

        # Save metrics
        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Print metrics
        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

    elif has_labels:
        labels = df["label"].values
        acc = accuracy_score(labels, preds)
        print(f"\nAccuracy: {acc:.4f}")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
