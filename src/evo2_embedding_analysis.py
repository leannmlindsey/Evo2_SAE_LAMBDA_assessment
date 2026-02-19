#!/usr/bin/env python3
"""
Embedding Analysis Script for Evo2

This script performs embedding analysis similar to GENERanno:
1. Extracts embeddings from Evo2 model for sequences in CSV files
2. Trains a linear probe (logistic regression) classifier
3. Calculates silhouette score to measure embedding quality
4. Creates PCA visualization showing class separation
5. Trains a simple 3-layer neural network classifier
6. Optionally compares against random baseline to measure "embedding power"

Usage:
    python evo2_embedding_analysis.py \
        --csv_dir /path/to/csv/data \
        --output_dir ./results/embedding_analysis \
        --model evo2_7b \
        --layer blocks.28.mlp.l3 \
        --include_random_baseline
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings and perform embedding analysis with Evo2"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Path to directory containing train.csv, dev.csv, test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/embedding_analysis",
        help="Directory to save results",
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
        help="Layer name for embedding extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "first", "last", "max"],
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--nn_epochs",
        type=int,
        default=100,
        help="Number of epochs for 3-layer NN training",
    )
    parser.add_argument(
        "--nn_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for 3-layer NN",
    )
    parser.add_argument(
        "--nn_lr",
        type=float,
        default=1e-3,
        help="Learning rate for 3-layer NN",
    )
    parser.add_argument(
        "--include_random_baseline",
        action="store_true",
        help="Include random embedding baseline to measure embedding power",
    )
    return parser.parse_args()


def load_csv_data(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test CSV files."""
    train_path = os.path.join(csv_dir, "train.csv")
    test_path = os.path.join(csv_dir, "test.csv")

    # Check for dev.csv or val.csv
    dev_path = os.path.join(csv_dir, "dev.csv")
    val_path = os.path.join(csv_dir, "val.csv")
    if os.path.exists(dev_path):
        validation_path = dev_path
    elif os.path.exists(val_path):
        validation_path = val_path
    else:
        raise FileNotFoundError(f"No validation file found in {csv_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


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
    labels: List[int],
    layer_name: str,
    batch_size: int,
    max_length: Optional[int],
    pooling: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from Evo2 model for given sequences.

    Returns:
        Tuple of (embeddings array, labels array)
    """
    all_embeddings = []
    all_labels = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        for seq, label in zip(batch_seqs, batch_labels):
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

                layer_embeddings = embeddings[layer_name]
                pooled = pool_embeddings(layer_embeddings, pooling)
                all_embeddings.append(pooled.cpu().float().numpy())
                all_labels.append(label)

    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)

    return embeddings_array, labels_array


def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, float], Dict]:
    """Train a linear probe (logistic regression) classifier."""
    print("\n" + "=" * 60)
    print("Training Linear Probe (Logistic Regression)")
    print("=" * 60)

    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver='lbfgs',
        n_jobs=-1,
    )
    clf.fit(train_scaled, train_labels)

    # Predict
    test_preds = clf.predict(test_scaled)
    test_probs = clf.predict_proba(test_scaled)[:, 1]

    # Calculate metrics
    metrics = {
        "linear_probe_accuracy": float(accuracy_score(test_labels, test_preds)),
        "linear_probe_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "linear_probe_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    try:
        metrics["linear_probe_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["linear_probe_auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["linear_probe_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["linear_probe_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"  Accuracy: {metrics['linear_probe_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['linear_probe_f1']:.4f}")
    print(f"  MCC: {metrics['linear_probe_mcc']:.4f}")
    print(f"  AUC: {metrics['linear_probe_auc']:.4f}")

    predictions = {
        "test_preds": test_preds,
        "test_probs": test_probs,
    }
    return metrics, predictions


def calculate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score for embeddings."""
    print("\n" + "=" * 60)
    print("Calculating Silhouette Score")
    print("=" * 60)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    score = silhouette_score(scaled_embeddings, labels)
    print(f"  Silhouette Score: {score:.4f}")
    print(f"  Interpretation: ", end="")
    if score > 0.5:
        print("Strong structure (embeddings well-separated by class)")
    elif score > 0.25:
        print("Reasonable structure")
    elif score > 0:
        print("Weak structure (some overlap between classes)")
    else:
        print("No apparent structure (classes highly overlapped)")

    return float(score)


def create_pca_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "PCA Visualization of Embeddings",
) -> Dict[str, float]:
    """Create PCA visualization of embeddings colored by class."""
    print("\n" + "=" * 60)
    print("Creating PCA Visualization")
    print("=" * 60)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(scaled_embeddings)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]*100:.2f}% of variance")
    print(f"  PC2 explains {explained_var[1]*100:.2f}% of variance")
    print(f"  Total: {sum(explained_var)*100:.2f}%")

    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e']
    class_names = ['Class 0', 'Class 1']

    for class_idx in [0, 1]:
        mask = labels == class_idx
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[class_idx],
            label=f'{class_names[class_idx]} (n={mask.sum()})',
            alpha=0.6,
            s=30,
        )

    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_path}")

    return {
        "pca_explained_variance_pc1": float(explained_var[0]),
        "pca_explained_variance_pc2": float(explained_var[1]),
        "pca_total_explained_variance": float(sum(explained_var)),
    }


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


def train_three_layer_nn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, float], nn.Module, StandardScaler, Dict]:
    """Train a 3-layer neural network classifier on embeddings."""
    print("\n" + "=" * 60)
    print("Training 3-Layer Neural Network")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize embeddings
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)
    test_scaled = scaler.transform(test_embeddings)

    # Create tensors
    train_X = torch.FloatTensor(train_scaled).to(device)
    train_y = torch.LongTensor(train_labels).to(device)
    val_X = torch.FloatTensor(val_scaled).to(device)
    val_y = torch.LongTensor(val_labels).to(device)
    test_X = torch.FloatTensor(test_scaled).to(device)
    test_y = torch.LongTensor(test_labels).to(device)

    # Create data loaders
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model
    input_dim = train_embeddings.shape[1]
    model = ThreeLayerNN(input_dim, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

    # Calculate metrics
    metrics = {
        "nn_accuracy": float(accuracy_score(test_labels, test_preds)),
        "nn_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "nn_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "nn_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "nn_mcc": float(matthews_corrcoef(test_labels, test_preds)),
    }

    try:
        metrics["nn_auc"] = float(roc_auc_score(test_labels, test_probs))
    except ValueError:
        metrics["nn_auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds, labels=[0, 1]).ravel()
    metrics["nn_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["nn_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"\n  Final Test Results:")
    print(f"  Accuracy: {metrics['nn_accuracy']:.4f}")
    print(f"  F1 Score: {metrics['nn_f1']:.4f}")
    print(f"  MCC: {metrics['nn_mcc']:.4f}")
    print(f"  AUC: {metrics['nn_auc']:.4f}")

    predictions = {
        "test_preds": test_preds,
        "test_probs": test_probs,
    }
    return metrics, model, scaler, predictions


def generate_random_embeddings(n_samples: int, hidden_dim: int, seed: int) -> np.ndarray:
    """Generate random baseline embeddings (Gaussian noise).

    This serves as a null baseline — if the pretrained model has learned meaningful
    structure, classifiers trained on its embeddings should significantly outperform
    classifiers trained on random embeddings of the same shape.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, hidden_dim)


def run_random_baseline(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    hidden_dim: int,
    seed: int,
    nn_hidden_dim: int,
    nn_epochs: int,
    nn_lr: float,
    device: torch.device,
    output_dir: str,
) -> Tuple[Dict[str, float], Dict]:
    """Run the full evaluation pipeline on random embeddings.

    Returns:
        Tuple of (metrics dict with 'random_' prefix, predictions dict)
    """
    print("\n" + "#" * 60)
    print("RANDOM BASELINE EVALUATION")
    print("#" * 60)
    print(f"Generating random embeddings: ({len(train_labels)}, {hidden_dim}), "
          f"({len(val_labels)}, {hidden_dim}), ({len(test_labels)}, {hidden_dim})")

    # Generate random embeddings matching pretrained shapes
    train_random = generate_random_embeddings(len(train_labels), hidden_dim, seed)
    val_random = generate_random_embeddings(len(val_labels), hidden_dim, seed + 1)
    test_random = generate_random_embeddings(len(test_labels), hidden_dim, seed + 2)

    results = {}

    # Linear probe on random embeddings
    lp_metrics, lp_preds = train_linear_probe(
        train_random, train_labels,
        test_random, test_labels,
        seed,
    )
    results.update(lp_metrics)

    # Silhouette score on random embeddings
    silhouette = calculate_silhouette(test_random, test_labels)
    results["silhouette_score"] = silhouette

    # PCA visualization of random embeddings
    pca_path = os.path.join(output_dir, "pca_visualization_random.png")
    pca_metrics = create_pca_visualization(
        test_random, test_labels,
        pca_path,
        title=f"Random Baseline Embeddings - PCA\n(Silhouette: {silhouette:.3f})",
    )
    results.update(pca_metrics)

    # 3-layer NN on random embeddings
    nn_metrics, _, _, nn_preds = train_three_layer_nn(
        train_random, train_labels,
        val_random, val_labels,
        test_random, test_labels,
        nn_hidden_dim, nn_epochs, nn_lr,
        seed, device,
    )
    results.update(nn_metrics)

    # Prefix all keys with "random_"
    random_results = {f"random_{k}": v for k, v in results.items()}

    # Save random predictions
    predictions = {
        "linear_probe_preds": lp_preds["test_preds"],
        "linear_probe_probs": lp_preds["test_probs"],
        "nn_preds": nn_preds["test_preds"],
        "nn_probs": nn_preds["test_probs"],
    }

    return random_results, predictions


def calculate_embedding_power(
    pretrained_metrics: Dict[str, float],
    random_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute embedding power = pretrained - random for each metric.

    Returns dict with 'embedding_power_' prefix.
    """
    power = {}
    # Map pretrained keys to random keys
    for key, value in pretrained_metrics.items():
        random_key = f"random_{key}"
        if random_key in random_metrics and isinstance(value, (int, float)):
            random_value = random_metrics[random_key]
            if isinstance(random_value, (int, float)):
                power[f"embedding_power_{key}"] = float(value - random_value)
    return power


def print_embedding_power_summary(
    pretrained_metrics: Dict[str, float],
    random_metrics: Dict[str, float],
    power_metrics: Dict[str, float],
) -> None:
    """Print a formatted comparison table of pretrained vs random metrics."""
    print("\n" + "=" * 60)
    print("EMBEDDING POWER SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<24} {'Pretrained':>10} {'Random':>10} {'Power':>10}")
    print("-" * 60)

    # Define which metrics to show in the summary table
    display_metrics = [
        ("LP Accuracy", "linear_probe_accuracy"),
        ("LP Precision", "linear_probe_precision"),
        ("LP Recall", "linear_probe_recall"),
        ("LP F1", "linear_probe_f1"),
        ("LP MCC", "linear_probe_mcc"),
        ("LP AUC", "linear_probe_auc"),
        ("LP Sensitivity", "linear_probe_sensitivity"),
        ("LP Specificity", "linear_probe_specificity"),
        ("NN Accuracy", "nn_accuracy"),
        ("NN Precision", "nn_precision"),
        ("NN Recall", "nn_recall"),
        ("NN F1", "nn_f1"),
        ("NN MCC", "nn_mcc"),
        ("NN AUC", "nn_auc"),
        ("NN Sensitivity", "nn_sensitivity"),
        ("NN Specificity", "nn_specificity"),
        ("Silhouette", "silhouette_score"),
    ]

    for display_name, key in display_metrics:
        pretrained_val = pretrained_metrics.get(key)
        random_val = random_metrics.get(f"random_{key}")
        power_val = power_metrics.get(f"embedding_power_{key}")
        if pretrained_val is not None and random_val is not None and power_val is not None:
            sign = "+" if power_val >= 0 else ""
            print(f"{display_name:<24} {pretrained_val:>10.4f} {random_val:>10.4f} {sign}{power_val:>9.4f}")

    print("=" * 60)


def main():
    """Main function to run embedding analysis."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Evo2 Embedding Analysis")
    print("=" * 60)

    start_time = time.time()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df, val_df, test_df = load_csv_data(args.csv_dir)

    # Check if embeddings already exist (backward compat: try both names)
    embeddings_path = os.path.join(args.output_dir, "embeddings_pretrained.npz")
    legacy_path = os.path.join(args.output_dir, "embeddings.npz")
    if os.path.exists(embeddings_path):
        load_path = embeddings_path
    elif os.path.exists(legacy_path):
        load_path = legacy_path
    else:
        load_path = None

    if load_path is not None:
        print(f"\nFound existing embeddings at: {load_path}")
        print("Loading embeddings from file (delete file to re-extract)...")
        loaded = np.load(load_path)
        train_embeddings = loaded["train_embeddings"]
        train_labels = loaded["train_labels"]
        val_embeddings = loaded["val_embeddings"]
        val_labels = loaded["val_labels"]
        test_embeddings = loaded["test_embeddings"]
        test_labels = loaded["test_labels"]
        print(f"Loaded embeddings - shape: {test_embeddings.shape}")
    else:
        # Load Evo2 model
        print(f"\nLoading Evo2 model: {args.model}")
        from evo2 import Evo2
        evo2_model = Evo2(args.model)
        print(f"  Model loaded")

        # Extract embeddings
        print(f"\nExtracting train embeddings...")
        train_embeddings, train_labels = extract_embeddings(
            evo2_model,
            train_df["sequence"].tolist(),
            train_df["label"].tolist(),
            args.layer,
            args.batch_size,
            args.max_length,
            args.pooling,
        )

        print(f"\nExtracting validation embeddings...")
        val_embeddings, val_labels = extract_embeddings(
            evo2_model,
            val_df["sequence"].tolist(),
            val_df["label"].tolist(),
            args.layer,
            args.batch_size,
            args.max_length,
            args.pooling,
        )

        print(f"\nExtracting test embeddings...")
        test_embeddings, test_labels = extract_embeddings(
            evo2_model,
            test_df["sequence"].tolist(),
            test_df["label"].tolist(),
            args.layer,
            args.batch_size,
            args.max_length,
            args.pooling,
        )

        print(f"\nEmbedding shape: {test_embeddings.shape}")

        # Save embeddings with new name
        np.savez(
            embeddings_path,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )
        print(f"\nSaved embeddings to: {embeddings_path}")

        # Free model memory
        del evo2_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run pretrained analyses
    pretrained_results = {}

    # 1. Train linear probe
    linear_metrics, linear_preds = train_linear_probe(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        args.seed,
    )
    pretrained_results.update(linear_metrics)

    # 2. Calculate silhouette score
    silhouette = calculate_silhouette(test_embeddings, test_labels)
    pretrained_results["silhouette_score"] = silhouette

    # 3. Create PCA visualization
    if args.include_random_baseline:
        pca_path = os.path.join(args.output_dir, "pca_visualization_pretrained.png")
    else:
        pca_path = os.path.join(args.output_dir, "pca_visualization.png")
    pca_metrics = create_pca_visualization(
        test_embeddings, test_labels,
        pca_path,
        title=f"Evo2 ({args.model}) Embeddings - PCA\n(Silhouette: {silhouette:.3f})",
    )
    pretrained_results.update(pca_metrics)

    # 4. Train 3-layer NN
    nn_metrics, nn_model, nn_scaler, nn_preds = train_three_layer_nn(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        test_embeddings, test_labels,
        args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
        args.seed, device,
    )
    pretrained_results.update(nn_metrics)

    # 5. Save pretrained test predictions to CSV
    predictions_df = pd.DataFrame({
        "sequence": test_df["sequence"].tolist(),
        "label": test_labels,
        "linear_probe_pred": linear_preds["test_preds"],
        "linear_probe_prob": linear_preds["test_probs"],
        "nn_pred": nn_preds["test_preds"],
        "nn_prob": nn_preds["test_probs"],
    })
    if args.include_random_baseline:
        predictions_path = os.path.join(args.output_dir, "test_predictions_pretrained.csv")
    else:
        predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nSaved test predictions to: {predictions_path}")

    # Save NN model
    nn_model_path = os.path.join(args.output_dir, "three_layer_nn.pt")
    torch.save({
        "model_state_dict": nn_model.state_dict(),
        "input_dim": test_embeddings.shape[1],
        "hidden_dim": args.nn_hidden_dim,
    }, nn_model_path)
    print(f"Saved 3-layer NN to: {nn_model_path}")

    # Build final results dict
    results = {}

    # Random baseline evaluation
    if args.include_random_baseline:
        # Prefix pretrained metrics
        for k, v in pretrained_results.items():
            results[f"pretrained_{k}"] = v

        # Run random baseline
        hidden_dim = test_embeddings.shape[1]
        random_results, random_preds = run_random_baseline(
            train_labels, val_labels, test_labels,
            hidden_dim, args.seed,
            args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
            device, args.output_dir,
        )
        results.update(random_results)

        # Save random predictions CSV
        random_pred_df = pd.DataFrame({
            "sequence": test_df["sequence"].tolist(),
            "label": test_labels,
            "linear_probe_pred": random_preds["linear_probe_preds"],
            "linear_probe_prob": random_preds["linear_probe_probs"],
            "nn_pred": random_preds["nn_preds"],
            "nn_prob": random_preds["nn_probs"],
        })
        random_pred_path = os.path.join(args.output_dir, "test_predictions_random.csv")
        random_pred_df.to_csv(random_pred_path, index=False)
        print(f"\nSaved random predictions to: {random_pred_path}")

        # Compute embedding power
        power_metrics = calculate_embedding_power(pretrained_results, random_results)
        results.update(power_metrics)

        # Print comparison summary
        print_embedding_power_summary(pretrained_results, random_results, power_metrics)
    else:
        results.update(pretrained_results)

    # Add metadata to results
    results["model"] = args.model
    results["layer"] = args.layer
    results["pooling"] = args.pooling
    results["embedding_dim"] = int(test_embeddings.shape[1])
    results["train_samples"] = len(train_labels)
    results["val_samples"] = len(val_labels)
    results["test_samples"] = len(test_labels)
    results["include_random_baseline"] = args.include_random_baseline

    # Save results
    results_path = os.path.join(args.output_dir, "embedding_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Embedding dimension: {test_embeddings.shape[1]}")

    if args.include_random_baseline:
        print(f"\nPretrained Linear Probe Results:")
        print(f"  Accuracy: {pretrained_results['linear_probe_accuracy']:.4f}")
        print(f"  MCC: {pretrained_results['linear_probe_mcc']:.4f}")
        print(f"  AUC: {pretrained_results['linear_probe_auc']:.4f}")
        print(f"\nPretrained 3-Layer NN Results:")
        print(f"  Accuracy: {pretrained_results['nn_accuracy']:.4f}")
        print(f"  MCC: {pretrained_results['nn_mcc']:.4f}")
        print(f"  AUC: {pretrained_results['nn_auc']:.4f}")
        print(f"\nPretrained Embedding Quality:")
        print(f"  Silhouette Score: {pretrained_results['silhouette_score']:.4f}")
        print(f"  PCA Variance Explained: {pretrained_results['pca_total_explained_variance']*100:.1f}%")
    else:
        print(f"\nLinear Probe Results:")
        print(f"  Accuracy: {pretrained_results['linear_probe_accuracy']:.4f}")
        print(f"  MCC: {pretrained_results['linear_probe_mcc']:.4f}")
        print(f"  AUC: {pretrained_results['linear_probe_auc']:.4f}")
        print(f"\n3-Layer NN Results:")
        print(f"  Accuracy: {pretrained_results['nn_accuracy']:.4f}")
        print(f"  MCC: {pretrained_results['nn_mcc']:.4f}")
        print(f"  AUC: {pretrained_results['nn_auc']:.4f}")
        print(f"\nEmbedding Quality:")
        print(f"  Silhouette Score: {pretrained_results['silhouette_score']:.4f}")
        print(f"  PCA Variance Explained: {pretrained_results['pca_total_explained_variance']*100:.1f}%")
    print("=" * 60)

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
