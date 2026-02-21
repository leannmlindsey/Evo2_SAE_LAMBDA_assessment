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
        "--pretrained_embeddings",
        type=str,
        default=None,
        help="Path to pre-extracted pretrained embeddings (.npz file with "
             "train_embeddings, train_labels, val_embeddings, val_labels, "
             "test_embeddings, test_labels). Skips model loading and embedding "
             "extraction when provided.",
    )
    parser.add_argument(
        "--include_random_baseline",
        action="store_true",
        help="Include random embedding baseline (randomly-initialized model) to measure "
             "the value of pretraining. Uses the same architecture and tokenizer but with "
             "random weights.",
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


def replace_with_random_weights(evo2_model, model_name: str, seed: int = 0) -> None:
    """Replace pretrained weights with random initialization via load_state_dict.

    Creates a temporary StripedHyena from config (which gets PyTorch's correct
    default initialization for every layer type), extracts its state_dict
    (parameters AND buffers), then loads it into the existing pretrained backbone.

    Using load_state_dict preserves:
    - Vortex multi-GPU device placement (copy_ keeps tensors on their devices)
    - dtype (float16/bfloat16 is preserved, source is cast automatically)
    - The Evo2 wrapper's internal references to self.model

    The old named_parameters() approach failed because it missed buffers (e.g.
    Hyena convolution filters, SSM recurrence matrices, positional state) which
    are included in state_dict but not in named_parameters.

    Args:
        evo2_model: Evo2 model wrapper object
        model_name: Model config name (e.g. 'evo2_7b' or 'evo2_7b_base')
        seed: Random seed for reproducible initialization
    """
    import yaml
    import pkgutil
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict

    torch.manual_seed(seed)

    # Resolve config name: CONFIG_MAP may use different key conventions
    config_name = model_name
    if config_name not in CONFIG_MAP:
        config_name = f"{model_name}_base"
    if config_name not in CONFIG_MAP:
        raise ValueError(
            f"Config not found for '{model_name}'. "
            f"Available configs: {list(CONFIG_MAP.keys())}"
        )

    config_path = CONFIG_MAP[config_name]
    print(f"  Loading config from: {config_name} -> {config_path}")

    cfg = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    cfg = dotdict(cfg)

    # Step 1: Create a temporary StripedHyena to get properly-initialized random
    # weights. PyTorch's default init handles each layer type correctly
    # (Xavier for linear, ones for norm scales, etc.)
    # Note: Vortex will place it on GPU (CUDA is already initialized so we can't
    # force CPU via env vars). Both models briefly coexist in GPU memory.
    print(f"  Creating temporary StripedHyena for random state_dict (seed={seed})...")
    temp_model = StripedHyena(cfg)

    # Step 2: Extract the random state_dict (parameters + buffers) to CPU
    random_sd = {k: v.cpu() for k, v in temp_model.state_dict().items()}

    n_params = sum(p.numel() for p in temp_model.parameters())
    n_buffers = sum(b.numel() for b in temp_model.buffers())
    n_sd_keys = len(random_sd)
    print(f"  Random state_dict: {n_sd_keys} keys, "
          f"{n_params:,} parameter elements, {n_buffers:,} buffer elements")

    # Step 3: Free the temporary model
    del temp_model

    # Step 4: Verify key compatibility and snapshot pretrained values for comparison
    pretrained_sd = evo2_model.model.state_dict()
    pretrained_sd_keys = set(pretrained_sd.keys())
    random_sd_keys = set(random_sd.keys())
    missing = pretrained_sd_keys - random_sd_keys
    unexpected = random_sd_keys - pretrained_sd_keys
    if missing:
        print(f"  WARNING: {len(missing)} keys in pretrained model missing from random "
              f"state_dict (will keep pretrained values): {list(missing)[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in random state_dict "
              f"(will be ignored): {list(unexpected)[:5]}...")

    # Save a few pretrained values for comparison after loading
    check_keys = [k for k in list(random_sd.keys())[:3] if k in pretrained_sd_keys]
    pretrained_snapshots = {k: pretrained_sd[k].cpu().float().clone() for k in check_keys}
    pretrained_dtype = pretrained_sd[check_keys[0]].dtype if check_keys else None
    print(f"  Pretrained model dtype: {pretrained_dtype}")

    del pretrained_sd  # free memory

    # Step 5: Load random weights into the existing pretrained backbone.
    # load_state_dict uses .copy_() which preserves device placement and dtype.
    evo2_model.model.load_state_dict(random_sd, strict=(not missing and not unexpected))
    print(f"  Loaded random state_dict into pretrained backbone")

    # Step 6: Diagnostic sanity check
    new_sd = evo2_model.model.state_dict()
    print(f"\n  === SANITY CHECK ===")
    for key in check_keys:
        loaded_vals = new_sd[key].cpu().float()
        random_vals = random_sd[key].float()
        pretrained_vals = pretrained_snapshots[key]

        # Check: are loaded values close to the random values we intended?
        matches_random = torch.allclose(loaded_vals, random_vals, atol=1e-2)
        # Check: are loaded values still the same as pretrained (i.e. load failed)?
        matches_pretrained = torch.allclose(loaded_vals, pretrained_vals, atol=1e-5)
        # Correlation with random target (should be ~1.0 if load worked)
        flat_loaded = loaded_vals.flatten()[:1000].double()
        flat_random = random_vals.flatten()[:1000].double()
        flat_pretrained = pretrained_vals.flatten()[:1000].double()
        corr_random = torch.corrcoef(torch.stack([flat_loaded, flat_random]))[0, 1].item()
        corr_pretrained = torch.corrcoef(torch.stack([flat_loaded, flat_pretrained]))[0, 1].item()

        print(f"  Key: '{key}' (shape={list(loaded_vals.shape)}, dtype={new_sd[key].dtype})")
        print(f"    Matches random target (atol=1e-2): {matches_random}")
        print(f"    Still matches pretrained (atol=1e-5): {matches_pretrained}")
        print(f"    Correlation with random target: {corr_random:.6f}")
        print(f"    Correlation with pretrained: {corr_pretrained:.6f}")
        print(f"    Loaded sample:     {loaded_vals.flatten()[:5].tolist()}")
        print(f"    Random target:     {random_vals.flatten()[:5].tolist()}")
        print(f"    Pretrained before: {pretrained_vals.flatten()[:5].tolist()}")

        if matches_pretrained and not matches_random:
            print(f"    FAILURE: load_state_dict did NOT change this tensor!")
        elif matches_random:
            print(f"    OK: weights successfully replaced")
        else:
            print(f"    LIKELY OK: values changed, small differences from dtype "
                  f"conversion ({pretrained_dtype} roundtrip)")
    print(f"  === END SANITY CHECK ===")


def run_random_baseline(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    nn_hidden_dim: int,
    nn_epochs: int,
    nn_lr: float,
    device: torch.device,
    output_dir: str,
    random_model_embeddings: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict]:
    """Run the full evaluation pipeline on embeddings from a randomly-initialized model.

    The random model baseline uses the same architecture and tokenizer as the
    pretrained model, but with randomized weights. This isolates the contribution
    of pretraining from the architecture's inductive biases.

    Args:
        random_model_embeddings: Pre-extracted embeddings from a randomly-initialized
            model. Required keys: 'train', 'val', 'test'

    Returns:
        Tuple of (metrics dict with 'random_' prefix, predictions dict)
    """
    print("\n" + "#" * 60)
    print("RANDOM MODEL BASELINE EVALUATION")
    print("#" * 60)
    print("Using embeddings from randomly-initialized model")
    print("(same architecture + tokenizer, random weights)")
    train_random = random_model_embeddings['train']
    val_random = random_model_embeddings['val']
    test_random = random_model_embeddings['test']
    print(f"  Shapes: train={train_random.shape}, val={val_random.shape}, test={test_random.shape}")

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

    # Check if embeddings already exist
    # Priority: --pretrained_embeddings flag > output_dir cache > legacy path
    if args.pretrained_embeddings:
        if not os.path.exists(args.pretrained_embeddings):
            raise FileNotFoundError(f"Pretrained embeddings not found: {args.pretrained_embeddings}")
        load_path = args.pretrained_embeddings
    else:
        embeddings_path = os.path.join(args.output_dir, "embeddings_pretrained.npz")
        legacy_path = os.path.join(args.output_dir, "embeddings.npz")
        if os.path.exists(embeddings_path):
            load_path = embeddings_path
        elif os.path.exists(legacy_path):
            load_path = legacy_path
        else:
            load_path = None

    # Check if random model embeddings are cached
    random_model_path = os.path.join(args.output_dir, "embeddings_random_model.npz")
    need_random_model = args.include_random_baseline
    random_model_cached = os.path.exists(random_model_path) if need_random_model else True
    random_model_embeddings = None

    # Determine if we need to load the model
    pretrained_cached = load_path is not None
    need_model = not pretrained_cached or (need_random_model and not random_model_cached)

    if pretrained_cached:
        print(f"\nFound existing pretrained embeddings at: {load_path}")
        print("Loading embeddings from file (delete file to re-extract)...")
        loaded = np.load(load_path)
        train_embeddings = loaded["train_embeddings"]
        train_labels = loaded["train_labels"]
        val_embeddings = loaded["val_embeddings"]
        val_labels = loaded["val_labels"]
        test_embeddings = loaded["test_embeddings"]
        test_labels = loaded["test_labels"]
        print(f"Loaded embeddings - shape: {test_embeddings.shape}")

    if need_random_model and random_model_cached:
        print(f"\nFound existing random model embeddings at: {random_model_path}")
        print("Loading random model embeddings from file (delete file to re-extract)...")
        loaded_random = np.load(random_model_path)
        random_model_embeddings = {
            'train': loaded_random["train_embeddings"],
            'val': loaded_random["val_embeddings"],
            'test': loaded_random["test_embeddings"],
        }
        print(f"Loaded random model embeddings - shape: {random_model_embeddings['test'].shape}")

    if need_model:
        # Load Evo2 model
        print(f"\nLoading Evo2 model: {args.model}")
        from evo2 import Evo2
        evo2_model = Evo2(args.model)
        print(f"  Model loaded")

        # Extract pretrained embeddings if not cached
        if not pretrained_cached:
            print(f"\nExtracting pretrained train embeddings...")
            train_embeddings, train_labels = extract_embeddings(
                evo2_model,
                train_df["sequence"].tolist(),
                train_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"\nExtracting pretrained validation embeddings...")
            val_embeddings, val_labels = extract_embeddings(
                evo2_model,
                val_df["sequence"].tolist(),
                val_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"\nExtracting pretrained test embeddings...")
            test_embeddings, test_labels = extract_embeddings(
                evo2_model,
                test_df["sequence"].tolist(),
                test_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"\nPretrained embedding shape: {test_embeddings.shape}")

            # Save pretrained embeddings
            np.savez(
                embeddings_path,
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
            )
            print(f"Saved pretrained embeddings to: {embeddings_path}")

        # Extract random model embeddings if needed and not cached
        if need_random_model and not random_model_cached:
            print("\n" + "=" * 60)
            print("Extracting embeddings from randomly-initialized model")
            print("=" * 60)
            print("Replacing pretrained weights with random initialization...")
            replace_with_random_weights(evo2_model, args.model, seed=args.seed + 100)

            # Cast to float32 to prevent NaN from bfloat16 overflow with random weights.
            # Pretrained weights are in a stable regime learned during training, but random
            # weights can cause activations to explode through 32 layers in bfloat16.
            print("  Casting model to float32 for numerical stability with random weights...")
            evo2_model.model.float()

            # Smoke test: run ONE sequence to verify no NaN before full extraction
            print("  Smoke test: running one sequence through random model...")
            test_seq = train_df["sequence"].iloc[0]
            if args.max_length is not None and len(test_seq) > args.max_length:
                test_seq = test_seq[:args.max_length]
            test_ids = torch.tensor(
                evo2_model.tokenizer.tokenize(test_seq), dtype=torch.int
            ).unsqueeze(0).to('cuda:0')
            with torch.no_grad():
                test_out, test_emb = evo2_model(
                    test_ids, return_embeddings=True, layer_names=[args.layer]
                )
            test_vec = test_emb[args.layer].cpu().float().numpy()
            has_nan = np.isnan(test_vec).any()
            has_inf = np.isinf(test_vec).any()
            print(f"    Output shape: {test_vec.shape}")
            print(f"    NaN: {has_nan}, Inf: {has_inf}")
            print(f"    Sample values: {test_vec.flatten()[:5]}")
            if has_nan or has_inf:
                raise RuntimeError(
                    "Random model produces NaN/Inf embeddings even in float32. "
                    "Cannot extract random baseline embeddings."
                )
            print("  Smoke test passed — proceeding with full extraction.")
            del test_ids, test_out, test_emb, test_vec

            print(f"\nExtracting random model train embeddings...")
            train_random_emb, _ = extract_embeddings(
                evo2_model,
                train_df["sequence"].tolist(),
                train_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"\nExtracting random model validation embeddings...")
            val_random_emb, _ = extract_embeddings(
                evo2_model,
                val_df["sequence"].tolist(),
                val_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"\nExtracting random model test embeddings...")
            test_random_emb, _ = extract_embeddings(
                evo2_model,
                test_df["sequence"].tolist(),
                test_df["label"].tolist(),
                args.layer,
                args.batch_size,
                args.max_length,
                args.pooling,
            )

            print(f"Random model embedding shape: {test_random_emb.shape}")

            # Sanity check: verify embeddings have variance and no NaN/Inf
            for name, emb in [("train", train_random_emb), ("val", val_random_emb), ("test", test_random_emb)]:
                nan_count = np.isnan(emb).sum()
                inf_count = np.isinf(emb).sum()
                nan_rows = np.any(np.isnan(emb), axis=1).sum()
                if nan_count > 0 or inf_count > 0:
                    print(f"  WARNING: {name} random embeddings contain "
                          f"{nan_count} NaN, {inf_count} Inf values "
                          f"({nan_rows}/{len(emb)} rows affected)")

            test_var = np.var(test_random_emb, axis=0).mean()
            test_range = np.ptp(test_random_emb, axis=0).mean()
            all_same = np.allclose(test_random_emb[0], test_random_emb[1]) if len(test_random_emb) > 1 else False
            print(f"  Random embedding sanity check:")
            print(f"    Mean per-feature variance: {test_var:.6f}")
            print(f"    Mean per-feature range: {test_range:.6f}")
            print(f"    First two embeddings identical: {all_same}")
            if test_var < 1e-10:
                print(f"  WARNING: Random embeddings have near-zero variance!")
                print(f"  All sequences are producing the same embedding.")
                print(f"  First embedding sample (first 10 dims): {test_random_emb[0, :10]}")

            # Cache random model embeddings
            np.savez(
                random_model_path,
                train_embeddings=train_random_emb,
                val_embeddings=val_random_emb,
                test_embeddings=test_random_emb,
            )
            print(f"Saved random model embeddings to: {random_model_path}")

            random_model_embeddings = {
                'train': train_random_emb,
                'val': val_random_emb,
                'test': test_random_emb,
            }

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

        # Run random baseline (randomly-initialized model)
        random_results, random_preds = run_random_baseline(
            train_labels, val_labels, test_labels,
            args.seed,
            args.nn_hidden_dim, args.nn_epochs, args.nn_lr,
            device, args.output_dir,
            random_model_embeddings=random_model_embeddings,
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
