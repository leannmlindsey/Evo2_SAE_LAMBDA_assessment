#!/usr/bin/env python3
"""
Evo2 SAE Prophage Detection Pipeline
=====================================
Detect prophages in bacterial genomes using Evo2's SAE feature f/19746.

Based on the official Evo2 SAE notebook from Goodfire/Arc Institute.

Usage:
    python prophage_detection.py --genome_dir /path/to/genomes --output_dir ./results

Requirements:
    - Evo2 installed (pip install evo2)
    - SAE weights downloaded
    - H200/A100 GPU
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from math import prod
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import hf_hub_download

from evo2 import Evo2

# Disable gradient computation for inference
torch.set_grad_enabled(False)

# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """Pipeline configuration."""
    # Paths
    genome_dir: str = ""
    output_dir: str = "./prophage_results"
    ground_truth: Optional[str] = None
    sae_weights_path: str = ""  # Will be set from HuggingFace download

    # Model
    model_name: str = "evo2_7b"  # Will use evo2_7b (not evo2_7b_262k for compatibility)
    device: str = "cuda:0"

    # SAE settings (from Evo2 paper)
    prophage_feature_idx: int = 19746
    sae_layer: str = "blocks-26"  # Note: hyphen, not dot (for hook system)
    d_hidden: int = 4096  # Evo2 hidden dimension at this layer
    expansion_factor: int = 8  # SAE expansion factor

    # Detection parameters
    window_size: int = 8192
    overlap: int = 1024
    activation_threshold: float = 0.5
    min_prophage_length: int = 5000
    merge_distance: int = 2000

    # Processing
    batch_size: int = 1
    save_activations: bool = True


# ============================================================
# ModelScope - Hook management for Evo2 (from official notebook)
# ============================================================

class ModelScope:
    """Class for adding, using, and removing PyTorch hooks with a model."""

    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self.override_store = {}
        self._build_module_dict()

    def _build_module_dict(self):
        """Walks the model's module tree and builds a name: module map."""
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

    def remove_hook(self, hook_name):
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_all_hooks(self):
        hooks = list(self.hooks.keys())
        for hook_name in hooks:
            self.remove_hook(hook_name)

    def clear_all_caches(self):
        for module_str in self.activations_cache.keys():
            self.activations_cache[module_str] = []


# ============================================================
# BatchTopKTiedSAE - SAE architecture (from official notebook)
# ============================================================

class BatchTopKTiedSAE(torch.nn.Module):
    """Batch TopK Tied-weight Sparse Autoencoder."""

    def __init__(self, d_in, d_hidden, k, device, dtype, tiebreaker_epsilon: float = 1e-6):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k

        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.device = device
        self.dtype = dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(self.device, self.dtype)

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

        # Convert to float32 for topk operation
        f_flat = f.flatten().float()
        f_topk = torch.topk(f_flat, numel, dim=-1)
        result = torch.zeros_like(f_flat).scatter(-1, f_topk.indices, f_topk.values)
        return result.reshape(f.shape).to(f.dtype)

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def load_topk_sae(sae_path: str, d_hidden: int, device: str, dtype: torch.dtype, expansion_factor: int = 8):
    """Load a TopK SAE from checkpoint."""
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")

    # Strip _orig_mod. and module. prefixes
    new_dict = {}
    for key, item in sae_dict.items():
        new_dict[key.replace("_orig_mod.", "").replace("module.", "")] = item
    sae_dict = new_dict

    cached_sae = BatchTopKTiedSAE(
        d_hidden,
        d_hidden * expansion_factor,
        64,  # TopK 64
        device,
        dtype,
    )
    cached_sae.load_state_dict(sae_dict)
    return cached_sae


# ============================================================
# ObservableEvo2 - Evo2 wrapper with hook support (from official notebook)
# ============================================================

INTERVENTION_INTERFACE = Callable[[torch.Tensor], torch.Tensor]


class ObservableEvo2:
    """Evo2 model wrapper with activation caching via hooks."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evo_model = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self):
        return next(self.evo_model.model.parameters()).device

    @property
    def dtype(self):
        return self.evo_model.dtype

    def list_modules(self):
        return self.scope.list_modules()

    def forward(
        self,
        toks: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions: dict[str, INTERVENTION_INTERFACE] = None,
    ):
        if not interventions:
            interventions = {}
        if not cache_activations_at:
            cache_activations_at = []

        output_cache = {}
        layers = list(set(list(interventions.keys()) + cache_activations_at))

        if layers:
            for layer in layers:
                def _intervene(model, input, output, layer=layer):
                    acts = output[0] if isinstance(output, tuple) else output

                    if layer in interventions:
                        acts = interventions[layer](acts)

                    if layer in cache_activations_at:
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
# Prophage Detector
# ============================================================

class ProphageDetector:
    """Detect prophages using Evo2 SAE features."""

    def __init__(self, config: Config):
        self.config = config

        print(f"\n{'='*60}")
        print("Initializing Prophage Detector")
        print(f"{'='*60}")
        print(f"Model: {config.model_name}")

        # Load Evo2 with hook support
        print(f"\nLoading Evo2 model...")
        self.model = ObservableEvo2(config.model_name)
        print(f"  Device: {self.model.device}")
        print(f"  d_hidden: {self.model.d_hidden}")
        print(f"  ✓ Evo2 loaded")

        # Download and load SAE
        print(f"\nLoading SAE...")
        if not config.sae_weights_path:
            config.sae_weights_path = hf_hub_download(
                repo_id="Goodfire/Evo-2-Layer-26-Mixed",
                filename="sae-layer26-mixed-expansion_8-k_64.pt",
                repo_type="model"
            )
        print(f"  SAE path: {config.sae_weights_path}")

        self.sae = load_topk_sae(
            config.sae_weights_path,
            d_hidden=self.model.d_hidden,
            device=self.model.device,
            dtype=torch.bfloat16,
            expansion_factor=config.expansion_factor
        )
        print(f"  d_in: {self.sae.d_in}, d_hidden: {self.sae.d_hidden}")
        print(f"  ✓ SAE loaded")

        # Verify prophage feature exists
        print(f"\nProphage feature index: {config.prophage_feature_idx}")
        if config.prophage_feature_idx >= self.sae.d_hidden:
            raise ValueError(
                f"Prophage feature {config.prophage_feature_idx} >= SAE dimension {self.sae.d_hidden}"
            )
        print(f"  ✓ Feature index valid")

    def get_feature_activations(self, sequence: str) -> np.ndarray:
        """Get SAE feature activations for a sequence."""
        toks = self.model.tokenizer.tokenize(sequence)
        toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            logits, acts = self.model.forward(toks, cache_activations_at=[self.config.sae_layer])
            feats = self.sae.encode(acts[self.config.sae_layer][0])

        return feats.cpu().detach().float().numpy()

    def get_prophage_activations(self, sequence: str) -> np.ndarray:
        """Get prophage feature activations across a sequence."""
        seq_len = len(sequence)
        window_size = self.config.window_size
        overlap = self.config.overlap

        # Initialize
        activations = np.zeros(seq_len)
        counts = np.zeros(seq_len)

        # Process in windows
        positions = list(range(0, seq_len, window_size - overlap))

        for start in tqdm(positions, desc="  Processing windows", leave=False):
            end = min(start + window_size, seq_len)
            window_seq = sequence[start:end]

            if len(window_seq) < 100:
                continue

            try:
                # Get all SAE feature activations for this window
                all_features = self.get_feature_activations(window_seq)

                # Extract prophage feature (shape: [seq_len, n_features])
                prophage_acts = all_features[:, self.config.prophage_feature_idx]

                # Handle length mismatch (tokenization may differ from bp)
                actual_len = min(len(prophage_acts), end - start)
                activations[start:start+actual_len] += prophage_acts[:actual_len]
                counts[start:start+actual_len] += 1

            except Exception as e:
                print(f"    Warning: Error at position {start}: {e}")
                continue

        # Average overlapping regions
        counts[counts == 0] = 1
        activations = activations / counts

        return activations
    
    def call_prophages(self, activations: np.ndarray, genome_id: str) -> List[Dict]:
        """Call prophage regions from activation signal."""
        threshold = self.config.activation_threshold
        min_length = self.config.min_prophage_length
        merge_dist = self.config.merge_distance
        
        # Find regions above threshold
        above = activations > threshold
        
        # Find contiguous regions
        regions = []
        in_region = False
        start = 0
        
        for i, is_above in enumerate(above):
            if is_above and not in_region:
                start = i
                in_region = True
            elif not is_above and in_region:
                regions.append((start, i))
                in_region = False
        
        if in_region:
            regions.append((start, len(activations)))
        
        # Filter by minimum length
        regions = [(s, e) for s, e in regions if e - s >= min_length]
        
        # Merge nearby regions
        merged = []
        for region in regions:
            if merged and region[0] - merged[-1][1] < merge_dist:
                merged[-1] = (merged[-1][0], region[1])
            else:
                merged.append(region)
        
        # Format output
        predictions = []
        for i, (start, end) in enumerate(merged):
            predictions.append({
                'genome_id': genome_id,
                'prophage_id': f"{genome_id}_prophage_{i+1}",
                'start': int(start),
                'end': int(end),
                'length': int(end - start),
                'mean_activation': float(activations[start:end].mean()),
                'max_activation': float(activations[start:end].max()),
            })
        
        return predictions
    
    def process_genome(self, sequence: str, genome_id: str) -> Tuple[List[Dict], np.ndarray]:
        """Process a single genome."""
        print(f"\nProcessing: {genome_id} ({len(sequence):,} bp)")
        
        activations = self.get_prophage_activations(sequence)
        predictions = self.call_prophages(activations, genome_id)
        
        print(f"  Found {len(predictions)} prophage region(s)")
        for pred in predictions:
            print(f"    {pred['prophage_id']}: {pred['start']:,}-{pred['end']:,} "
                  f"({pred['length']:,} bp, activation={pred['mean_activation']:.3f})")
        
        return predictions, activations


# ============================================================
# File I/O
# ============================================================

def load_fasta(filepath: str) -> Dict[str, str]:
    """Load sequences from FASTA file."""
    sequences = {}
    current_name = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_name:
            sequences[current_name] = ''.join(current_seq)
    
    return sequences


def save_results(predictions: List[Dict], output_dir: Path) -> None:
    """Save predictions in multiple formats."""
    df = pd.DataFrame(predictions)
    
    if df.empty:
        print("No predictions to save.")
        return
    
    # CSV
    csv_path = output_dir / "prophage_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # BED format
    bed_path = output_dir / "prophage_predictions.bed"
    bed_df = df[['genome_id', 'start', 'end', 'prophage_id', 'mean_activation']].copy()
    bed_df.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'score']
    bed_df['score'] = (bed_df['score'] * 1000).astype(int).clip(0, 1000)
    bed_df.to_csv(bed_path, sep='\t', header=False, index=False)
    print(f"  Saved: {bed_path}")
    
    # GFF format
    gff_path = output_dir / "prophage_predictions.gff3"
    with open(gff_path, 'w') as f:
        f.write("##gff-version 3\n")
        for _, row in df.iterrows():
            f.write(f"{row['genome_id']}\tEvo2_SAE\tprophage\t{row['start']+1}\t{row['end']}\t"
                   f"{row['mean_activation']:.3f}\t.\t.\tID={row['prophage_id']}\n")
    print(f"  Saved: {gff_path}")


# ============================================================
# Evaluation
# ============================================================

def evaluate(predictions_df: pd.DataFrame, ground_truth_file: str) -> Dict:
    """Evaluate predictions against ground truth."""
    # Load ground truth (BED format)
    gt_df = pd.read_csv(
        ground_truth_file,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'name', 'score', 'strand'],
        usecols=[0, 1, 2, 3, 4, 5] if pd.read_csv(ground_truth_file, sep='\t', nrows=1).shape[1] >= 6 
                else [0, 1, 2]
    )
    
    # Standardize column names
    if len(gt_df.columns) == 3:
        gt_df.columns = ['chrom', 'start', 'end']
    
    # Compute metrics
    tp, fp, fn = 0, 0, 0
    total_overlap_bp = 0
    total_pred_bp = 0
    total_gt_bp = 0
    
    # Group by genome
    pred_genomes = set(predictions_df['genome_id'].unique())
    gt_genomes = set(gt_df['chrom'].unique())
    all_genomes = pred_genomes | gt_genomes
    
    for genome in all_genomes:
        preds = predictions_df[predictions_df['genome_id'] == genome]
        gts = gt_df[gt_df['chrom'] == genome]
        
        # Track which predictions and GTs are matched
        pred_matched = set()
        gt_matched = set()
        
        for pred_idx, pred in preds.iterrows():
            total_pred_bp += pred['end'] - pred['start']
            
            for gt_idx, gt in gts.iterrows():
                # Check overlap
                overlap_start = max(pred['start'], gt['start'])
                overlap_end = min(pred['end'], gt['end'])
                
                if overlap_start < overlap_end:
                    pred_matched.add(pred_idx)
                    gt_matched.add(gt_idx)
                    total_overlap_bp += overlap_end - overlap_start
        
        tp += len(pred_matched)
        fp += len(preds) - len(pred_matched)
        
        for gt_idx, gt in gts.iterrows():
            total_gt_bp += gt['end'] - gt['start']
            if gt_idx not in gt_matched:
                fn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    bp_precision = total_overlap_bp / total_pred_bp if total_pred_bp > 0 else 0
    bp_recall = total_overlap_bp / total_gt_bp if total_gt_bp > 0 else 0
    bp_f1 = 2 * bp_precision * bp_recall / (bp_precision + bp_recall) if (bp_precision + bp_recall) > 0 else 0
    
    union = total_pred_bp + total_gt_bp - total_overlap_bp
    iou = total_overlap_bp / union if union > 0 else 0
    
    return {
        'region_precision': precision,
        'region_recall': recall,
        'region_f1': f1,
        'bp_precision': bp_precision,
        'bp_recall': bp_recall,
        'bp_f1': bp_f1,
        'iou': iou,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'total_predictions': len(predictions_df),
        'total_ground_truth': len(gt_df),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detect prophages using Evo2 SAE feature f/19746",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_prophage_detection.py --genome_dir ./genomes --output_dir ./results
  
  # With ground truth evaluation
  python run_prophage_detection.py --genome_dir ./genomes --output_dir ./results --ground_truth prophages.bed
  
  # Adjust threshold
  python run_prophage_detection.py --genome_dir ./genomes --output_dir ./results --threshold 0.3
        """
    )
    
    parser.add_argument("--genome_dir", type=str, required=True,
                        help="Directory containing FASTA files")
    parser.add_argument("--output_dir", type=str, default="./prophage_results",
                        help="Output directory")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Ground truth BED file for evaluation")
    parser.add_argument("--model", type=str, default="evo2_7b",
                        choices=["evo2_7b", "evo2_40b", "evo2_7b_262k"],
                        help="Evo2 model to use")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Activation threshold for prophage calls")
    parser.add_argument("--min_length", type=int, default=5000,
                        help="Minimum prophage length (bp)")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save raw activation arrays")

    args = parser.parse_args()

    # Setup config
    config = Config(
        genome_dir=args.genome_dir,
        output_dir=args.output_dir,
        ground_truth=args.ground_truth,
        model_name=args.model,
        activation_threshold=args.threshold,
        min_prophage_length=args.min_length,
        save_activations=args.save_activations,
    )

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Evo2 SAE Prophage Detection")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now()}")
    print(f"Genome directory: {config.genome_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model: {config.model_name}")
    print(f"Threshold: {config.activation_threshold}")

    # Initialize detector
    detector = ProphageDetector(config)

    # Save config (after detector init so SAE path is populated)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Find genome files
    genome_dir = Path(config.genome_dir)
    genome_files = (
        list(genome_dir.glob("*.fasta")) +
        list(genome_dir.glob("*.fna")) +
        list(genome_dir.glob("*.fa"))
    )
    
    print(f"\nFound {len(genome_files)} genome file(s)")
    
    # Process genomes
    all_predictions = []
    
    for genome_file in tqdm(genome_files, desc="Processing genomes"):
        sequences = load_fasta(str(genome_file))
        
        for seq_name, sequence in sequences.items():
            genome_id = f"{genome_file.stem}_{seq_name}"
            
            predictions, activations = detector.process_genome(sequence, genome_id)
            all_predictions.extend(predictions)
            
            # Save activations
            if config.save_activations:
                np.save(output_dir / f"{genome_id}_activations.npy", activations)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    results_df = pd.DataFrame(all_predictions)
    save_results(all_predictions, output_dir)
    
    # Evaluate if ground truth provided
    if config.ground_truth and len(all_predictions) > 0:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        
        metrics = evaluate(results_df, config.ground_truth)
        
        with open(output_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nRegion-level metrics:")
        print(f"  Precision: {metrics['region_precision']:.4f}")
        print(f"  Recall:    {metrics['region_recall']:.4f}")
        print(f"  F1:        {metrics['region_f1']:.4f}")
        
        print(f"\nBase-pair level metrics:")
        print(f"  Precision: {metrics['bp_precision']:.4f}")
        print(f"  Recall:    {metrics['bp_recall']:.4f}")
        print(f"  F1:        {metrics['bp_f1']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")
        
        print(f"\nCounts:")
        print(f"  True positives:  {metrics['true_positives']}")
        print(f"  False positives: {metrics['false_positives']}")
        print(f"  False negatives: {metrics['false_negatives']}")
    
    print(f"\n{'='*60}")
    print(f"Complete! Total predictions: {len(all_predictions)}")
    print(f"Results saved to: {output_dir}")
    print(f"End time: {datetime.now()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
