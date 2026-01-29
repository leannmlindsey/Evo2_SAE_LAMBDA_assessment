#!/usr/bin/env python3
"""
Evo2 SAE Prophage Detection Pipeline
=====================================
Detect prophages in bacterial genomes using Evo2's SAE feature f/19746.

Usage:
    # First, inspect the SAE checkpoint:
    python inspect_sae_checkpoint.py
    
    # Then run detection:
    python run_prophage_detection.py --genome_dir /path/to/genomes --output_dir ./results
    
    # With ground truth evaluation:
    python run_prophage_detection.py --genome_dir /path/to/genomes --output_dir ./results --ground_truth annotations.bed

Requirements:
    - Evo2 installed (pip install evo2)
    - SAE weights downloaded to ~/evo2/sae_weights/
    - H200/A100 GPU with FP8 support
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime

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
    sae_weights_dir: str = "/home/lindseylm/evo2/sae_weights"
    
    # Model
    model_name: str = "evo2_7b"
    device: str = "cuda:0"  # Use first GPU by default
    
    # SAE settings (from Evo2 paper)
    prophage_feature_idx: int = 19746
    sae_layer: str = "blocks.26"  # Will be verified by inspect script
    
    # Detection parameters
    window_size: int = 8192
    overlap: int = 1024
    activation_threshold: float = 0.5
    min_prophage_length: int = 5000
    merge_distance: int = 2000
    
    # Processing
    batch_size: int = 1  # Sequences per batch (keep at 1 for variable lengths)
    save_activations: bool = True


# ============================================================
# SAE Module (Update after running inspect_sae_checkpoint.py)
# ============================================================

class SAEModule(torch.nn.Module):
    """
    Sparse Autoencoder for Evo2.

    Based on Goodfire SAE checkpoint inspection:
    - d_model = 8192 (Evo2 hidden dimension)
    - d_sae = 65536 (expansion factor = 8)
    - k = 64 (TopK sparsity)
    - Keys: W_enc, b_enc, W_dec, b_dec
    """

    def __init__(self, d_model: int = 8192, d_sae: int = 65536, k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k  # TopK sparsity

        # Weights (will be loaded from checkpoint)
        self.W_enc = torch.nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations."""
        # x: (..., d_model) -> pre_acts: (..., d_sae)
        pre_acts = torch.nn.functional.linear(x, self.W_enc, self.b_enc)

        # TopK activation
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_indices, torch.relu(topk_values))

        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sparse activations."""
        return torch.nn.functional.linear(acts, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, activations)."""
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cuda") -> "SAEModule":
        """Load pretrained SAE from Goodfire checkpoint.

        Checkpoint format:
        - _orig_mod.W: encoder weights (d_sae, d_model), decoder is transpose
        - _orig_mod.b_enc: encoder bias (d_sae,)
        - _orig_mod.b_dec: decoder bias (d_model,)
        """
        print(f"Loading SAE from {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"  Checkpoint keys: {list(state_dict.keys())}")

        # Get the weight matrix - handles _orig_mod prefix from torch.compile
        if '_orig_mod.W' in state_dict:
            W = state_dict['_orig_mod.W']
            b_enc = state_dict['_orig_mod.b_enc']
            b_dec = state_dict['_orig_mod.b_dec']
        elif 'W' in state_dict:
            W = state_dict['W']
            b_enc = state_dict['b_enc']
            b_dec = state_dict['b_dec']
        else:
            raise ValueError(f"Unexpected checkpoint format. Keys: {list(state_dict.keys())}")

        # W shape: (d_model, d_sae) - for tied-weight SAE
        # Encoder: x @ W + b_enc -> (batch, seq, d_sae)
        # Decoder: acts @ W.T + b_dec -> (batch, seq, d_model)
        d_model, d_sae = W.shape
        print(f"  Dimensions: d_model={d_model}, d_sae={d_sae} (expansion={d_sae//d_model}x)")

        # Create model
        model = cls(d_model=d_model, d_sae=d_sae)

        # Load weights - this is a tied-weight SAE
        # W is (d_model, d_sae), used as: x @ W + b_enc for encoding
        # For F.linear(x, weight, bias) which computes x @ weight.T + bias:
        # - W_enc should be (d_sae, d_model) so x @ W_enc.T = x @ W
        # - W_dec should be (d_model, d_sae) so acts @ W_dec.T = acts @ W.T
        model.W_enc.data = W.T  # (d_sae, d_model)
        model.b_enc.data = b_enc
        model.W_dec.data = W    # (d_model, d_sae), so F.linear gives acts @ W.T
        model.b_dec.data = b_dec

        model.to(device)
        model.eval()

        print(f"  ✓ SAE loaded successfully")
        return model


# ============================================================
# Prophage Detector
# ============================================================

class ProphageDetector:
    """Detect prophages using Evo2 SAE features."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"\n{'='*60}")
        print("Initializing Prophage Detector")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model: {config.model_name}")
        
        # Load Evo2
        print(f"\nLoading Evo2 model...")
        from evo2 import Evo2
        self.model = Evo2(config.model_name)
        print(f"  ✓ Evo2 loaded")
        
        # Load SAE
        print(f"\nLoading SAE...")
        sae_path = self._find_sae_checkpoint()
        self.sae = SAEModule.from_pretrained(sae_path, str(self.device))
        print(f"  ✓ SAE loaded")
        
        # Verify prophage feature exists
        print(f"\nProphage feature index: {config.prophage_feature_idx}")
        if config.prophage_feature_idx >= self.sae.d_sae:
            raise ValueError(
                f"Prophage feature {config.prophage_feature_idx} >= SAE dimension {self.sae.d_sae}"
            )
        print(f"  ✓ Feature index valid")
        
    def _find_sae_checkpoint(self) -> str:
        """Find the SAE checkpoint file."""
        sae_dir = Path(self.config.sae_weights_dir)
        
        # Look for checkpoint files
        for pattern in ['*.pt', '*.pth', '*.safetensors', '*.bin']:
            files = list(sae_dir.glob(pattern))
            if files:
                # Prefer the largest file (likely the weights)
                return str(max(files, key=lambda f: f.stat().st_size))
        
        # Check subdirectories
        for pattern in ['**/*.pt', '**/*.safetensors']:
            files = list(sae_dir.glob(pattern))
            if files:
                return str(max(files, key=lambda f: f.stat().st_size))
        
        raise FileNotFoundError(f"No checkpoint found in {sae_dir}")
    
    def get_embeddings(self, sequence: str) -> torch.Tensor:
        """Extract embeddings from Evo2 at the SAE layer (block 26)."""
        # Tokenize
        input_ids = torch.tensor(
            self.model.tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0).to(self.device)

        # Based on inspect_sae_checkpoint.py output:
        # blocks.26 is a ParallelGatedConvBlock with mlp.l3 as the MLP output
        # The SAE was trained on layer 26 outputs (d_model=8192)
        layer_candidates = [
            "blocks.26.mlp.l3",      # MLP output (most likely)
            "blocks.26.post_norm",   # After normalization
            "blocks.26",             # Full block output
        ]

        for layer_name in layer_candidates:
            try:
                with torch.no_grad():
                    outputs, embeddings = self.model(
                        input_ids,
                        return_embeddings=True,
                        layer_names=[layer_name]
                    )
                if layer_name in embeddings:
                    emb = embeddings[layer_name].squeeze(0)
                    # Verify dimension matches SAE input
                    if emb.shape[-1] == self.sae.d_model:
                        return emb
                    else:
                        print(f"    Layer {layer_name} has dim {emb.shape[-1]}, expected {self.sae.d_model}")
                        continue
            except Exception as e:
                print(f"    Layer {layer_name} failed: {e}")
                continue

        raise RuntimeError(f"Could not extract embeddings with d_model={self.sae.d_model}. Tried: {layer_candidates}")
    
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
                # Get Evo2 embeddings
                embeddings = self.get_embeddings(window_seq)
                
                # Get SAE activations
                with torch.no_grad():
                    _, sae_acts = self.sae(embeddings)
                
                # Extract prophage feature
                prophage_acts = sae_acts[:, self.config.prophage_feature_idx].cpu().numpy()
                
                # Handle length mismatch (tokenization may differ)
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
                        choices=["evo2_7b", "evo2_40b"],
                        help="Evo2 model to use")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Activation threshold for prophage calls")
    parser.add_argument("--min_length", type=int, default=5000,
                        help="Minimum prophage length (bp)")
    parser.add_argument("--save_activations", action="store_true",
                        help="Save raw activation arrays")
    parser.add_argument("--sae_weights_dir", type=str, default="/home/lindseylm/evo2/sae_weights",
                        help="Directory containing SAE weights")

    args = parser.parse_args()

    # Setup config
    config = Config(
        genome_dir=args.genome_dir,
        output_dir=args.output_dir,
        ground_truth=args.ground_truth,
        model_name=args.model,
        device=args.device,
        activation_threshold=args.threshold,
        min_prophage_length=args.min_length,
        save_activations=args.save_activations,
        sae_weights_dir=args.sae_weights_dir,
    )
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evo2 SAE Prophage Detection")
    print(f"{'='*60}")
    print(f"Start time: {datetime.now()}")
    print(f"Genome directory: {config.genome_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Threshold: {config.activation_threshold}")
    
    # Initialize detector
    detector = ProphageDetector(config)
    
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
