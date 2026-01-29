#!/usr/bin/env python3
"""
Convert SAE activation signals to genomic regions using density-based clustering.

This script:
1. Loads activation data from .npy files
2. Identifies positions above a threshold
3. Clusters positions using HDBSCAN and/or OPTICS
4. Filters clusters to keep only those > min_size (default 1kb)
5. Merges clusters within merge_distance (default 3kb) of each other
6. Outputs predicted regions in BED format and compares to ground truth

Usage:
    python cluster_activations.py \
        --results_dir ./lambda_results_7b \
        --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
        --output_dir ./cluster_results
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Clustering imports
from sklearn.cluster import HDBSCAN, OPTICS


def load_ground_truth(csv_path):
    """Load ground truth prophage regions."""
    gt = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assembly = row['Assembly']
            if assembly not in gt:
                gt[assembly] = []
            gt[assembly].append({
                'ncbi_id': row['NCBI Id'],
                'start': int(row['start']),
                'end': int(row['end']),
                'organism': row['Organism Name'],
            })
    return gt


def get_assembly_from_filename(filename):
    """Extract assembly ID from activation filename."""
    name = Path(filename).stem
    return name.replace('_activations', '')


def cluster_positions_simple(positions, max_gap=100):
    """
    Simple O(n) clustering for 1D genomic positions.

    Groups positions that are within max_gap of each other into regions.
    Much faster than HDBSCAN/OPTICS for genomic data.

    Args:
        positions: 1D array of genomic positions (must be sorted)
        max_gap: Maximum gap between positions to consider them part of same cluster

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) == 0:
        return []

    positions = np.sort(positions)
    regions = []

    # Start first region
    region_start = positions[0]
    region_end = positions[0]

    for pos in positions[1:]:
        if pos - region_end <= max_gap:
            # Extend current region
            region_end = pos
        else:
            # Save current region and start new one
            regions.append((int(region_start), int(region_end)))
            region_start = pos
            region_end = pos

    # Don't forget the last region
    regions.append((int(region_start), int(region_end)))

    return regions


def cluster_positions_hdbscan(positions, min_cluster_size=100, min_samples=10, cluster_selection_epsilon=0.0):
    """
    Cluster genomic positions using HDBSCAN.

    Args:
        positions: 1D array of genomic positions
        min_cluster_size: Minimum number of positions to form a cluster
        min_samples: HDBSCAN min_samples parameter
        cluster_selection_epsilon: Distance threshold for cluster merging (0 = disabled)

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) < min_cluster_size:
        return []

    # Reshape for sklearn (needs 2D input)
    X = positions.reshape(-1, 1).astype(np.float64)

    try:
        # Use simpler HDBSCAN config without epsilon
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'  # Excess of Mass
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        print(f"    HDBSCAN failed: {e}")
        return []

    # Extract regions from clusters
    regions = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for label in unique_labels:
        cluster_positions = positions[labels == label]
        start = int(cluster_positions.min())
        end = int(cluster_positions.max())
        regions.append((start, end))

    return sorted(regions, key=lambda x: x[0])


def cluster_positions_optics(positions, min_samples=50, xi=0.05, min_cluster_size=100):
    """
    Cluster genomic positions using OPTICS.

    Args:
        positions: 1D array of genomic positions
        min_samples: OPTICS min_samples parameter
        xi: Steepness threshold for cluster extraction
        min_cluster_size: Minimum cluster size

    Returns:
        List of (start, end) tuples for each cluster
    """
    if len(positions) < min_cluster_size:
        return []

    X = positions.reshape(-1, 1).astype(np.float64)

    try:
        clusterer = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            n_jobs=-1  # Use all cores
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        print(f"    OPTICS failed: {e}")
        return []

    regions = []
    unique_labels = set(labels)
    unique_labels.discard(-1)

    for label in unique_labels:
        cluster_positions = positions[labels == label]
        start = int(cluster_positions.min())
        end = int(cluster_positions.max())
        regions.append((start, end))

    return sorted(regions, key=lambda x: x[0])


def merge_nearby_regions(regions, merge_distance=3000):
    """
    Merge regions that are within merge_distance of each other.

    Args:
        regions: List of (start, end) tuples, sorted by start
        merge_distance: Maximum gap between regions to merge (default 3kb)

    Returns:
        List of merged (start, end) tuples
    """
    if not regions:
        return []

    merged = [list(regions[0])]

    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]

        # Check if this region is within merge_distance of the previous
        if start - prev_end <= merge_distance:
            # Merge: extend the previous region
            merged[-1][1] = max(prev_end, end)
        else:
            # Start new region
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def filter_by_size(regions, min_size=1000):
    """Filter regions to keep only those >= min_size."""
    return [(s, e) for s, e in regions if e - s >= min_size]


def calculate_metrics(predicted_regions, gt_regions, seq_len):
    """
    Calculate precision, recall, F1 for predicted vs ground truth regions.

    Uses overlap-based matching:
    - A predicted region is a true positive if it overlaps any GT region by >= 50%
    - Precision = TP / predicted
    - Recall = GT regions with any overlap / total GT
    """
    if not predicted_regions and not gt_regions:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}

    if not predicted_regions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_regions)}

    if not gt_regions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted_regions), 'fn': 0}

    # Check each predicted region for overlap with GT
    tp = 0
    for pred_start, pred_end in predicted_regions:
        pred_len = pred_end - pred_start
        for gt in gt_regions:
            gt_start, gt_end = gt['start'], gt['end']
            overlap_start = max(pred_start, gt_start)
            overlap_end = min(pred_end, gt_end)
            overlap = max(0, overlap_end - overlap_start)

            # Consider it a TP if overlap >= 50% of predicted region
            if overlap >= 0.5 * pred_len:
                tp += 1
                break

    fp = len(predicted_regions) - tp

    # Check recall: how many GT regions have any overlap with predictions
    gt_found = 0
    for gt in gt_regions:
        gt_start, gt_end = gt['start'], gt['end']
        for pred_start, pred_end in predicted_regions:
            overlap_start = max(pred_start, gt_start)
            overlap_end = min(pred_end, gt_end)
            if overlap_end > overlap_start:
                gt_found += 1
                break

    fn = len(gt_regions) - gt_found

    precision = tp / len(predicted_regions) if predicted_regions else 0
    recall = gt_found / len(gt_regions) if gt_regions else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt_found': gt_found,
        'gt_total': len(gt_regions),
        'pred_total': len(predicted_regions),
    }


def process_genome(activations, assembly_id, gt_regions, args):
    """Process a single genome and return clustering results."""

    seq_len = len(activations)

    # Find positions above threshold
    positions = np.where(activations > args.threshold)[0]

    results = {
        'assembly': assembly_id,
        'seq_len': seq_len,
        'positions_above_threshold': len(positions),
        'gt_regions': len(gt_regions),
    }

    if len(positions) == 0:
        results['simple_regions'] = []
        results['hdbscan_regions'] = []
        results['optics_regions'] = []
        results['simple_metrics'] = calculate_metrics([], gt_regions, seq_len)
        results['hdbscan_metrics'] = calculate_metrics([], gt_regions, seq_len)
        results['optics_metrics'] = calculate_metrics([], gt_regions, seq_len)
        return results

    # Simple clustering (fast, O(n))
    simple_regions = cluster_positions_simple(positions, max_gap=args.max_gap)
    simple_regions = merge_nearby_regions(simple_regions, args.merge_distance)
    simple_regions = filter_by_size(simple_regions, args.min_region_size)
    results['simple_regions'] = simple_regions
    results['simple_metrics'] = calculate_metrics(simple_regions, gt_regions, seq_len)

    # HDBSCAN/OPTICS only if requested (slow)
    if args.use_hdbscan:
        hdbscan_regions = cluster_positions_hdbscan(
            positions,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
        hdbscan_regions = merge_nearby_regions(hdbscan_regions, args.merge_distance)
        hdbscan_regions = filter_by_size(hdbscan_regions, args.min_region_size)
        results['hdbscan_regions'] = hdbscan_regions
        results['hdbscan_metrics'] = calculate_metrics(hdbscan_regions, gt_regions, seq_len)
    else:
        results['hdbscan_regions'] = []
        results['hdbscan_metrics'] = {}

    if args.use_optics:
        optics_regions = cluster_positions_optics(
            positions,
            min_samples=args.min_samples,
            xi=args.xi,
            min_cluster_size=args.min_cluster_size
        )
        optics_regions = merge_nearby_regions(optics_regions, args.merge_distance)
        optics_regions = filter_by_size(optics_regions, args.min_region_size)
        results['optics_regions'] = optics_regions
        results['optics_metrics'] = calculate_metrics(optics_regions, gt_regions, seq_len)
    else:
        results['optics_regions'] = []
        results['optics_metrics'] = {}

    return results


def save_bed(regions, output_path, assembly_id):
    """Save regions in BED format."""
    with open(output_path, 'w') as f:
        for i, (start, end) in enumerate(regions):
            name = f"{assembly_id}_prophage_{i+1}"
            f.write(f"{assembly_id}\t{start}\t{end}\t{name}\t0\t+\n")


def plot_comparison(activations, simple_regions, hdbscan_regions, gt_regions,
                    assembly_id, output_path, threshold):
    """Generate comparison plot showing clustering results vs ground truth."""

    seq_len = len(activations)

    # Downsample for plotting
    max_points = 50000
    if seq_len > max_points:
        bin_size = seq_len // max_points
        n_bins = seq_len // bin_size
        truncated = activations[:n_bins * bin_size]
        reshaped = truncated.reshape(n_bins, bin_size)
        plot_acts = reshaped.max(axis=1)
        x_coords = np.arange(n_bins) * bin_size + bin_size // 2
    else:
        plot_acts = activations
        x_coords = np.arange(seq_len)

    fig, axes = plt.subplots(4, 1, figsize=(20, 10), height_ratios=[3, 1, 1, 1], sharex=True)

    # Top: Activation signal
    ax1 = axes[0]
    ax1.fill_between(x_coords, 0, plot_acts, alpha=0.3, color='blue')
    ax1.plot(x_coords, plot_acts, lw=0.5, alpha=0.9, color='blue')
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    ax1.set_ylabel('SAE Activation')
    ax1.set_title(f'{assembly_id} - Clustering Comparison')
    ax1.set_xlim(0, seq_len)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Simple clustering regions (fast method)
    ax2 = axes[1]
    ax2.set_xlim(0, seq_len)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Predicted')
    ax2.set_yticks([])
    for start, end in simple_regions:
        ax2.axvspan(start, end, alpha=0.7, color='green')
        mid = (start + end) / 2
        size_kb = (end - start) / 1000
        ax2.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not simple_regions:
        ax2.text(0.5, 0.5, 'No regions', transform=ax2.transAxes, ha='center', va='center', color='gray')

    # HDBSCAN regions (if available)
    ax3 = axes[2]
    ax3.set_xlim(0, seq_len)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('HDBSCAN')
    ax3.set_yticks([])
    for start, end in hdbscan_regions:
        ax3.axvspan(start, end, alpha=0.7, color='purple')
        mid = (start + end) / 2
        size_kb = (end - start) / 1000
        ax3.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not hdbscan_regions:
        ax3.text(0.5, 0.5, 'Not run (use --use_hdbscan)', transform=ax3.transAxes, ha='center', va='center', color='gray')

    # Ground truth
    ax4 = axes[3]
    ax4.set_xlim(0, seq_len)
    ax4.set_ylim(0, 1)
    ax4.set_ylabel('Ground\nTruth')
    ax4.set_xlabel('Genomic Position (bp)')
    ax4.set_yticks([])
    for gt in gt_regions:
        start, end = gt['start'], gt['end']
        if start < seq_len:
            ax4.axvspan(start, min(end, seq_len), alpha=0.7, color='red')
            mid = (start + min(end, seq_len)) / 2
            size_kb = (min(end, seq_len) - start) / 1000
            ax4.text(mid, 0.5, f'{size_kb:.1f}kb', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    if not gt_regions:
        ax4.text(0.5, 0.5, 'No ground truth', transform=ax4.transAxes, ha='center', va='center', color='gray')

    # Format x-axis
    def format_mb(x, p):
        return f'{x/1e6:.1f} Mb'
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(format_mb))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cluster SAE activations into prophage regions")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with *_activations.npy files")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth CSV file")
    parser.add_argument("--output_dir", type=str, default="./cluster_results", help="Output directory")

    # Threshold for considering a position "active"
    parser.add_argument("--threshold", type=float, default=0.3, help="Activation threshold (default: 0.3)")

    # Clustering parameters
    parser.add_argument("--max_gap", type=int, default=100, help="Max gap between positions in simple clustering (default: 100bp)")
    parser.add_argument("--use_hdbscan", action="store_true", help="Also run HDBSCAN clustering (slow)")
    parser.add_argument("--use_optics", action="store_true", help="Also run OPTICS clustering (slow)")
    parser.add_argument("--min_cluster_size", type=int, default=100, help="Minimum positions for HDBSCAN/OPTICS")
    parser.add_argument("--min_samples", type=int, default=20, help="HDBSCAN/OPTICS min_samples")
    parser.add_argument("--xi", type=float, default=0.05, help="OPTICS xi parameter")

    # Region filtering
    parser.add_argument("--min_region_size", type=int, default=1000, help="Minimum region size in bp (default: 1kb)")
    parser.add_argument("--merge_distance", type=int, default=3000, help="Merge regions within this distance (default: 3kb)")

    # Options
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--fix_artifacts", action="store_true", help="Fix window boundary artifacts before clustering")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    bed_dir = output_dir / "bed"
    bed_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Cluster SAE Activations into Prophage Regions")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Min region size: {args.min_region_size} bp")
    print(f"Merge distance: {args.merge_distance} bp")
    print(f"Clustering params: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  Found {len(gt)} assemblies with ground truth")

    # Find activation files
    results_dir = Path(args.results_dir)
    npy_files = sorted(results_dir.glob("*_activations.npy"))
    print(f"\nFound {len(npy_files)} activation files")

    # Process each genome
    all_results = []

    for npy_file in tqdm(npy_files, desc="Processing genomes"):
        assembly_id = get_assembly_from_filename(npy_file)

        # Load activations
        activations = np.load(npy_file)

        # Optionally fix window artifacts
        if args.fix_artifacts:
            window_size, overlap, startup_trim = 50000, 1000, 10
            stride = window_size - overlap
            win_idx = 1
            while True:
                win_start = win_idx * stride
                if win_start >= len(activations):
                    break
                trim_end = min(win_start + startup_trim, len(activations))
                activations[win_start:trim_end] = 0.0
                win_idx += 1

        # Get ground truth
        gt_regions = gt.get(assembly_id, [])
        if not gt_regions:
            for gt_assembly, regions in gt.items():
                if regions and regions[0].get('ncbi_id') == assembly_id:
                    gt_regions = regions
                    break

        # Process
        results = process_genome(activations, assembly_id, gt_regions, args)
        all_results.append(results)

        # Save BED files
        if results['simple_regions']:
            save_bed(results['simple_regions'], bed_dir / f"{assembly_id}_simple.bed", assembly_id)
        if results.get('hdbscan_regions'):
            save_bed(results['hdbscan_regions'], bed_dir / f"{assembly_id}_hdbscan.bed", assembly_id)
        if results.get('optics_regions'):
            save_bed(results['optics_regions'], bed_dir / f"{assembly_id}_optics.bed", assembly_id)

        # Generate plot
        if not args.no_plots:
            plot_comparison(
                activations,
                results['simple_regions'],
                results.get('hdbscan_regions', []),
                gt_regions,
                assembly_id,
                plots_dir / f"{assembly_id}_clusters.png",
                args.threshold
            )

    # Save results
    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    genomes_with_gt = [r for r in all_results if r['gt_regions'] > 0]
    print(f"Genomes with ground truth: {len(genomes_with_gt)} / {len(all_results)}")

    if genomes_with_gt:
        print("\nSimple Clustering Performance:")
        simple_precision = np.mean([r['simple_metrics']['precision'] for r in genomes_with_gt])
        simple_recall = np.mean([r['simple_metrics']['recall'] for r in genomes_with_gt])
        simple_f1 = np.mean([r['simple_metrics']['f1'] for r in genomes_with_gt])
        print(f"  Precision: {simple_precision:.1%}")
        print(f"  Recall:    {simple_recall:.1%}")
        print(f"  F1:        {simple_f1:.1%}")

        if args.use_hdbscan:
            print("\nHDBSCAN Performance:")
            hdb_precision = np.mean([r['hdbscan_metrics'].get('precision', 0) for r in genomes_with_gt])
            hdb_recall = np.mean([r['hdbscan_metrics'].get('recall', 0) for r in genomes_with_gt])
            hdb_f1 = np.mean([r['hdbscan_metrics'].get('f1', 0) for r in genomes_with_gt])
            print(f"  Precision: {hdb_precision:.1%}")
            print(f"  Recall:    {hdb_recall:.1%}")
            print(f"  F1:        {hdb_f1:.1%}")

        if args.use_optics:
            print("\nOPTICS Performance:")
            opt_precision = np.mean([r['optics_metrics'].get('precision', 0) for r in genomes_with_gt])
            opt_recall = np.mean([r['optics_metrics'].get('recall', 0) for r in genomes_with_gt])
            opt_f1 = np.mean([r['optics_metrics'].get('f1', 0) for r in genomes_with_gt])
            print(f"  Precision: {opt_precision:.1%}")
            print(f"  Recall:    {opt_recall:.1%}")
            print(f"  F1:        {opt_f1:.1%}")

        # Count total regions
        total_simple = sum(len(r['simple_regions']) for r in all_results)
        total_gt = sum(r['gt_regions'] for r in genomes_with_gt)
        print(f"\nTotal regions predicted (Simple): {total_simple}")
        print(f"Total ground truth regions:       {total_gt}")

    print(f"\nResults saved to: {output_dir}")
    print(f"BED files: {bed_dir}")
    if not args.no_plots:
        print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
