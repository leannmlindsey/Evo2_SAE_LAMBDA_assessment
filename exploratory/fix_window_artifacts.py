#!/usr/bin/env python3
"""
Fix window boundary artifacts in existing activation files.

The batch processing creates artifacts where positions 3-7 of each new window
have artificially high activations (~1.0) due to model startup effects.

This script removes those artifacts by zeroing out the first N positions of
each window boundary (except the very first window).
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def fix_artifacts(acts, window_size=50000, overlap=1000, startup_trim=10):
    """Remove window boundary artifacts from activation array."""
    stride = window_size - overlap
    fixed = acts.copy()

    # For each window except the first, zero out the startup positions
    win_idx = 1
    while True:
        win_start = win_idx * stride
        if win_start >= len(acts):
            break

        trim_end = min(win_start + startup_trim, len(acts))
        fixed[win_start:trim_end] = 0.0
        win_idx += 1

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Fix window boundary artifacts")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with *_activations.npy files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: overwrite in place)")
    parser.add_argument("--window_size", type=int, default=50000)
    parser.add_argument("--overlap", type=int, default=1000)
    parser.add_argument("--startup_trim", type=int, default=10, help="Positions to zero out at each window start")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(results_dir.glob("*_activations.npy"))
    print(f"Found {len(npy_files)} activation files")
    print(f"Window size: {args.window_size}, overlap: {args.overlap}, trim: {args.startup_trim}")

    for npy_file in tqdm(npy_files, desc="Fixing artifacts"):
        acts = np.load(npy_file)
        fixed = fix_artifacts(acts, args.window_size, args.overlap, args.startup_trim)

        output_path = output_dir / npy_file.name
        np.save(output_path, fixed)

    print(f"Done! Fixed files saved to {output_dir}")


if __name__ == "__main__":
    main()
