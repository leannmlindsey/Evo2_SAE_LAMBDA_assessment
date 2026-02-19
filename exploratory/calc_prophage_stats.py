#!/usr/bin/env python3
"""Calculate prophage size statistics from ground truth CSV."""

import csv
import numpy as np

csv_path = "/net/intdev/metagut/lindseylm/LAMBDA_DATA/lambda_genomes/Lambda_Genome_Wide_Evaluation_Test_Set.csv"
output_path = "./prophage_size_stats.txt"

sizes = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        size = int(row['end']) - int(row['start'])
        sizes.append(size)

sizes = np.array(sizes)

with open(output_path, 'w') as out:
    out.write("PROPHAGE SIZE STATISTICS\n")
    out.write("=" * 40 + "\n\n")
    out.write(f"Total prophage regions: {len(sizes)}\n\n")
    out.write("Size Statistics:\n")
    out.write(f"  Mean:   {np.mean(sizes)/1000:.1f} kb\n")
    out.write(f"  Median: {np.median(sizes)/1000:.1f} kb\n")
    out.write(f"  Std:    {np.std(sizes)/1000:.1f} kb\n")
    out.write(f"  Min:    {np.min(sizes)/1000:.1f} kb ({np.min(sizes):,} bp)\n")
    out.write(f"  Max:    {np.max(sizes)/1000:.1f} kb ({np.max(sizes):,} bp)\n\n")

    out.write("Size Distribution:\n")
    out.write(f"  < 10 kb:  {np.sum(sizes < 10000)}\n")
    out.write(f"  10-20 kb: {np.sum((sizes >= 10000) & (sizes < 20000))}\n")
    out.write(f"  20-40 kb: {np.sum((sizes >= 20000) & (sizes < 40000))}\n")
    out.write(f"  40-60 kb: {np.sum((sizes >= 40000) & (sizes < 60000))}\n")
    out.write(f"  > 60 kb:  {np.sum(sizes >= 60000)}\n")

print(f"Stats written to {output_path}")
