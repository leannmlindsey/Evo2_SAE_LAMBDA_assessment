# Evo2 SAE Prophage Detection

Detect prophage regions in bacterial genomes using a Sparse Autoencoder (SAE) trained on Evo2's internal representations. A single SAE feature (f/19746) activates specifically on prophage sequences, enabling genome-wide prophage scanning without any task-specific training. This repository provides scripts for inference on short segments, genome-wide scanning with windowed processing, post-processing activations into predicted regions, and visualization/benchmarking against the LAMBDA ground truth dataset.

## Background: Sparse Autoencoders (SAEs)

Sparse Autoencoders are a mechanistic interpretability technique that extract interpretable features from neural network internals. An SAE is trained on a model's hidden activations to learn a sparse overcomplete dictionary of features, where each feature corresponds to a specific concept the model has learned.

The SAE used here was trained by [Goodfire](https://goodfire.ai) on [Evo2](https://github.com/ArcInstitute/evo2) (Arc Institute), a 7-billion-parameter DNA language model. Among the 32,768 learned features, **feature f/19746 activates specifically on prophage sequences** — viral DNA integrated into bacterial genomes.

**SAE Architecture:**

| Property | Value |
|----------|-------|
| Type | BatchTopK tied-weight SAE |
| Input dimension | 4,096 (Evo2 hidden dimension) |
| SAE dimension | 32,768 (8x expansion) |
| TopK | 64 (only top 64 features active per position) |
| Hook location | Layer 26 (`blocks-26`) |
| HuggingFace repo | [`Goodfire/Evo-2-Layer-26-Mixed`](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed) |
| Weights file | `sae-layer26-mixed-expansion_8-k_64.pt` |

The extraction process is:
1. Tokenize a DNA sequence with Evo2's tokenizer
2. Forward pass through Evo2, capturing hidden states at layer 26 via PyTorch hooks
3. Encode the layer-26 activations through the SAE: `features = sae.encode(hidden_states)`
4. Extract prophage feature: `prophage_signal = features[:, 19746]`

## Data Availability

The LAMBDA benchmark dataset — genome FASTA files, per-window segment CSVs
(`train_val_test`, `fpr_test`, `fnr_test`, `shuffled_controls`, `genome_wide`), and
the genome-wide ground truth (`Lambda_Genome_Wide_Evaluation_Test_Set.csv`) — is
distributed separately from this code repository:

> **Zenodo:** _DOI to be added_ (`10.5281/zenodo.XXXXXXX`)

Download and unpack it, then pass the relevant paths to the scripts (every script
takes data locations as arguments, e.g. `--ground_truth`, `--fasta_dir`,
`--input_csv` — none are hardcoded to the repo). The ground truth file is **not**
shipped in this repo; it lives in the dataset download.

## Setup & Dependencies

### Evo2 (GPU inference)

Evo2 must be installed for running inference. Follow the instructions at [github.com/ArcInstitute/evo2](https://github.com/ArcInstitute/evo2). SAE weights are downloaded automatically from HuggingFace on first run via `huggingface_hub`.

Key dependencies for inference:
- `torch`
- `evo2`
- `huggingface_hub`
- `numpy`
- `tqdm`

### Analysis-only environment (no GPU)

For post-processing, clustering, visualization, and PDF report generation (no GPU required):

```bash
conda env create -f environment.yml
conda activate evo2-sae
```

This installs: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `tqdm`, `pillow` (Python 3.10).

## Quick Start: Inference on Short Segments

**Script:** `src/sae_inference.py`

Run SAE feature extraction on a CSV of short (~2 kb) DNA segments and get per-segment activation metrics.

> **⚠️ Critical — use the `evo2_7b_262k` checkpoint for SAE.** The Goodfire SAE
> (`Evo-2-Layer-26-Mixed`, feature f/19746) was trained on **`evo2_7b_262k`**
> activations and only fires correctly on that checkpoint. Running SAE on plain
> `evo2_7b` silently produces a near-dead signal (the prophage feature drops out of
> the SAE's BatchTopK-64 selection — prophage `fraction_firing` collapses from ~0.36
> to ~0). `sae_inference.py` now defaults to `evo2_7b_262k`; do not override it unless
> your SAE was trained on a different model. (This applies only to the SAE — the
> linear-probe / 3-layer-NN embedding surfaces use `evo2_7b`, which is correct.)

### Input CSV format

| Column | Description |
|--------|-------------|
| `segment_id` | Unique identifier for each segment |
| `sequence` | DNA sequence string |
| `label` | Ground truth label (1 = prophage, 0 = non-prophage) |
| `source` | Source annotation |

### Output CSV columns

The input columns are preserved, with these columns appended:

| Column | Description |
|--------|-------------|
| `max_activation` | Maximum SAE activation across all positions |
| `mean_activation` | Mean SAE activation across all positions |
| `fraction_firing` | Fraction of positions with activation > 0 |
| `pred_label` | Predicted label (0 or 1) |

**Prediction logic:** `pred_label = 1` if **ANY** of the following conditions is met:
- `max_activation > max_threshold`
- `mean_activation > mean_threshold`
- `fraction_firing > fraction_threshold`

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | *(required)* | Input CSV with columns: segment_id, sequence, label, source |
| `--output` | *(required)* | Output CSV path |
| `--model` | `evo2_7b_262k` | Evo2 checkpoint the SAE was trained on — **keep this** (see warning above) |
| `--feature_idx` | `19746` | SAE feature index |
| `--max_threshold` | `0.5` | Max activation threshold for pred_label |
| `--mean_threshold` | `0.1` | Mean activation threshold for pred_label |
| `--fraction_threshold` | `0.3` | Fraction firing threshold for pred_label |
| `--save_activations` | off | Save per-segment `.npy` activation arrays |
| `--batch_size` | `1` | Batch size (reserved for future use) |

### Running with wrapper scripts (recommended)

For batch inference on multiple input files, use the GENERanno-style wrapper scripts:

| Script | Purpose |
|--------|---------|
| `scripts/wrapper_run_batch_sae_inference.sh` | **Edit this file.** Set your input file list, output dir, and parameters. |
| `scripts/submit_batch_sae_inference.sh` | Reads the input list and submits one SLURM job per file. Called by the wrapper. |
| `scripts/run_sae_inference.sh` | SLURM batch script for a single input file. Called by the submit script. |
| `scripts/run_batch_sae_inference_interactive.sh` | Runs all files sequentially on the current node (no SLURM). Sources config from the wrapper. |

**Step 1: Create an input file list**

Create a text file with one input CSV path per line:

```
/data/prophage/dataset1.csv
/data/prophage/dataset2.csv
/data/prophage/dataset3.csv
```

**Step 2: Edit the wrapper script**

Open `scripts/wrapper_run_batch_sae_inference.sh` and set the configuration:

```bash
INPUT_LIST="/path/to/input_files.txt"
OUTPUT_DIR="/path/to/output_directory"
MODEL="evo2_7b"
FEATURE_IDX="19746"
MAX_THRESHOLD="0.5"
MEAN_THRESHOLD="0.1"
FRACTION_THRESHOLD="0.3"
SAVE_ACTIVATIONS="true"
```

**Step 3: Run**

```bash
# Option A: Submit one SLURM job per input file
bash scripts/wrapper_run_batch_sae_inference.sh

# Option B: Run all files sequentially on the current node
bash scripts/run_batch_sae_inference_interactive.sh
```

Output files are named `<input_basename>_sae_results.csv` in the output directory.

### Running the Python script directly

For a single file without the wrapper:

```bash
python src/sae_inference.py \
    --input_csv gc_control_2k_test.csv \
    --output gc_control_2k_results.csv \
    --max_threshold 0.5 \
    --save_activations
```

## Classification Metrics

**Script:** `src/calculate_metrics.py`

Calculate accuracy, precision, recall, F1, MCC, FPR, and FNR from any result CSV with `label` and `pred_label` columns. Runs automatically after each file in the batch inference workflow, and can also be run standalone on existing results.

```bash
# Single file
python src/calculate_metrics.py --input results.csv

# Multiple files
python src/calculate_metrics.py --input results1.csv results2.csv

# All result CSVs in a directory (with aggregate metrics across files)
python src/calculate_metrics.py --input_dir ./output_directory

# Save metrics to JSON
python src/calculate_metrics.py --input_dir ./output_directory --output_json metrics.json
```

When multiple files are provided, per-file metrics are printed followed by aggregate metrics across all files.

## Genome-Wide Scanning

**Script:** `src/run_lambda_batch.py`

Process full bacterial genomes (~1-10 Mb) by sliding a window across the genome and extracting SAE feature activations at each position.

### Windowing strategy

- **Window size:** 50,000 bp (default)
- **Overlap:** 1,000 bp between adjacent windows
- **Stride:** 49,000 bp (window_size - overlap)
- **Startup trim:** First 10 positions of each window (except the first) are zeroed to remove model startup artifacts
- **Overlap resolution:** MAX pooling — in overlap regions, the maximum activation from either window is kept, preserving the sparse prophage signal

### Output

| File | Description |
|------|-------------|
| `<assembly_id>_activations.npy` | Per-position activation array for each genome |
| `all_results.json` | Detailed per-genome results with region stats |
| `summary.csv` | Summary table (assembly, sequence_length, ground_truth_count, total_firing, max_activation, fraction_in_gt) |

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fasta_dir` | *(required)* | Directory containing FASTA files (`.fna` or `.fasta`) |
| `--ground_truth` | *(required)* | Ground truth CSV file |
| `--output_dir` | `./lambda_results` | Output directory |
| `--model` | `evo2_7b_262k` | SAE checkpoint — **keep this** (the SAE only fires correctly on `evo2_7b_262k`) |
| `--window_size` | `50000` | Window size for processing |
| `--startup_trim` | `10` | Positions to trim from window start to remove artifacts |

### Example

```bash
python src/run_lambda_batch.py \
    --fasta_dir /path/to/LAMBDA/FASTA \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./lambda_results_7b
```

## Post-Processing: Activations to Predicted Regions

**Script:** `src/cluster_activations.py`

Convert per-position SAE activation arrays into discrete predicted prophage regions using a pipeline of normalize, threshold, cluster, filter, and merge.

### Pipeline

1. **Normalize** — z-score normalization: `(x - mean) / std` per genome
2. **Threshold** — select positions with normalized activation > threshold
3. **Cluster** — group nearby above-threshold positions (max_gap parameter)
4. **Filter** — remove regions smaller than min_region_size
5. **Merge** — combine regions within merge_distance of each other

### Best parameters (from optimization)

| Parameter | Value |
|-----------|-------|
| Normalization | z-score (`--normalize zscore`) |
| Threshold | 7.0 (`--threshold 7.0`) |
| Max gap | 300 bp (`--max_gap 300`) |
| Merge distance | 5,000 bp (`--merge_distance 5000`) |
| Min region size | 1,000 bp (`--min_region_size 1000`) |

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | *(required)* | Directory with `*_activations.npy` files |
| `--ground_truth` | *(required)* | Ground truth CSV file |
| `--output_dir` | `./cluster_results` | Output directory |
| `--threshold` | `0.3` | Activation threshold |
| `--normalize` | `none` | Normalization method: `none`, `zscore`, `robust_zscore`, `percentile`, `local_baseline`, `minmax`, `quantile` |
| `--norm_window` | `10000` | Window size for `local_baseline` normalization |
| `--max_gap` | `100` | Max gap between positions in simple clustering (bp) |
| `--min_region_size` | `1000` | Minimum region size (bp) |
| `--merge_distance` | `3000` | Merge regions within this distance (bp) |
| `--use_hdbscan` | off | Also run HDBSCAN clustering |
| `--use_optics` | off | Also run OPTICS clustering |
| `--use_mws` | off | Use Moving Window Sum algorithm |
| `--fix_artifacts` | off | Fix window boundary artifacts before clustering |
| `--no_plots` | off | Skip generating plots |

### Example (best parameters)

```bash
python src/cluster_activations.py \
    --results_dir ./lambda_results_7b \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./clustering_results_best \
    --normalize zscore \
    --threshold 7.0 \
    --max_gap 300 \
    --merge_distance 5000 \
    --min_region_size 1000
```

Output includes BED files per genome, comparison plots, accuracy bins (high/medium/low by F1), and `clustering_results.json`.

## Nucleotide-Level Prophage Evaluation (from Segments)

**Script:** `src/nucleotide_evaluation.py`

When SAE inference is run on overlapping 2 kb segments (from `sae_inference.py --save_activations`), this script stitches the per-segment activation arrays back into genome-wide activation tracks and evaluates prophage detection at nucleotide resolution. This leverages the full per-position activation signal rather than reducing to binary segment labels.

### Pipeline

```
Segment .npy files ──→ Stitch (MAX pool) ──→ Genome-wide activation track
                                                      │
                                    Normalize (optional) → Threshold → Cluster → Merge → Filter
                                                      │
                                              Predicted prophage regions
                                                      │
Ground truth (prophage_start/end) ──────────────→ Nucleotide-level evaluation
                                                      │
                                     Per-genome CSV + Aggregate JSON + BED file + Plots
```

### Prerequisites

1. Run `sae_inference.py` with `--save_activations` on a segment CSV that includes `seq_id`, `start`, `end`, `prophage_start`, `prophage_end` columns
2. The activation `.npy` files and the output CSV are inputs to this script

### Input format

The input CSV (output from `sae_inference.py`) must contain:

| Column | Description |
|--------|-------------|
| `segment_id` | Unique segment identifier (auto-generated if not in original input) |
| `seq_id` | Genome/contig identifier |
| `start` | 0-based start position of segment in genome |
| `end` | 0-based end position of segment in genome |
| `prophage_start` | Ground truth prophage start (may be empty) |
| `prophage_end` | Ground truth prophage end (may be empty) |

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | *(required)* | Output CSV from sae_inference.py |
| `--activations_dir` | *(required)* | Directory with per-segment `.npy` files |
| `--output_dir` | `./nucleotide_eval_results` | Output directory |
| `--output_prefix` | `nucleotide_eval` | Prefix for output files |
| `--feature_idx` | `19746` | SAE feature index (for multi-feature `.npy` files) |
| `--threshold` | `0.5` | Activation threshold for calling positions |
| `--normalization` | `none` | Normalization: `none`, `zscore`, `robust_zscore`, `percentile`, `local_baseline`, `minmax`, `quantile` |
| `--norm_window` | `10000` | Window size for `local_baseline` normalization |
| `--max_gap` | `100` | Max gap between positions for clustering (bp) |
| `--merge_distance` | `3000` | Max distance for merging regions (bp) |
| `--min_region_size` | `1000` | Minimum prophage region size (bp) |
| `--adaptive_threshold` | off | Use per-genome adaptive threshold |
| `--adaptive_method` | `mad` | Adaptive threshold method: `mad`, `std`, `percentile` |
| `--adaptive_k` | `3.0` | Adaptive threshold sensitivity |
| `--plot` | off | Generate per-genome activation track plots |

### Running with wrapper scripts (recommended)

| Script | Purpose |
|--------|---------|
| `scripts/wrapper_run_nucleotide_evaluation.sh` | **Edit this file.** Set paths and parameters, then run to submit a SLURM job. |
| `scripts/run_nucleotide_evaluation.sh` | SLURM batch script (CPU only, no GPU). Called by the wrapper. |
| `scripts/run_nucleotide_evaluation_interactive.sh` | Runs directly on the current node (no SLURM). Sources config from the wrapper. |

```bash
# Option A: Submit as a SLURM batch job
bash scripts/wrapper_run_nucleotide_evaluation.sh

# Option B: Run interactively
bash scripts/run_nucleotide_evaluation_interactive.sh
```

### Running the Python script directly

```bash
python src/nucleotide_evaluation.py \
    --input_csv results/sae_results.csv \
    --activations_dir results/sae_results_activations/ \
    --output_dir ./nucleotide_eval_results \
    --threshold 0.5 \
    --max_gap 100 \
    --merge_distance 3000 \
    --min_region_size 1000 \
    --plot
```

### Output files

| File | Description |
|------|-------------|
| `<prefix>_per_genome.csv` | Per-genome metrics: precision, recall, F1, MCC, Jaccard |
| `<prefix>_aggregate.json` | Micro and macro-averaged metrics across all genomes |
| `<prefix>_predicted.bed` | Predicted prophage regions in BED format |
| `plots/<seq_id>_activation_track.png` | Per-genome activation track with predictions and ground truth (with `--plot`) |

### Key design decisions

- **MAX pooling for overlaps**: In overlap regions between adjacent segments, the maximum activation is kept. This preserves sparse prophage signal — a strong activation in one segment window shouldn't be diluted by the overlap.
- **Reuses functions from `cluster_activations.py`**: Normalization, clustering, merging, filtering, and metric calculation are imported rather than duplicated.
- **CPU-only**: No GPU needed — all computation is numpy-based on already-extracted activations.
- **Both micro and macro averaging**: Micro averages over nucleotides across all genomes; macro averages per-genome metrics.

## Visualization & PDF Reports

### Step 1: Generate activation plots

**Script:** `src/generate_lambda_plots.py`

Creates PNG plots from `.npy` activation files overlaid with ground truth prophage regions.

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | *(required)* | Directory containing `*_activations.npy` files |
| `--ground_truth` | *(required)* | Ground truth CSV file |
| `--output_dir` | `<results_dir>/plots` | Output directory for plots |
| `--threshold` | `0.5` | Activation threshold for highlighting |
| `--dpi` | `150` | DPI for output images |

```bash
python src/generate_lambda_plots.py \
    --results_dir ./lambda_results_7b \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./lambda_plots
```

### Step 2: Analyze performance factors

**Script:** `src/analyze_performance_factors.py`

Computes per-genome statistics (GC content, genome size, taxonomy) and correlates them with detection performance (F1, precision, recall, MCC). Generates comparison plots and a `genome_stats.csv`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustering_results` | *(required)* | Path to `clustering_results.json` |
| `--ground_truth` | *(required)* | Ground truth CSV file |
| `--fasta_dir` | *(required)* | Directory containing FASTA files |
| `--output_dir` | `./performance_analysis` | Output directory |

```bash
python src/analyze_performance_factors.py \
    --clustering_results ./clustering_results_best/clustering_results.json \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --fasta_dir /path/to/LAMBDA/FASTA \
    --output_dir ./performance_analysis
```

### Step 3: Create categorized PDF reports

**Script:** `src/create_categorized_pdfs.py`

Generates categorized PDF reports (high, medium, low performance) with plots arranged 3 per page. Can use genome stats for full annotations or work standalone with plot summaries.

| Argument | Default | Description |
|----------|---------|-------------|
| `--plots_dir` | *(required)* | Directory containing plot PNG files |
| `--genome_stats` | — | `genome_stats.csv` from `analyze_performance_factors.py` (preferred) |
| `--summary_json` | — | `plot_summary.json` or `clustering_results.json` (fallback) |
| `--clustering_results` | — | `clustering_results.json` for P/R/MCC metrics |
| `--ground_truth` | — | Ground truth CSV for taxonomy |
| `--fasta_dir` | — | FASTA directory for GC calculation |
| `--output_dir` | `./categorized_pdfs` | Output directory |
| `--high_thresh` | `0.7` | Threshold for high performance |
| `--low_thresh` | `0.3` | Threshold for low performance |

```bash
# With full stats (preferred)
python src/create_categorized_pdfs.py \
    --plots_dir ./clustering_results_best/plots \
    --genome_stats ./performance_analysis/genome_stats.csv

# Standalone with clustering results
python src/create_categorized_pdfs.py \
    --plots_dir ./clustering_results_best/plots \
    --clustering_results ./clustering_results_best/clustering_results.json
```

## Embedding Evaluation

Measure the quality of Evo2 embeddings by training downstream classifiers (linear probe + 3-layer NN) and comparing against a random baseline. This follows the GENERanno evaluation pattern to compute **embedding power** — the performance gain of pretrained embeddings over random noise.

The pipeline:
1. Extract embeddings from the pretrained Evo2 model (or load cached embeddings)
2. Train a linear probe (logistic regression) and a 3-layer NN on the pretrained embeddings
3. Generate random Gaussian embeddings of the same shape
4. Train the same classifiers on random embeddings
5. Compute embedding power = pretrained metrics − random metrics
6. Output all results to a single JSON file, with separate PCA plots and prediction CSVs

### Input format

A directory containing `train.csv`, `dev.csv` (or `val.csv`), and `test.csv`. Each CSV must have `sequence` and `label` columns.

### Running with wrapper scripts (recommended)

Three scripts follow the GENERanno three-tier pattern:

| Script | Purpose |
|--------|---------|
| `scripts/wrapper_run_embedding_analysis.sh` | **Edit this file.** Set your paths and parameters, then run it to submit a SLURM job. |
| `scripts/run_embedding_analysis.sh` | SLURM batch script (`#SBATCH` headers). Called by the wrapper — do not edit unless changing HPC resources. |
| `scripts/run_embedding_analysis_interactive.sh` | Runs directly on the current node (no SLURM). Sources config from the wrapper. |

**Step 1: Edit the wrapper script**

Open `scripts/wrapper_run_embedding_analysis.sh` and set the configuration variables:

```bash
# === REQUIRED ===
export CSV_DIR="/data/lindleys/prophage_segments"    # your actual data path

# === OPTIONAL (defaults shown) ===
export MODEL="evo2_7b"                   # evo2_7b or evo2_40b
export LAYER="blocks.28.mlp.l3"          # layer for embedding extraction
export OUTPUT_DIR=""                      # leave empty for auto: ./results/embedding_analysis/<dataset>
export BATCH_SIZE="1"
export MAX_LENGTH=""                      # leave empty for no truncation
export POOLING="mean"                    # mean, first, last, max
export SEED="42"
export NN_EPOCHS="100"
export NN_HIDDEN_DIM="256"
export NN_LR="0.001"
export INCLUDE_RANDOM_BASELINE="true"    # "true" or "false"
```

**Step 2: Run**

```bash
# Option A: Submit as a SLURM batch job
bash scripts/wrapper_run_embedding_analysis.sh

# Option B: Run interactively on the current node (e.g., from an sinteractive session)
bash scripts/run_embedding_analysis_interactive.sh
```

The interactive script reads the same configuration from the wrapper, so you only edit one file. You can also point it at a different wrapper:

```bash
bash scripts/run_embedding_analysis_interactive.sh scripts/my_custom_wrapper.sh
```

### Running the Python script directly

If you prefer to bypass the wrapper scripts:

```bash
python src/evo2_embedding_analysis.py \
    --csv_dir /path/to/csv/data \
    --output_dir ./results/embedding_analysis \
    --model evo2_7b \
    --layer blocks.28.mlp.l3 \
    --include_random_baseline
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv_dir` | *(required)* | Directory containing train.csv, dev.csv, test.csv |
| `--output_dir` | `./results/embedding_analysis` | Output directory |
| `--model` | `evo2_7b` | Evo2 model (`evo2_7b` or `evo2_40b`) |
| `--layer` | `blocks.28.mlp.l3` | Layer for embedding extraction |
| `--pooling` | `mean` | Pooling strategy: `mean`, `first`, `last`, `max` |
| `--batch_size` | `1` | Batch size for extraction |
| `--max_length` | none | Maximum sequence length (truncate longer) |
| `--seed` | `42` | Random seed |
| `--nn_epochs` | `100` | Training epochs for 3-layer NN |
| `--nn_hidden_dim` | `256` | Hidden dimension for 3-layer NN |
| `--nn_lr` | `0.001` | Learning rate for 3-layer NN |
| `--include_random_baseline` | off | Include random embedding baseline |

### Output files

| File | Description |
|------|-------------|
| `embeddings_pretrained.npz` | Cached pretrained embeddings (train/val/test) |
| `embedding_analysis_results.json` | All metrics (pretrained, random, embedding power) |
| `pca_visualization_pretrained.png` | PCA plot of pretrained embeddings |
| `pca_visualization_random.png` | PCA plot of random embeddings (with `--include_random_baseline`) |
| `test_predictions_pretrained.csv` | Pretrained model predictions |
| `test_predictions_random.csv` | Random baseline predictions (with `--include_random_baseline`) |
| `three_layer_nn.pt` | Trained 3-layer NN classifier |

### Metrics

- **Linear probe**: Logistic regression on embeddings (accuracy, precision, recall, F1, MCC, AUC, sensitivity, specificity)
- **3-layer NN**: Neural network classifier on embeddings (same metrics)
- **Silhouette score**: Measures class separation in embedding space
- **PCA explained variance**: How much variance the first 2 principal components capture
- **Embedding power** (with `--include_random_baseline`): Difference between pretrained and random metrics — quantifies how much the model's learned representations improve over random noise

## Reproducing the LAMBDA Benchmark

End-to-end pipeline from raw FASTA files to benchmark results:

```bash
# Step 1: Genome-wide scanning (requires GPU + Evo2)
python src/run_lambda_batch.py \
    --fasta_dir /path/to/LAMBDA/FASTA \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./lambda_results_7b

# Step 2: Cluster activations into predicted regions (CPU only)
python src/cluster_activations.py \
    --results_dir ./lambda_results_7b \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./clustering_results_best \
    --normalize zscore \
    --threshold 7.0 \
    --max_gap 300 \
    --merge_distance 5000 \
    --min_region_size 1000

# Step 3: Generate plots (CPU only)
python src/generate_lambda_plots.py \
    --results_dir ./lambda_results_7b \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --output_dir ./lambda_plots

# Step 4: Analyze performance factors (CPU only)
python src/analyze_performance_factors.py \
    --clustering_results ./clustering_results_best/clustering_results.json \
    --ground_truth /path/to/Lambda_Genome_Wide_Evaluation_Test_Set.csv \
    --fasta_dir /path/to/LAMBDA/FASTA \
    --output_dir ./performance_analysis

# Step 5: Create PDF reports (CPU only)
python src/create_categorized_pdfs.py \
    --plots_dir ./clustering_results_best/plots \
    --genome_stats ./performance_analysis/genome_stats.csv \
    --output_dir ./categorized_pdfs
```

### Expected results

| Metric | Value |
|--------|-------|
| MCC | 0.599 |
| Precision | 71.9% |
| Recall | 52.0% |

These are nucleotide-level metrics averaged across LAMBDA genomes with ground truth, using the best parameters (z-score normalization, threshold 7.0, max_gap 300, merge_distance 5000, min_region_size 1000).

## Repository Structure

```
├── src/                              # Core pipeline scripts
│   ├── sae_inference.py              # SAE inference on short DNA segments (CSV in, CSV out)
│   ├── nucleotide_evaluation.py      # Nucleotide-level prophage evaluation from segment activations
│   ├── run_lambda_batch.py           # Genome-wide scanning with windowed SAE feature extraction
│   ├── cluster_activations.py        # Convert activation arrays to predicted prophage regions
│   ├── generate_lambda_plots.py      # Generate PNG activation plots per genome
│   ├── analyze_performance_factors.py # Correlate performance with genome stats (GC, size, taxonomy)
│   ├── create_categorized_pdfs.py    # Categorized PDF reports (high/medium/low performance)
│   ├── calculate_metrics.py          # Classification metrics from result CSVs (standalone + integrated)
│   ├── evo2_embedding_extraction.py  # Standalone embedding extraction from Evo2
│   └── evo2_embedding_analysis.py    # Embedding evaluation with random baseline comparison
├── scripts/                          # Bash wrappers & setup
│   ├── setup.sh                      # Environment setup script
│   ├── run_lambda_batch.sh           # Batch processing wrapper
│   ├── wrapper_run_batch_sae_inference.sh       # SAE inference: user config + submit
│   ├── submit_batch_sae_inference.sh            # SAE inference: submit one job per file
│   ├── run_sae_inference.sh                     # SAE inference: SLURM single-file script
│   ├── run_batch_sae_inference_interactive.sh   # SAE inference: interactive batch runner
│   ├── wrapper_run_nucleotide_evaluation.sh     # Nucleotide eval: user config + submit
│   ├── run_nucleotide_evaluation.sh             # Nucleotide eval: SLURM batch script (CPU only)
│   ├── run_nucleotide_evaluation_interactive.sh # Nucleotide eval: interactive runner
│   ├── wrapper_run_embedding_analysis.sh        # Embedding eval: user config + submit
│   ├── run_embedding_analysis.sh                # Embedding eval: SLURM batch script
│   └── run_embedding_analysis_interactive.sh    # Embedding eval: interactive runner
├── exploratory/                      # Non-core development & profiling scripts
├── experiments/                      # Parameter sweep experiments
│   └── normalization/                # Normalization & threshold experiments
├── environment.yml                   # Conda environment for analysis-only (no GPU)
├── METHODS_SUMMARY.md                # Technical summary of SAE extraction and post-processing
└── PROFILING_SUMMARY.md              # Evo2 profiling results and optimization notes
```

## References & Links

- **Evo2** — Arc Institute: [github.com/ArcInstitute/evo2](https://github.com/ArcInstitute/evo2)
- **SAE weights** — Goodfire: [huggingface.co/Goodfire/Evo-2-Layer-26-Mixed](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed)
- **LAMBDA dataset** — Genome-wide prophage evaluation benchmark, Lindsey et al. 2026 (in preparation)
