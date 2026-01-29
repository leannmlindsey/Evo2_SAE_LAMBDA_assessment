# Evo2 SAE Analysis - Prophage Detection & Embedding Extraction

## Your Setup
- 8x H200 GPUs (Hopper architecture)
- No SLURM (direct execution)

## Available Scripts

### SAE-based Prophage Detection
- `run_lambda_batch.py` - Batch process LAMBDA genomes with SAE feature f/19746
- `visualize_prophage_feature.py` - Visualize prophage feature activations
- `prophage_detection.py` - Prophage detection using SAE features

### Embedding Extraction (GENERanno-style)
- `evo2_embedding_extraction.py` - Extract embeddings from Evo2 for sequences
- `evo2_embedding_analysis.py` - Full embedding analysis (linear probe, PCA, 3-layer NN)
- `evo2_inference.py` - Run inference with trained classifiers

## Step-by-Step Instructions

### 1. Setup Environment (one-time)

```bash
# SSH to your node
ssh your-h200-node

# Clone this repo or copy the scripts
mkdir ~/evo2_prophage && cd ~/evo2_prophage

# Run setup (takes ~10-15 minutes)
bash setup_evo2_h200.sh
```

### 2. Inspect SAE Checkpoint (important!)

Before running detection, we need to understand the SAE format:

```bash
conda activate evo2-sae
python inspect_sae_checkpoint.py
```

This will show you:
- The exact file format and keys in the SAE checkpoint
- The correct layer names in Evo2 for extracting embeddings
- Any necessary updates to the detection script

### 3. Test on E. coli First

Download E. coli K12 MG1655 (the genome from the paper):

```bash
# Download E. coli K12 MG1655 reference
mkdir -p ~/test_genomes
cd ~/test_genomes
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
gunzip GCF_000005845.2_ASM584v2_genomic.fna.gz
mv GCF_000005845.2_ASM584v2_genomic.fna ecoli_k12_mg1655.fasta
```

Run detection:

```bash
cd ~/evo2_prophage
python run_prophage_detection.py \
    --genome_dir ~/test_genomes \
    --output_dir ~/test_results \
    --device cuda:0 \
    --save_activations
```

Expected output for E. coli K12:
- Should detect ~9 cryptic prophage regions
- Known prophages: CP4-6, DLP12, e14, rac, Qin, CP4-44, CPS-53, CPZ-55, CP4-57

### 4. Run on Your 80 Genome Test Set

```bash
# Put your genomes in a directory
ls /path/to/your/bacterial_genomes/
# genome1.fasta genome2.fasta ...

# Create ground truth BED file (if you have annotations)
# Format: chrom<TAB>start<TAB>end<TAB>name
# Example:
# CP000948.1    123456    167890    prophage_1
# CP000948.1    234567    289012    prophage_2

# Run detection with evaluation
python run_prophage_detection.py \
    --genome_dir /path/to/your/bacterial_genomes \
    --output_dir ~/prophage_results \
    --ground_truth /path/to/ground_truth.bed \
    --threshold 0.5 \
    --min_length 5000 \
    --save_activations
```

### 5. Multi-GPU Processing (Optional)

For 80 genomes, you can parallelize across your 8 GPUs:

```bash
# Split genomes into 8 batches
ls /path/to/genomes/*.fasta | split -n l/8 - genome_batch_

# Run in parallel (in separate terminals or using GNU parallel)
for i in {0..7}; do
    python run_prophage_detection.py \
        --genome_dir /path/to/genomes \
        --output_dir ~/results_gpu${i} \
        --device cuda:${i} \
        --genome_list genome_batch_a${i} &  # Note: you'd need to implement --genome_list
done
wait

# Merge results
cat ~/results_gpu*/prophage_predictions.csv > ~/all_predictions.csv
```

## Output Files

```
prophage_results/
├── config.json                  # Run configuration
├── prophage_predictions.csv     # Main results table
├── prophage_predictions.bed     # For genome browsers
├── prophage_predictions.gff3    # For annotation tools
├── evaluation_metrics.json      # Metrics vs ground truth
└── *_activations.npy           # Raw activation arrays
```

## Troubleshooting

### "Could not find encoder weights"
Run `inspect_sae_checkpoint.py` and update the `SAEModule.from_pretrained()` method with the correct key names.

### "Could not extract embeddings"
The layer name might be different. Check the output of `inspect_sae_checkpoint.py` for available layer names.

### Out of memory
- Reduce `window_size` in the config (default: 8192)
- Use `evo2_7b` instead of `evo2_40b`



---

## Embedding Extraction & Analysis (GENERanno-style)

These scripts follow the same workflow as GENERanno for downstream task evaluation.

### 1. Extract Embeddings

```bash
# Extract embeddings from a single CSV file
python evo2_embedding_extraction.py \
    --input_csv /path/to/sequences.csv \
    --output_dir ./embeddings \
    --model evo2_7b \
    --layer blocks.28.mlp.l3 \
    --pooling mean
```

Input CSV format:
```csv
sequence,label
ATCGATCG...,1
GCTAGCTA...,0
```

### 2. Full Embedding Analysis

```bash
# Run complete analysis pipeline (linear probe, PCA, 3-layer NN)
python evo2_embedding_analysis.py \
    --csv_dir /path/to/data \
    --output_dir ./results \
    --model evo2_7b \
    --layer blocks.28.mlp.l3 \
    --pooling mean
```

The `csv_dir` should contain:
- `train.csv`
- `dev.csv` (or `val.csv`)
- `test.csv`

Output files:
- `embeddings.npz` - Cached embeddings
- `embedding_analysis_results.json` - All metrics
- `pca_visualization.png` - PCA plot
- `test_predictions.csv` - Test set predictions
- `three_layer_nn.pt` - Trained classifier

### 3. Inference with Trained Classifier

```bash
# Using pre-trained 3-layer NN
python evo2_inference.py \
    --input_csv /path/to/test.csv \
    --classifier_path ./results/three_layer_nn.pt \
    --output_csv predictions.csv \
    --save_metrics
```

### Available Layers for Embedding Extraction

Common layer names for Evo2 7B:
- `blocks.28.mlp.l3` - MLP output at layer 28 (recommended)
- `blocks.26` - Full layer 26 output (used for SAE)
- `blocks.{N}.mlp.l3` - MLP output at any layer N

### Or Use the Pipeline Script

```bash
./run_evo2_embedding_pipeline.sh \
    --csv_dir /path/to/data \
    --model evo2_7b \
    --layer blocks.28.mlp.l3 \
    --output_dir ./results
```

---

## Questions?

- Evo2 issues: https://github.com/ArcInstitute/evo2/issues
- SAE/Goodfire: https://goodfire.ai
