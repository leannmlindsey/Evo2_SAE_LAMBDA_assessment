#!/bin/bash
# Run Evo2 embedding extraction and analysis pipeline
# Similar to GENERanno downstream task workflow

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evo2-sae

# Default values
MODEL="evo2_7b"
LAYER="blocks.28.mlp.l3"
POOLING="mean"
OUTPUT_DIR="./evo2_embedding_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv_dir)
            CSV_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --pooling)
            POOLING="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CSV_DIR" ]; then
    echo "Usage: $0 --csv_dir /path/to/csv/data [--model evo2_7b] [--layer blocks.28.mlp.l3] [--output_dir ./results]"
    echo ""
    echo "Options:"
    echo "  --csv_dir     Directory containing train.csv, dev.csv, test.csv (required)"
    echo "  --model       Evo2 model to use (default: evo2_7b)"
    echo "  --layer       Layer for embedding extraction (default: blocks.28.mlp.l3)"
    echo "  --pooling     Pooling strategy: mean, first, last, max (default: mean)"
    echo "  --output_dir  Output directory (default: ./evo2_embedding_results)"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "Evo2 Embedding Analysis Pipeline"
echo "============================================================"
echo "CSV dir: $CSV_DIR"
echo "Model: $MODEL"
echo "Layer: $LAYER"
echo "Pooling: $POOLING"
echo "Output: $OUTPUT_DIR"
echo ""

# Run embedding analysis
python exploratory/evo2_embedding_analysis.py \
    --csv_dir $CSV_DIR \
    --output_dir $OUTPUT_DIR \
    --model $MODEL \
    --layer $LAYER \
    --pooling $POOLING \
    --nn_epochs 100 \
    --nn_hidden_dim 256

echo ""
echo "Done! Results in $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - embeddings.npz: Extracted embeddings"
echo "  - embedding_analysis_results.json: All metrics"
echo "  - pca_visualization.png: PCA plot"
echo "  - test_predictions.csv: Test set predictions"
echo "  - three_layer_nn.pt: Trained classifier"
