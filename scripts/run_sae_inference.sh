#!/bin/bash
# Run SAE inference on short DNA segments (CSV in, CSV out)

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate evo2-sae

# Set paths
INPUT_CSV="/net/intdev/metagut/lindseylm/LAMBDA_DATA/lambda_genomes/gc_control_2k_test.csv"
OUTPUT_CSV="./sae_inference_results.csv"

# Run inference
echo "Starting SAE inference on short segments..."
echo "Input:  $INPUT_CSV"
echo "Output: $OUTPUT_CSV"
echo ""

python src/sae_inference.py \
    --input_csv $INPUT_CSV \
    --output $OUTPUT_CSV \
    --model evo2_7b \
    --max_threshold 0.5 \
    --mean_threshold 0.1 \
    --fraction_threshold 0.3 \
    --save_activations

echo ""
echo "Done! Results in $OUTPUT_CSV"
