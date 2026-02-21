#!/bin/bash

# Wrapper script for running ALL inference methods on a test CSV:
#   1. SAE inference (feature 19746 prophage detector)
#   2. 3-Layer NN inference
#   3. Linear Probe inference
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash scripts/wrapper_run_all_inference.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Input/Output ===
INPUT_CSV="/path/to/test.csv"
OUTPUT_DIR="/path/to/output_directory"

# === REQUIRED: Trained Model Artifacts (from evo2_embedding_analysis.py) ===
# Directory containing: three_layer_nn.pt, three_layer_nn_scaler.pkl,
#                       linear_probe.pkl, linear_probe_scaler.pkl
MODEL_DIR="/path/to/results/embedding_analysis/2k"

# === OPTIONAL: Pre-extracted Embeddings ===
# Path to embeddings_pretrained.npz (skips Evo2 model loading for NN/LP)
# Leave empty to extract embeddings live (requires GPU + Evo2 model)
EMBEDDINGS_PATH=""

# === OPTIONAL: Which methods to run (true/false) ===
RUN_SAE="true"
RUN_NN="true"
RUN_LP="true"

# === OPTIONAL: Model Configuration ===
MODEL="evo2_7b"
LAYER="blocks.28.mlp.l3"
POOLING="mean"
BATCH_SIZE="16"

# === OPTIONAL: SAE Parameters ===
FEATURE_IDX="19746"
SAE_MAX_THRESHOLD="0.5"
SAE_MEAN_THRESHOLD="0.1"
SAE_FRACTION_THRESHOLD="0.3"

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate
if [ "${INPUT_CSV}" == "/path/to/test.csv" ]; then
    echo "ERROR: Please set INPUT_CSV"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ]; then
    echo "ERROR: Please set OUTPUT_DIR"
    exit 1
fi

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Get script/repo directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="${SCRIPT_DIR}/.."
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)

echo "============================================================"
echo "Evo2 All-Methods Inference"
echo "============================================================"
echo "  Input CSV:    ${INPUT_CSV}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Model dir:    ${MODEL_DIR}"
echo "  Embeddings:   ${EMBEDDINGS_PATH:-'will extract live'}"
echo ""
echo "  Methods:  SAE=${RUN_SAE}  NN=${RUN_NN}  LP=${RUN_LP}"
echo "============================================================"
echo ""

# Build embeddings flag for NN/LP
EMB_FLAG=""
if [ -n "${EMBEDDINGS_PATH}" ]; then
    EMB_FLAG="--embeddings_path ${EMBEDDINGS_PATH}"
fi

# ---------------------------------------------------------------
# 1. SAE Inference
# ---------------------------------------------------------------
if [ "${RUN_SAE}" == "true" ]; then
    echo "============================================================"
    echo "1. SAE Inference (feature ${FEATURE_IDX})"
    echo "============================================================"

    SAE_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_sae_results.csv"

    python "${REPO_DIR}/src/sae_inference.py" \
        --input_csv "${INPUT_CSV}" \
        --output "${SAE_OUTPUT}" \
        --model "${MODEL}" \
        --feature_idx ${FEATURE_IDX} \
        --max_threshold ${SAE_MAX_THRESHOLD} \
        --mean_threshold ${SAE_MEAN_THRESHOLD} \
        --fraction_threshold ${SAE_FRACTION_THRESHOLD} \
        --batch_size ${BATCH_SIZE} \
        --save_activations

    echo ""
fi

# ---------------------------------------------------------------
# 2. 3-Layer NN Inference
# ---------------------------------------------------------------
if [ "${RUN_NN}" == "true" ]; then
    echo "============================================================"
    echo "2. 3-Layer NN Inference"
    echo "============================================================"

    NN_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_nn_predictions.csv"

    python "${REPO_DIR}/src/evo2_nn_inference.py" \
        --input_csv "${INPUT_CSV}" \
        --classifier_path "${MODEL_DIR}/three_layer_nn.pt" \
        --scaler_path "${MODEL_DIR}/three_layer_nn_scaler.pkl" \
        --output_csv "${NN_OUTPUT}" \
        --model "${MODEL}" \
        --layer "${LAYER}" \
        --pooling "${POOLING}" \
        --batch_size ${BATCH_SIZE} \
        --save_metrics \
        ${EMB_FLAG}

    echo ""
fi

# ---------------------------------------------------------------
# 3. Linear Probe Inference
# ---------------------------------------------------------------
if [ "${RUN_LP}" == "true" ]; then
    echo "============================================================"
    echo "3. Linear Probe Inference"
    echo "============================================================"

    LP_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_lp_predictions.csv"

    python "${REPO_DIR}/src/evo2_lp_inference.py" \
        --input_csv "${INPUT_CSV}" \
        --classifier_path "${MODEL_DIR}/linear_probe.pkl" \
        --scaler_path "${MODEL_DIR}/linear_probe_scaler.pkl" \
        --output_csv "${LP_OUTPUT}" \
        --model "${MODEL}" \
        --layer "${LAYER}" \
        --pooling "${POOLING}" \
        --batch_size ${BATCH_SIZE} \
        --save_metrics \
        ${EMB_FLAG}

    echo ""
fi

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo "============================================================"
echo "All Inference Complete"
echo "============================================================"
echo "Results in: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"/${INPUT_BASENAME}_*
echo "============================================================"
