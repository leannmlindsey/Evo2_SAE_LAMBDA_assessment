#!/bin/bash

# Interactive script for running all inference methods (SAE, NN, LP)
# Usage: bash scripts/run_all_inference_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_all_inference.sh (or specify another)
# and runs inference directly on the current node (sequentially for each input file).
# For each input file, it runs SAE, NN, and LP inference as configured.

# Source the wrapper to get all the environment variables
WRAPPER_SCRIPT="${1:-wrapper_run_all_inference.sh}"

# Also check in scripts/ directory
if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_run_all_inference.sh"
fi

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_all_inference_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Extract variable assignments from wrapper (lines with = that aren't comments)
source <(grep -E '^[A-Z_]+=' "${WRAPPER_SCRIPT}" | grep -v '^#')

echo ""
echo "Evo2 All-Methods Batch Inference (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load CUDA/12.8 2>/dev/null || true

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi 2>/dev/null || echo "No GPU detected (nvidia-smi not found)"
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Validate configuration
if [ "${INPUT_LIST}" == "/path/to/input_files.txt" ] || [ -z "${INPUT_LIST}" ]; then
    echo "ERROR: INPUT_LIST is not set properly in wrapper"
    exit 1
fi

if [ "${OUTPUT_DIR}" == "/path/to/output_directory" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is not set properly in wrapper"
    exit 1
fi

if [ "${MODEL_DIR}" == "/path/to/results/embedding_analysis/2k" ] || [ -z "${MODEL_DIR}" ]; then
    echo "ERROR: MODEL_DIR is not set properly in wrapper"
    exit 1
fi

# Verify input list exists
if [ ! -f "${INPUT_LIST}" ]; then
    echo "ERROR: Input list file not found: ${INPUT_LIST}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set defaults
MODEL=${MODEL:-evo2_7b}
LAYER=${LAYER:-blocks.28.mlp.l3}
POOLING=${POOLING:-mean}
BATCH_SIZE=${BATCH_SIZE:-16}
FEATURE_IDX=${FEATURE_IDX:-19746}
SAE_MAX_THRESHOLD=${SAE_MAX_THRESHOLD:-0.5}
SAE_MEAN_THRESHOLD=${SAE_MEAN_THRESHOLD:-0.1}
SAE_FRACTION_THRESHOLD=${SAE_FRACTION_THRESHOLD:-0.3}
SAVE_ACTIVATIONS=${SAVE_ACTIVATIONS:-true}
RUN_SAE=${RUN_SAE:-true}
RUN_NN=${RUN_NN:-true}
RUN_LP=${RUN_LP:-true}

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Input list:    ${INPUT_LIST}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Model dir:     ${MODEL_DIR}"
echo "  Model:         ${MODEL}"
echo "  Layer:         ${LAYER}"
echo "  Pooling:       ${POOLING}"
echo "  Batch size:    ${BATCH_SIZE}"
echo ""
echo "  Methods:  SAE=${RUN_SAE}  NN=${RUN_NN}  LP=${RUN_LP}"
echo ""
echo "  SAE config:"
echo "    Feature idx:         ${FEATURE_IDX}"
echo "    Max threshold:       ${SAE_MAX_THRESHOLD}"
echo "    Mean threshold:      ${SAE_MEAN_THRESHOLD}"
echo "    Fraction threshold:  ${SAE_FRACTION_THRESHOLD}"
echo "    Save activations:    ${SAVE_ACTIVATIONS}"
echo "============================================================"
echo ""

# Verify model artifacts exist
if [ "${RUN_NN}" == "true" ]; then
    if [ ! -f "${MODEL_DIR}/three_layer_nn.pt" ]; then
        echo "ERROR: NN model not found: ${MODEL_DIR}/three_layer_nn.pt"
        exit 1
    fi
    if [ ! -f "${MODEL_DIR}/three_layer_nn_scaler.pkl" ]; then
        echo "ERROR: NN scaler not found: ${MODEL_DIR}/three_layer_nn_scaler.pkl"
        exit 1
    fi
fi

if [ "${RUN_LP}" == "true" ]; then
    if [ ! -f "${MODEL_DIR}/linear_probe.pkl" ]; then
        echo "ERROR: LP model not found: ${MODEL_DIR}/linear_probe.pkl"
        exit 1
    fi
    if [ ! -f "${MODEL_DIR}/linear_probe_scaler.pkl" ]; then
        echo "ERROR: LP scaler not found: ${MODEL_DIR}/linear_probe_scaler.pkl"
        exit 1
    fi
fi

# Build save_activations flag
SAVE_ACT_FLAG=""
if [ "${SAVE_ACTIVATIONS}" == "true" ]; then
    SAVE_ACT_FLAG="--save_activations"
fi

# Count input files (non-empty, non-comment lines)
NUM_FILES=$(grep -c -v -E '^[[:space:]]*$|^[[:space:]]*#' "${INPUT_LIST}" || echo 0)
echo "Found ${NUM_FILES} input files to process"
echo ""

# Process each input file sequentially
COUNT=0
while IFS= read -r INPUT_CSV || [ -n "${INPUT_CSV}" ]; do
    # Skip empty lines and comments
    if [[ -z "${INPUT_CSV}" ]] || [[ "${INPUT_CSV}" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    INPUT_CSV=$(echo "${INPUT_CSV}" | xargs)

    # Validate input file exists
    if [ ! -f "${INPUT_CSV}" ]; then
        echo "WARNING: Input file not found, skipping: ${INPUT_CSV}"
        continue
    fi

    COUNT=$((COUNT + 1))
    INPUT_BASENAME=$(basename "${INPUT_CSV}" .csv)

    echo ""
    echo "############################################################"
    echo "Processing file ${COUNT}/${NUM_FILES}: ${INPUT_BASENAME}"
    echo "  Input: ${INPUT_CSV}"
    echo "############################################################"

    # -----------------------------------------------------------
    # SAE Inference
    # -----------------------------------------------------------
    if [ "${RUN_SAE}" == "true" ]; then
        SAE_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_sae_results.csv"
        echo ""
        echo "------------------------------------------------------------"
        echo "[${COUNT}/${NUM_FILES}] SAE Inference: ${INPUT_BASENAME}"
        echo "  Output: ${SAE_OUTPUT}"
        echo "------------------------------------------------------------"

        python src/sae_inference.py \
            --input_csv "${INPUT_CSV}" \
            --output "${SAE_OUTPUT}" \
            --model "${MODEL}" \
            --feature_idx ${FEATURE_IDX} \
            --max_threshold ${SAE_MAX_THRESHOLD} \
            --mean_threshold ${SAE_MEAN_THRESHOLD} \
            --fraction_threshold ${SAE_FRACTION_THRESHOLD} \
            --batch_size ${BATCH_SIZE} \
            ${SAVE_ACT_FLAG}
    fi

    # -----------------------------------------------------------
    # 3-Layer NN Inference (extracts embeddings, saves for LP reuse)
    # -----------------------------------------------------------
    CACHED_EMB="${OUTPUT_DIR}/${INPUT_BASENAME}_embeddings.npz"

    if [ "${RUN_NN}" == "true" ]; then
        NN_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_nn_predictions.csv"
        echo ""
        echo "------------------------------------------------------------"
        echo "[${COUNT}/${NUM_FILES}] NN Inference: ${INPUT_BASENAME}"
        echo "  Output: ${NN_OUTPUT}"
        echo "------------------------------------------------------------"

        # Save embeddings so LP can reuse them
        SAVE_EMB_FLAG=""
        if [ "${RUN_LP}" == "true" ]; then
            SAVE_EMB_FLAG="--save_embeddings ${CACHED_EMB}"
        fi

        python src/evo2_nn_inference.py \
            --input_csv "${INPUT_CSV}" \
            --classifier_path "${MODEL_DIR}/three_layer_nn.pt" \
            --scaler_path "${MODEL_DIR}/three_layer_nn_scaler.pkl" \
            --output_csv "${NN_OUTPUT}" \
            --model "${MODEL}" \
            --layer "${LAYER}" \
            --pooling "${POOLING}" \
            --batch_size ${BATCH_SIZE} \
            --save_metrics \
            ${SAVE_EMB_FLAG}
    fi

    # -----------------------------------------------------------
    # Linear Probe Inference (reuses cached embeddings if available)
    # -----------------------------------------------------------
    if [ "${RUN_LP}" == "true" ]; then
        LP_OUTPUT="${OUTPUT_DIR}/${INPUT_BASENAME}_lp_predictions.csv"
        echo ""
        echo "------------------------------------------------------------"
        echo "[${COUNT}/${NUM_FILES}] LP Inference: ${INPUT_BASENAME}"
        echo "  Output: ${LP_OUTPUT}"
        echo "------------------------------------------------------------"

        # Use cached embeddings from NN step if available
        EMB_FLAG=""
        if [ -f "${CACHED_EMB}" ]; then
            EMB_FLAG="--embeddings_path ${CACHED_EMB}"
        fi

        python src/evo2_lp_inference.py \
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

        # Clean up cached embeddings
        if [ -f "${CACHED_EMB}" ]; then
            rm "${CACHED_EMB}"
        fi
    fi

done < "${INPUT_LIST}"

echo ""
echo "============================================================"
echo "All Inference Complete"
echo "============================================================"
echo "Processed ${COUNT} files"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Job completed at: $(date)"
echo "============================================================"
