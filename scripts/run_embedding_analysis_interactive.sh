#!/bin/bash

# Interactive script for running Evo2 embedding analysis WITHOUT sbatch
# Usage: bash run_embedding_analysis_interactive.sh [wrapper_script.sh]
#
# This script reads configuration from wrapper_run_embedding_analysis.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
# Change this path if your wrapper has a different name
WRAPPER_SCRIPT="${1:-wrapper_run_embedding_analysis.sh}"

# Also check in scripts/ directory
if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WRAPPER_SCRIPT="${SCRIPT_DIR}/wrapper_run_embedding_analysis.sh"
fi

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_embedding_analysis_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but just get the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

# Now run the main script logic

echo ""
echo "Evo2 Embedding Analysis (Interactive Mode)"
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

# Ignore user site-packages to avoid conflicts with ~/.local packages
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

# Set defaults for optional parameters
MODEL=${MODEL:-evo2_7b}
LAYER=${LAYER:-blocks.28.mlp.l3}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-}
POOLING=${POOLING:-mean}
SEED=${SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-true}

# Validate required parameters
if [ -z "${CSV_DIR}" ] || [ "${CSV_DIR}" == "/path/to/your/csv/data" ]; then
    echo "ERROR: CSV_DIR is not set or still has placeholder value"
    echo "Please edit the wrapper script: ${WRAPPER_SCRIPT}"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Add to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set output directory
OUTPUT_DIR=${OUTPUT_DIR:-./results/embedding_analysis/$(basename ${CSV_DIR})}
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL}"
echo "  Layer: ${LAYER}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH:-none}"
echo "  Pooling: ${POOLING}"
echo "  Seed: ${SEED}"
echo "  NN epochs: ${NN_EPOCHS}"
echo "  NN hidden dim: ${NN_HIDDEN_DIM}"
echo "  NN learning rate: ${NN_LR}"
echo "  Include random baseline: ${INCLUDE_RANDOM_BASELINE}"
echo "============================================================"
echo ""

# Build optional flags
RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

MAX_LENGTH_FLAG=""
if [ -n "${MAX_LENGTH}" ]; then
    MAX_LENGTH_FLAG="--max_length=${MAX_LENGTH}"
fi

# Run embedding analysis
python src/evo2_embedding_analysis.py \
    --csv_dir="${CSV_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --model="${MODEL}" \
    --layer="${LAYER}" \
    --batch_size=${BATCH_SIZE} \
    --pooling="${POOLING}" \
    --seed=${SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${MAX_LENGTH_FLAG} \
    ${RANDOM_BASELINE_FLAG}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
