#!/bin/bash
#SBATCH --job-name=evo2_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=evo2_emb_%j.out
#SBATCH --error=evo2_emb_%j.err

# Batch script for Evo2 embedding analysis
# Usage: sbatch run_embedding_analysis.sh
#
# Required environment variables:
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv

echo "============================================================"
echo "Evo2 Embedding Analysis"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

# Activate conda environment
source activate evo2-sae

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
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
PRETRAINED_EMBEDDINGS=${PRETRAINED_EMBEDDINGS:-}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

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
echo "  Pretrained embeddings: ${PRETRAINED_EMBEDDINGS:-none}"
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

PRETRAINED_EMB_FLAG=""
if [ -n "${PRETRAINED_EMBEDDINGS}" ]; then
    PRETRAINED_EMB_FLAG="--pretrained_embeddings=${PRETRAINED_EMBEDDINGS}"
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
    ${PRETRAINED_EMB_FLAG} \
    ${RANDOM_BASELINE_FLAG}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
