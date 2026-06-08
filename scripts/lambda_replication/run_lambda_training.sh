#!/bin/bash
#
# Evo2 + SAE LAMBDA_v1 replication — STAGE 1: train the embedding classifiers.
# DIRECT execution (single NCBI server, NO SLURM, module-free conda).
#
# Evo2 does NOT finetune. The only trainable units are the linear probe (evo2_lp)
# and the 3-layer NN (evo2_nn) fit on FROZEN Evo2 embeddings. The SAE variant
# (evo2_sae) is zero-shot and needs NO Stage 1. For each window in SEGMENT_LENGTHS
# this runs ONE src/evo2_embedding_analysis.py IN-PROCESS (one Evo2 load per
# window, sequential) that trains BOTH classifiers (linear_probe.pkl +
# three_layer_nn.pt + scalers) and writes embedding_analysis_results.json
# (pretrained-vs-random metrics, Surface D). No seed sweep, no select_best_model.
#
# Usage:
#   1. Edit lambda_replication.conf — set LAMBDA_BASE, OUTPUT_DIR (confirm MODEL/LAYER).
#   2. bash scripts/lambda_replication/run_lambda_training.sh
#   3. bash scripts/lambda_replication/check_training.sh
#   4. bash scripts/lambda_replication/run_lambda_inference.sh


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi

# --- activation block (module-free conda; server is online) -------------------
source "${SCRIPT_DIR}/lambda_replication.conf"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evo2-sae}"
export PYTHONNOUSERSITE=1
[ -z "${CUDA_HOME:-}" ] && export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

[ -n "${HF_HOME:-}" ] && export HF_HOME

# --- validate -----------------------------------------------------------------

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to the /path/to placeholder"
    exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"
    exit 1
}
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi
if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL is empty (set evo2_7b / evo2_40b in ${CONFIG})"; exit 1
fi

# Validate per-window train_val_test dirs exist before running anything.
RUN_LENGTHS=""
for W in ${SEGMENT_LENGTHS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${W}"
    if [ ! -d "${LDIR}" ]; then
        echo "WARNING: ${LDIR} not found — skipping ${W}"; continue
    fi
    [ -f "${LDIR}/train.csv" ] || { echo "ERROR: ${LDIR}/train.csv not found"; exit 1; }
    [ -f "${LDIR}/test.csv" ]  || { echo "ERROR: ${LDIR}/test.csv not found"; exit 1; }
    if [ ! -f "${LDIR}/dev.csv" ] && [ ! -f "${LDIR}/val.csv" ]; then
        echo "ERROR: ${LDIR} must contain dev.csv or val.csv"; exit 1
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${W}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
if [ -z "${RUN_LENGTHS}" ]; then
    echo "ERROR: no runnable windows after validation"; exit 1
fi

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "Evo2 LAMBDA replication — Stage 1: train embedding classifiers (lp + nn)"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  REPO_ROOT:       ${REPO_ROOT}"
echo "  conda env:       ${CONDA_DEFAULT_ENV:-?}   python: $(command -v python)"
echo "  MODEL/LAYER:     ${MODEL} / ${LAYER}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  POOLING/SEED:    ${POOLING} / ${EMB_SEED}"
echo "  NN params:       epochs=${NN_EPOCHS} hidden=${NN_HIDDEN_DIM} lr=${NN_LR} (random baseline ALWAYS on)"
echo "  (evo2_sae is zero-shot — no Stage 1)"
echo "============================================================"

NUM_OK=0
NUM_FAIL=0

# Conditional flags built exactly as the original run_embedding_analysis script.
RANDOM_BASELINE_FLAG=""
[ "${INCLUDE_RANDOM_BASELINE:-true}" = "true" ] && RANDOM_BASELINE_FLAG="--include_random_baseline"
MAX_LENGTH_FLAG=""
[ -n "${EMB_MAX_LENGTH:-}" ] && MAX_LENGTH_FLAG="--max_length=${EMB_MAX_LENGTH}"
PRETRAINED_EMB_FLAG=""    # default empty -> fresh extraction (new output dir)
[ -n "${PRETRAINED_EMBEDDINGS:-}" ] && PRETRAINED_EMB_FLAG="--pretrained_embeddings=${PRETRAINED_EMBEDDINGS}"

for W in ${RUN_LENGTHS}; do
    CSV_DIR="${LAMBDA_BASE}/train_val_test/${W}"
    OUT_DIR="${OUTPUT_DIR}/${W}/embedding"
    mkdir -p "${OUT_DIR}"

    echo ""
    echo "--- window: ${W} ---"
    echo "    csv dir:    ${CSV_DIR}"
    echo "    output dir: ${OUT_DIR}"
    echo "    running src/evo2_embedding_analysis.py (loads Evo2 once)..."

    python src/evo2_embedding_analysis.py \
        --csv_dir "${CSV_DIR}" \
        --output_dir "${OUT_DIR}" \
        --model "${MODEL}" \
        --layer "${LAYER}" \
        --batch_size "${EMB_BATCH_SIZE}" \
        --pooling "${POOLING}" \
        --seed "${EMB_SEED}" \
        --nn_epochs "${NN_EPOCHS}" \
        --nn_hidden_dim "${NN_HIDDEN_DIM}" \
        --nn_lr "${NN_LR}" \
        ${MAX_LENGTH_FLAG} \
        ${PRETRAINED_EMB_FLAG} \
        ${RANDOM_BASELINE_FLAG}
    RC=$?

    if [ "${RC}" -ne 0 ]; then
        echo "    ERROR: window ${W} failed (exit ${RC})"
        NUM_FAIL=$((NUM_FAIL + 1))
        continue
    fi
    if [ -f "${OUT_DIR}/embedding_analysis_results.json" ]; then
        echo "    wrote ${OUT_DIR}/embedding_analysis_results.json"
        NUM_OK=$((NUM_OK + 1))
    else
        echo "    WARNING: ${OUT_DIR}/embedding_analysis_results.json not found — analysis may have failed"
        NUM_FAIL=$((NUM_FAIL + 1))
    fi
done

echo ""
echo "Stage 1 done: ${NUM_OK} ok, ${NUM_FAIL} failed."
echo "Verify with:  bash ${SCRIPT_DIR}/check_training.sh"
echo "Then run:     bash ${SCRIPT_DIR}/run_lambda_inference.sh"
