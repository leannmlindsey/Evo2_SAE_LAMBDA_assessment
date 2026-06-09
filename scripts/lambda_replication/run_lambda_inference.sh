#!/bin/bash
#
# Evo2 + SAE LAMBDA_v1 replication — STAGE 2: run all three variants through the
# unified batch_inference.py and rename outputs into the harvest-canonical layout.
# DIRECT execution (single NCBI server, NO SLURM, module-free conda).
#
# batch_inference.py loads Evo2 ONCE and runs SAE + NN + LP on a whole list of
# input CSVs in a single call. So per window we run ONE inference pass that
# processes BOTH the diagnostics and the genome-wide CSVs: (a) build the input
# list, (b) run batch_inference with --run_sae --run_nn --run_lp, then (c) rename
# batch_inference's `<basename>_<method>` outputs into:
#     inference/evo2_lp/<canon>_predictions.csv  (+ _predictions_metrics.json)
#     inference/evo2_nn/<canon>_predictions.csv  (+ _predictions_metrics.json)
#     inference/evo2_sae/<canon>_predictions.csv (SAE CSV has activation cols, no metrics)
#
# Canonical <canon> per input file:
#   test       <- train_val_test/<W>/test.csv
#   fpr        <- fpr_test/<W>/bacteria_segments_<W>.csv          (auto-derived)
#   gc_control <- shuffled_controls/<W>/test_shuffled.csv          (auto-derived)
#   fnr        <- FNR_<W> (if set + exists)
#   genome_wide_<stem>  <- each GENOME_WIDE_<W>/*.csv stem
#
# Missing diagnostics / genome dirs are warned-and-skipped. NO native 50kb scanner
# / nucleotide_evaluation.py / run_lambda_batch.py — genome-level aggregation is
# done CENTRALLY by harvest.
#
# Usage (after run_lambda_training.sh has finished):
#   bash scripts/lambda_replication/run_lambda_inference.sh


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
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# Only run windows that have a Stage-1 embedding dir (lp + nn classifiers).
RUN_LENGTHS=""
for W in ${SEGMENT_LENGTHS}; do
    MODEL_DIR="${OUTPUT_DIR}/${W}/embedding"
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "WARNING: ${MODEL_DIR} missing — skipping ${W}"
        echo "         (run run_lambda_training.sh first)"
        continue
    fi
    if [ ! -f "${MODEL_DIR}/three_layer_nn.pt" ] || [ ! -f "${MODEL_DIR}/linear_probe.pkl" ]; then
        echo "WARNING: ${MODEL_DIR} missing classifier artifacts (three_layer_nn.pt / linear_probe.pkl) — skipping ${W}"
        continue
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${W}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
[ -n "${RUN_LENGTHS}" ] || { echo "ERROR: no windows with completed Stage 1"; exit 1; }

# Where per-window input lists + name maps are written.
LISTDIR="${OUTPUT_DIR}/_inference_lists"
mkdir -p "${LISTDIR}"

MODEL=${MODEL:-evo2_7b}
LAYER=${LAYER:-blocks.28.mlp.l3}
POOLING=${POOLING:-mean}
INF_BATCH_SIZE=${INF_BATCH_SIZE:-1}
FEATURE_IDX=${FEATURE_IDX:-19746}
SAE_MAX_THRESHOLD=${SAE_MAX_THRESHOLD:-0.5}
SAE_MEAN_THRESHOLD=${SAE_MEAN_THRESHOLD:-0.1}
SAE_FRACTION_THRESHOLD=${SAE_FRACTION_THRESHOLD:-0.3}

# PHASE controls which input surfaces run, so diagnostics can be secured before
# the (much larger) genome-wide pass on a single GPU:
#   diag   -> only test/fpr/gc/fnr   (skip genome-wide)
#   genome -> only genome-wide       (skip diagnostics)
#   all    -> both (default; original behavior)
PHASE="${PHASE:-all}"
case "${PHASE}" in diag|genome|all) ;; *) echo "ERROR: PHASE must be diag|genome|all"; exit 1;; esac

echo "============================================================"
echo "Evo2 LAMBDA replication — Stage 2: batch inference (lp + nn + sae)"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  conda env:       ${CONDA_DEFAULT_ENV:-?}   python: $(command -v python)"
echo "  MODEL/LAYER:     ${MODEL} / ${LAYER}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  PHASE:           ${PHASE}   (RUN_SAE=${RUN_SAE:-true})"
echo "============================================================"

NUM_WINDOWS=0

for W in ${RUN_LENGTHS}; do
    echo ""
    echo "--- window: ${W} ---"

    REPL_W_DIR="${OUTPUT_DIR}/${W}"
    MODEL_DIR="${REPL_W_DIR}/embedding"

    INPUT_LIST="${LISTDIR}/${W}_inputs.txt"
    NAME_MAP="${LISTDIR}/${W}_namemap.tsv"
    : > "${INPUT_LIST}"
    : > "${NAME_MAP}"

    # add_input <path> <canon-name>: append to list + name map if the file exists.
    add_input() {
        local path="$1" canon="$2"
        if [ -f "${path}" ]; then
            local base
            base="$(basename "${path}" .csv)"
            echo "${path}" >> "${INPUT_LIST}"
            printf '%s\t%s\n' "${base}" "${canon}" >> "${NAME_MAP}"
        else
            echo "  WARNING: input '${canon}' missing: ${path} — skipping"
        fi
    }

    # --- diagnostics (Surfaces A + B) ---
    if [ "${PHASE}" != "genome" ]; then
        add_input "${LAMBDA_BASE}/train_val_test/${W}/test.csv"                "test"
        add_input "${LAMBDA_BASE}/fpr_test/${W}/bacteria_segments_${W}.csv"    "fpr"
        add_input "${LAMBDA_BASE}/shuffled_controls/${W}/test_shuffled.csv"    "gc_control"

        fnr_var="FNR_${W}"
        FNR_PATH="${!fnr_var:-}"
        if [ -n "${FNR_PATH}" ]; then
            add_input "${FNR_PATH}" "fnr"
        fi
    fi

    # --- genome-wide (Surface C): each CSV stem -> genome_wide_<stem> ---
    gw_var="GENOME_WIDE_${W}"
    GW_PATH="${!gw_var:-}"
    GW_COUNT=0
    if [ "${PHASE}" = "diag" ]; then
        GW_PATH=""
    fi
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            stem="$(basename "${GW_PATH}" .csv)"
            add_input "${GW_PATH}" "genome_wide_${stem}"
            GW_COUNT=1
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            for csv in "${GW_PATH}"/*.csv; do
                stem="$(basename "${csv}" .csv)"
                add_input "${csv}" "genome_wide_${stem}"
                GW_COUNT=$((GW_COUNT + 1))
            done
            shopt -u nullglob
            [ "${GW_COUNT}" -eq 0 ] && \
                echo "  WARNING: ${gw_var}=${GW_PATH} has no *.csv — no genome-wide inputs for ${W}"
        else
            echo "  WARNING: ${gw_var}=${GW_PATH} not a file/dir — skipping genome-wide for ${W}"
        fi
    fi

    NUM_INPUTS=$(grep -cv '^[[:space:]]*$' "${INPUT_LIST}" 2>/dev/null || echo 0)
    if [ "${NUM_INPUTS}" -eq 0 ]; then
        echo "  WARNING: no inputs for ${W} — skipping window"
        continue
    fi
    echo "  inputs:        ${NUM_INPUTS} (genome-wide: ${GW_COUNT})"
    echo "  input list:    ${INPUT_LIST}"
    echo "  name map:      ${NAME_MAP}"
    echo "  model_dir:     ${MODEL_DIR}"

    # Raw batch_inference outputs land in a tmp dir; renamed into canonical layout below.
    TMP_DIR="${REPL_W_DIR}/inference/_raw"
    mkdir -p "${TMP_DIR}"
    for V in evo2_lp evo2_nn evo2_sae; do
        mkdir -p "${REPL_W_DIR}/inference/${V}"
    done

    echo "  SAE: feature_idx=${FEATURE_IDX} max=${SAE_MAX_THRESHOLD} mean=${SAE_MEAN_THRESHOLD} fraction=${SAE_FRACTION_THRESHOLD}"
    echo "  running src/batch_inference.py (loads Evo2 once for the whole list)..."

    # method + save-activations flags, built exactly as the original inference script
    METHOD_FLAGS=""
    [ "${RUN_SAE:-true}" = "true" ] && METHOD_FLAGS="${METHOD_FLAGS} --run_sae"
    [ "${RUN_NN:-true}" = "true" ] && METHOD_FLAGS="${METHOD_FLAGS} --run_nn"
    [ "${RUN_LP:-true}" = "true" ] && METHOD_FLAGS="${METHOD_FLAGS} --run_lp"
    SAVE_ACT_FLAG=""
    [ "${SAVE_ACTIVATIONS:-true}" = "true" ] && SAVE_ACT_FLAG="--save_activations"

    # --- run batch inference (loads Evo2 once; SAE + NN + LP for every input) ---
    python src/batch_inference.py \
        --input_list "${INPUT_LIST}" \
        --output_dir "${TMP_DIR}" \
        --model_dir "${MODEL_DIR}" \
        --model "${MODEL}" \
        --layer "${LAYER}" \
        --pooling "${POOLING}" \
        --batch_size "${INF_BATCH_SIZE}" \
        --feature_idx "${FEATURE_IDX}" \
        --sae_max_threshold "${SAE_MAX_THRESHOLD}" \
        --sae_mean_threshold "${SAE_MEAN_THRESHOLD}" \
        --sae_fraction_threshold "${SAE_FRACTION_THRESHOLD}" \
        ${SAVE_ACT_FLAG} \
        ${METHOD_FLAGS}
    RC=$?
    if [ "${RC}" -ne 0 ]; then
        echo "  ERROR: batch_inference failed for window ${W} (exit ${RC}) — skipping rename"
        continue
    fi

    # --- rename/move into harvest-canonical inference/<variant>/ layout ---
    # NAME_MAP: each line "basename<TAB>canon".
    #   <basename>_sae_results.csv             -> inference/evo2_sae/<canon>_predictions.csv
    #   <basename>_nn_predictions.csv          -> inference/evo2_nn/<canon>_predictions.csv
    #   <basename>_nn_predictions_metrics.json -> inference/evo2_nn/<canon>_predictions_metrics.json
    #   <basename>_lp_predictions.csv          -> inference/evo2_lp/<canon>_predictions.csv
    #   <basename>_lp_predictions_metrics.json -> inference/evo2_lp/<canon>_predictions_metrics.json
    echo "  renaming outputs into canonical inference/<variant>/ ..."

    LP_DIR="${REPL_W_DIR}/inference/evo2_lp"
    NN_DIR="${REPL_W_DIR}/inference/evo2_nn"
    SAE_DIR="${REPL_W_DIR}/inference/evo2_sae"

    # move_if <src> <dst>: move a produced file into place, or warn if absent.
    move_if() {
        local src="$1" dst="$2"
        if [ -f "${src}" ]; then
            mv -f "${src}" "${dst}"
            echo "    ${src##*/}  ->  ${dst}"
        else
            echo "    NOTE: ${src##*/} not produced (labels absent? method skipped?) — skipped"
        fi
    }

    while IFS=$'\t' read -r BASE CANON; do
        [ -z "${BASE}" ] && continue

        # SAE (no metrics.json — schema is activation cols + pred_label).
        move_if "${TMP_DIR}/${BASE}_sae_results.csv"             "${SAE_DIR}/${CANON}_predictions.csv"

        # NN
        move_if "${TMP_DIR}/${BASE}_nn_predictions.csv"          "${NN_DIR}/${CANON}_predictions.csv"
        move_if "${TMP_DIR}/${BASE}_nn_predictions_metrics.json" "${NN_DIR}/${CANON}_predictions_metrics.json"

        # LP
        move_if "${TMP_DIR}/${BASE}_lp_predictions.csv"          "${LP_DIR}/${CANON}_predictions.csv"
        move_if "${TMP_DIR}/${BASE}_lp_predictions_metrics.json" "${LP_DIR}/${CANON}_predictions_metrics.json"
    done < "${NAME_MAP}"

    # Clean up tmp dir if empty (leftover files stay for inspection).
    rmdir "${TMP_DIR}" 2>/dev/null && echo "  removed empty ${TMP_DIR}" || true

    NUM_WINDOWS=$((NUM_WINDOWS + 1))
done

echo ""
echo "Stage 2 done: processed ${NUM_WINDOWS} window(s)."
echo "Results: ${OUTPUT_DIR}/<W>/inference/<variant>/"
echo "Verify with: bash ${SCRIPT_DIR}/check_inference.sh"
