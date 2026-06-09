#!/bin/bash
#
# Evo2 LAMBDA — COMPREHENSIVE results manifest + completion status.
#
#   bash check_evo2_status.sh
#
# Walks EVERY file we intend to transfer as results and reports [ ok ] / [MISS]:
#
#   STAGE 1  per window <W>/embedding/  (pretrained + random surface)
#            embedding_analysis_results.json, test_predictions_{pretrained,random}.csv,
#            linear_probe.pkl(+scaler), three_layer_nn.pt(+scaler),
#            pca_visualization_{pretrained,random}.png, embeddings_*.npz
#
#   STAGE 2  per window x variant (evo2_lp / evo2_nn / evo2_sae):
#     DIAGS     <diag>_predictions.csv  for test/fpr/gc_control/fnr
#               (+ _predictions_metrics.json for lp/nn on the 2-class diags)
#               NOTE: evo2_sae writes NO metrics.json (activation cols, by design).
#     GENOME    genome_wide_*_predictions.csv  count vs CSVs in GENOME_WIDE_<W>
#
# Also reports whether the inference process is still running, and a one-line
# VERDICT. Run anytime; it reads the real files (the tee'd log is unreliable).

set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
# shellcheck disable=SC1090
[ -f "${CONFIG}" ] && source "${CONFIG}" 2>/dev/null

# Results root: prefer OUTPUT_DIR from the conf, else the hardcoded deployment
# path. Guard against the /path/to placeholder (e.g. if run on a laptop checkout).
HARDCODED="/net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/results"
RESULTS="${OUTPUT_DIR:-${HARDCODED}}"
case "${RESULTS}" in /path/to/*) RESULTS="${HARDCODED}";; esac

WINDOWS="$(echo "${SEGMENT_LENGTHS:-2k 4k 8k}" | xargs)"
VARIANTS_INF="evo2_lp evo2_nn evo2_sae"

# 2-class diagnostics get metrics.json from lp/nn; single-class (fpr=bacteria-only,
# fnr=phage-only) legitimately may not — so metrics are optional for those.
TWOCLASS_DIAGS=" test gc_control "

# --- counters + helpers -------------------------------------------------------
present=0; missing=0
req() {  # req <label> <path> : required file
    if [ -e "$2" ]; then printf "      [ ok ] %s\n" "$1"; present=$((present+1))
    else                  printf "      [MISS] %s\n" "$1"; missing=$((missing+1)); fi
}
opt() {  # opt <label> <path> : optional/info file (never counts as MISS)
    if [ -e "$2" ]; then printf "      [ ok ] %s\n" "$1"
    else                  printf "      [ -- ] %s  (optional/absent)\n" "$1"; fi
}

echo "============================================================"
echo "Evo2 LAMBDA — full results manifest   @ $(date)"
echo "============================================================"
echo "  RESULTS: ${RESULTS}"
echo "  WINDOWS: ${WINDOWS}"
echo "============================================================"

# --- is the inference process running, and on which window? --------------------
PID=$(pgrep -f 'src/batch_inference.py' | head -1)
if [ -n "${PID}" ]; then
    cur=$(ps -o args= -p "${PID}" 2>/dev/null | grep -oE 'results/[0-9]+k/' | head -1 | grep -oE '[0-9]+k')
    echo "PROCESS: RUNNING  (pid ${PID}, currently on ${cur:-?} window)"
else
    echo "PROCESS: not running"
fi

# ============================================================================
# STAGE 1 — embedding / pretrained-vs-random (per window)
# ============================================================================
echo ""
echo "############ STAGE 1 — embedding (pretrained + random) ############"
for W in ${WINDOWS}; do
    E="${RESULTS}/${W}/embedding"
    echo ""
    echo "  == ${W}/embedding =="
    req "embedding_analysis_results.json"  "${E}/embedding_analysis_results.json"
    req "linear_probe.pkl"                 "${E}/linear_probe.pkl"
    req "linear_probe_scaler.pkl"          "${E}/linear_probe_scaler.pkl"
    req "three_layer_nn.pt"                "${E}/three_layer_nn.pt"
    req "three_layer_nn_scaler.pkl"        "${E}/three_layer_nn_scaler.pkl"
    req "test_predictions_pretrained.csv"  "${E}/test_predictions_pretrained.csv"
    req "test_predictions_random.csv"      "${E}/test_predictions_random.csv"
    opt "pca_visualization_pretrained.png" "${E}/pca_visualization_pretrained.png"
    opt "pca_visualization_random.png"     "${E}/pca_visualization_random.png"
    opt "embeddings_pretrained.npz"        "${E}/embeddings_pretrained.npz"
    opt "embeddings_random_model.npz"      "${E}/embeddings_random_model.npz"
done

# ============================================================================
# STAGE 2 — inference (diagnostics + genome-wide) per window x variant
# ============================================================================
echo ""
echo "############ STAGE 2 — inference (lp / nn / sae) ############"
for W in ${WINDOWS}; do
    echo ""
    echo "  ======== window: ${W} ========"

    # diagnostics expected for this window (fnr only if FNR_<W> is set + exists)
    DIAGS="test fpr gc_control"
    fnr_var="FNR_${W}"
    if [ -n "${!fnr_var:-}" ] && [ -f "${!fnr_var}" ]; then DIAGS="${DIAGS} fnr"; fi

    # genome-wide: how many CSVs did we point at?
    gw_var="GENOME_WIDE_${W}"
    GW_PATH="${!gw_var:-}"
    GW_EXPECTED=0
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_EXPECTED=1
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob; gwf=("${GW_PATH}"/*.csv); shopt -u nullglob
            GW_EXPECTED="${#gwf[@]}"
        fi
    fi

    for V in ${VARIANTS_INF}; do
        INF="${RESULTS}/${W}/inference/${V}"
        echo ""
        echo "    --- ${V} ---"

        # diagnostics
        for d in ${DIAGS}; do
            req "${d}_predictions.csv"  "${INF}/${d}_predictions.csv"
            if [ "${V}" != "evo2_sae" ]; then
                MJ="${INF}/${d}_predictions_metrics.json"
                case "${TWOCLASS_DIAGS}" in
                    *" ${d} "*) req "${d}_predictions_metrics.json" "${MJ}" ;;
                    *)          opt "${d}_predictions_metrics.json" "${MJ}" ;;
                esac
            fi
        done

        # genome-wide count vs expected
        if [ "${GW_EXPECTED}" -gt 0 ]; then
            shopt -s nullglob
            gwp=("${INF}"/genome_wide_*_predictions.csv)
            shopt -u nullglob
            GW_GOT="${#gwp[@]}"
            if [ "${GW_GOT}" -eq "${GW_EXPECTED}" ]; then
                printf "      [ ok ] genome_wide predictions = %s/%s\n" "${GW_GOT}" "${GW_EXPECTED}"
                present=$((present+1))
            else
                printf "      [MISS] genome_wide predictions = %s/%s\n" "${GW_GOT}" "${GW_EXPECTED}"
                missing=$((missing+1))
            fi
        else
            printf "      [ -- ] genome_wide: no GENOME_WIDE_%s configured\n" "${W}"
        fi
    done
done

# ============================================================================
# Summary + verdict
# ============================================================================
echo ""
echo "============================================================"
echo "MANIFEST: ${present} present, ${missing} missing (required items only)"
echo "------------------------------------------------------------"
if [ -n "${PID}" ]; then
    echo "VERDICT: STILL RUNNING  -- let it continue; missing items are simply"
    echo "         not produced yet. Re-check later."
elif [ "${missing}" -eq 0 ]; then
    echo "VERDICT: ALL COMPLETE  -- every required result is present and the"
    echo "         process has stopped. Next: stage results + transfer."
else
    echo "VERDICT: STOPPED but INCOMPLETE  -- the process ended with ${missing}"
    echo "         required item(s) still MISSING (see [MISS] rows above)."
    echo "         The run likely needs a restart."
fi
echo "============================================================"
