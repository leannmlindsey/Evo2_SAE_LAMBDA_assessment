#!/bin/bash
#
# Evo2 LAMBDA replication — check that all STAGE 2 inference outputs landed in the
# harvest-canonical layout. For every (window x variant) reports:
#   <diag>   inference/<variant>/<diag>_predictions.csv for test/fpr/gc_control/fnr
#            (fnr only if FNR_<W> set+exists), with acc & mcc from the metrics JSON.
#            NOTE: evo2_sae writes NO _metrics.json (its CSV holds activation cols +
#            pred_label, not prob_0/prob_1) — its rows show "csv ok (no metrics)".
#   GENOME   genome_wide_*_predictions.csv count vs CSVs in GENOME_WIDE_<W>
#
# Usage:
#   bash scripts/lambda_replication/check_inference.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is empty (check ${CONFIG})"; exit 1
fi
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR not found: ${OUTPUT_DIR}"; exit 1
fi

RUN_LENGTHS="$(echo "${SEGMENT_LENGTHS}" | xargs)"
VARIANTS="${VARIANTS:-evo2_lp evo2_nn evo2_sae}"

# Print "acc=.. mcc=.." from an inference metrics JSON, or a dash if absent.
metrics_line() {
    python - "$1" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    acc = d.get("accuracy"); mcc = d.get("mcc")
    fa = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
    fm = f"{mcc:.4f}" if isinstance(mcc, (int, float)) else "?"
    print(f"acc={fa} mcc={fm}")
except Exception:
    print("-")
PY
}

echo "============================================================"
echo "Evo2 LAMBDA replication — inference check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "============================================================"

for W in ${RUN_LENGTHS}; do
    REPL_W_DIR="${OUTPUT_DIR}/${W}"

    # diagnostics expected for this window: include fnr only if FNR_<W> set+exists.
    DIAGS="test fpr gc_control"
    fnr_var="FNR_${W}"
    if [ -n "${!fnr_var:-}" ] && [ -f "${!fnr_var}" ]; then
        DIAGS="${DIAGS} fnr"
    fi

    # genome-wide: how many CSVs did we point at?
    gw_var="GENOME_WIDE_${W}"
    GW_PATH="${!gw_var:-}"
    GW_EXPECTED=0
    if [ -n "${GW_PATH}" ]; then
        if [ -f "${GW_PATH}" ]; then
            GW_EXPECTED=1
        elif [ -d "${GW_PATH}" ]; then
            shopt -s nullglob
            gw_files=("${GW_PATH}"/*.csv)
            shopt -u nullglob
            GW_EXPECTED="${#gw_files[@]}"
        fi
    fi

    echo ""
    echo "######## window: ${W} ########"

    for VARIANT in ${VARIANTS}; do
        echo ""
        echo "  --- variant: ${VARIANT} ---"
        INF_DIR="${REPL_W_DIR}/inference/${VARIANT}"

        for NAME in ${DIAGS}; do
            CSV="${INF_DIR}/${NAME}_predictions.csv"
            MJSON="${INF_DIR}/${NAME}_predictions_metrics.json"
            if [ -f "${CSV}" ]; then
                if [ -f "${MJSON}" ]; then
                    printf "    %-10s ok   %s\n" "${NAME}" "$(metrics_line "${MJSON}")"
                elif [ "${VARIANT}" = "evo2_sae" ]; then
                    printf "    %-10s ok   (SAE: activation cols, no metrics by design)\n" "${NAME}"
                else
                    printf "    %-10s ok   (no _metrics.json — labels absent?)\n" "${NAME}"
                fi
            else
                printf "    %-10s MISSING\n" "${NAME}"
            fi
        done

        if [ "${GW_EXPECTED}" -gt 0 ]; then
            shopt -s nullglob
            gw_pred=("${INF_DIR}"/genome_wide_*_predictions.csv)
            shopt -u nullglob
            GW_GOT="${#gw_pred[@]}"
            if [ "${GW_GOT}" -eq "${GW_EXPECTED}" ]; then
                GWS=ok
            else
                GWS="INCOMPLETE"
            fi
            printf "    %-10s %s  predictions=%s/%s\n" "genome" "${GWS}" "${GW_GOT}" "${GW_EXPECTED}"
        fi
    done
done
