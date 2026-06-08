#!/bin/bash
#
# Evo2 LAMBDA replication — check that all STAGE 1 (embedding-classifier) jobs
# finished. Evo2 has no finetuning: Stage 1 trains the linear probe (evo2_lp) and
# the 3-layer NN (evo2_nn) on frozen embeddings, one job per window. The SAE
# variant is zero-shot and has NO Stage 1 (nothing to check here).
#
# For every window reports:
#   RESULTS  embedding/embedding_analysis_results.json present
#   LP       embedding/linear_probe.pkl + linear_probe_scaler.pkl present
#   NN       embedding/three_layer_nn.pt + three_layer_nn_scaler.pkl present
#   MCC      pretrained vs random MCC (from embedding_analysis_results.json)
#
# Usage:
#   bash scripts/lambda_replication/check_training.sh

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

echo "============================================================"
echo "Evo2 LAMBDA replication — Stage 1 check (evo2_lp + evo2_nn)"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  (evo2_sae is zero-shot — no Stage 1)"
echo "============================================================"
echo ""

TOTAL=0
OK=0

printf "%-4s  %-8s  %-8s  %-8s  %-22s\n" WIN RESULTS LP NN "MCC(pre/rand)"
for W in ${RUN_LENGTHS}; do
    TOTAL=$((TOTAL + 1))
    D="${OUTPUT_DIR}/${W}/embedding"

    if [ -f "${D}/embedding_analysis_results.json" ]; then R=ok; else R=MISSING; fi
    if [ -f "${D}/linear_probe.pkl" ] && [ -f "${D}/linear_probe_scaler.pkl" ]; then LP=ok; else LP=MISSING; fi
    if [ -f "${D}/three_layer_nn.pt" ] && [ -f "${D}/three_layer_nn_scaler.pkl" ]; then NN=ok; else NN=MISSING; fi

    MCC=$(python - "${D}/embedding_analysis_results.json" 2>/dev/null <<'PY'
import json, sys
def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "?"
try:
    d = json.load(open(sys.argv[1]))
    # Prefer pretrained NN MCC vs random NN MCC; fall back to LP / unprefixed keys.
    pre = d.get('pretrained_nn_mcc',
          d.get('pretrained_three_layer_nn_mcc',
          d.get('nn_mcc',
          d.get('pretrained_linear_probe_mcc',
          d.get('linear_probe_mcc')))))
    rnd = d.get('random_nn_mcc',
          d.get('random_three_layer_nn_mcc',
          d.get('random_linear_probe_mcc')))
    print(f"{fmt(pre)} / {fmt(rnd)}")
except Exception:
    print("- / -")
PY
)

    [ "${R}" = ok ] && [ "${LP}" = ok ] && [ "${NN}" = ok ] && OK=$((OK + 1))

    printf "%-4s  %-8s  %-8s  %-8s  %-22s\n" "${W}" "${R}" "${LP}" "${NN}" "${MCC}"
done

echo ""
echo "Healthy: ${OK} / ${TOTAL}"
