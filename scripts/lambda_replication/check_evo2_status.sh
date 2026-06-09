#!/bin/bash
#
# Evo2 LAMBDA inference — completion status. Run anytime (the live log is
# unreliable due to tee buffering; this reads the actual process + output files).
#
#   bash check_evo2_status.sh
#
# Tells you: is it still running, what each window has finished, and a one-line
# VERDICT of whether the whole run is DONE and ready to stage for Globus.

set -o pipefail

RESULTS="/net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/results"
WINDOWS="2k 4k 8k"
DIAGS="test fpr gc_control fnr"

echo "============================================================"
echo "Evo2 inference status   @ $(date)"
echo "============================================================"

# --- is the process running, and on which window? ---
PID=$(pgrep -f 'src/batch_inference.py' | head -1)
if [ -n "${PID}" ]; then
    cur=$(ps -o args= -p "${PID}" 2>/dev/null | grep -oE 'results/[0-9]+k/' | head -1 | grep -oE '[0-9]+k')
    echo "PROCESS: RUNNING  (pid ${PID}, currently on ${cur:-?} window)"
else
    echo "PROCESS: not running"
fi
echo

# --- per-window completion (a window is done+renamed when its diagnostics land
#     in inference/evo2_lp/ ; while in progress they sit unrenamed in _raw/) ---
all_done=1
for W in ${WINDOWS}; do
    LP="${RESULTS}/${W}/inference/evo2_lp"
    have=""; miss=""
    for d in ${DIAGS}; do
        if [ -f "${LP}/${d}_predictions.csv" ]; then have="${have} ${d}"; else miss="${miss} ${d}"; fi
    done
    gw=$(ls "${LP}"/genome_wide_*_predictions.csv 2>/dev/null | wc -l)
    raw=$(ls "${RESULTS}/${W}/inference/_raw"/*.csv 2>/dev/null | wc -l)
    if [ -z "${miss}" ]; then
        printf "  %-3s DONE     diagnostics:[%s ]  genome_wide=%s\n" "${W}" "${have}" "${gw}"
    else
        all_done=0
        printf "  %-3s pending  have:[%s ]  missing:[%s ]  genome_wide=%s  (_raw csvs:%s)\n" \
               "${W}" "${have}" "${miss}" "${gw}" "${raw}"
    fi
done

echo
echo "------------------------------------------------------------"
if [ -z "${PID}" ] && [ "${all_done}" -eq 1 ]; then
    echo "VERDICT: ALL DONE  -- process stopped and all 3 windows renamed."
    echo "         Next: stage results + Globus."
elif [ -n "${PID}" ]; then
    echo "VERDICT: STILL RUNNING  -- let it continue, re-check later."
else
    echo "VERDICT: STOPPED but INCOMPLETE  -- the process ended before finishing"
    echo "         every window (see 'pending' rows above). May need a restart."
fi
echo "------------------------------------------------------------"
