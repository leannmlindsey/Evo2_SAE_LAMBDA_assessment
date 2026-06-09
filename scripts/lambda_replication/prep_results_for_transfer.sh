#!/bin/bash
#
# Evo2 LAMBDA — prep results for Globus transfer.
#
# Run this AFTER `check_evo2_status.sh` reports VERDICT: ALL COMPLETE. It packages
# the small result files into one staging folder (+ one tarball) ready to drop into
# a Globus endpoint. It does NOT check completeness itself — run the manifest first.
#
# What it does:
#   1. activates the conda env (so pandas is available for the stager)
#   2. runs stage_results_for_globus.py: copies ONLY the small result files
#      (*_predictions.csv with the DNA 'sequence' column dropped, *_sae_results.csv,
#      *_metrics.json, embedding_analysis_results.json) into STAGE, preserving the
#      <W>/inference/<variant>/ tree. Large *_activations/ .npy and *.npz are NEVER
#      staged.
#   3. makes a single tarball of the staged folder for easy transfer.
#
# Usage:
#   bash scripts/lambda_replication/prep_results_for_transfer.sh [STAGE_DIR]
#
#   STAGE_DIR  (optional) where to write the staged copy + tarball.
#              Default: ~/evo2_lambda_results_to_send

set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
# shellcheck disable=SC1090
[ -f "${CONFIG}" ] && source "${CONFIG}" 2>/dev/null

# --- resolve the results root (same logic as the manifest) --------------------
HARDCODED="/net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/results"
RESULTS="${OUTPUT_DIR:-${HARDCODED}}"
case "${RESULTS}" in /path/to/*) RESULTS="${HARDCODED}";; esac

# Default staging dir: a SIBLING of the results tree (guaranteed writable, and
# outside it as the stager requires). $HOME is unreliable on this server
# (/home/<user> may not exist), so derive from RESULTS rather than ~.
DEFAULT_STAGE="$(dirname "${RESULTS}")/evo2_lambda_results_to_send"
STAGE="${1:-${DEFAULT_STAGE}}"
TARBALL="${STAGE}.tar.gz"

# --- deployment-specific conda (module-free) ----------------------------------
CONDA_BASE="/net/intdev/metagut/lindseylm/miniconda3"

echo "============================================================"
echo "Evo2 LAMBDA — prep results for transfer"
echo "============================================================"
echo "  RESULTS (source): ${RESULTS}"
echo "  STAGE (dest):     ${STAGE}"
echo "  TARBALL:          ${TARBALL}"
echo "============================================================"

if [ ! -d "${RESULTS}" ]; then
    echo "ERROR: results dir not found: ${RESULTS}"
    echo "       (is OUTPUT_DIR set in ${CONFIG}, or are you on the right server?)"
    exit 1
fi

# --- reminder: this does NOT verify completeness ------------------------------
echo ""
echo "NOTE: this does not check that the run finished. Run the manifest first:"
echo "      bash ${SCRIPT_DIR}/check_evo2_status.sh"
echo "      and only continue if it says VERDICT: ALL COMPLETE."
echo ""

# --- 1. activate env ----------------------------------------------------------
if [ -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-evo2-sae}" 2>/dev/null
fi
if ! python -c "import pandas" 2>/dev/null; then
    echo "ERROR: could not import pandas — activate the evo2-sae env first:"
    echo "       source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate evo2-sae"
    exit 1
fi
echo "  python: $(command -v python)"

# --- 2. stage the small result files ------------------------------------------
echo ""
echo ">>> staging small result files (dropping DNA 'sequence' column) ..."
python "${SCRIPT_DIR}/stage_results_for_globus.py" "${RESULTS}" "${STAGE}"
RC=$?
if [ "${RC}" -ne 0 ]; then
    echo "ERROR: staging failed (exit ${RC})"; exit "${RC}"
fi

# --- 3. tar it up -------------------------------------------------------------
echo ""
echo ">>> creating tarball ..."
STAGE_PARENT="$( cd "$( dirname "${STAGE}" )" && pwd )"
STAGE_BASE="$( basename "${STAGE}" )"
tar -czf "${TARBALL}" -C "${STAGE_PARENT}" "${STAGE_BASE}"
RC=$?
if [ "${RC}" -ne 0 ]; then
    echo "ERROR: tar failed (exit ${RC})"; exit "${RC}"
fi
SIZE=$(du -h "${TARBALL}" 2>/dev/null | cut -f1)

echo ""
echo "============================================================"
echo "DONE. Two outputs you can transfer (pick whichever Globus prefers):"
echo "  - folder:  ${STAGE}"
echo "  - tarball: ${TARBALL}   (${SIZE:-?})"
echo "------------------------------------------------------------"
echo "NEXT (Globus): move ONE of the above into your Globus endpoint path,"
echo "then start the transfer to LeAnn from the Globus web UI / 'globus transfer'."
echo ""
echo "NOTE: the large per-position SAE activation arrays (*_activations/*.npy) and"
echo "      cached embeddings (*.npz) are intentionally NOT included (too big). If"
echo "      LeAnn needs those too, Globus the ${RESULTS} tree directly instead."
echo "============================================================"
