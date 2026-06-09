#!/bin/bash
#
# Restart-readiness check for the Evo2 LAMBDA inference job.
#
# Run this AS the account that may need to restart the job (e.g. the colleague
# taking over) — NOT as the owner — to confirm every dependency is reachable:
#   activate the env, import the stack, see the Evo2 weights, read the repo +
#   data, write the output dir, and see a GPU.
#
#   bash scripts/lambda_replication/check_restart_ready.sh
#
# Exits 0 only if all checks pass. Read-only except one tmp write-probe in
# OUTPUT_DIR. Paths below are hardcoded for THIS deployment (owner: lindseylm).
#
# NOTE: do not use `set -e`/`set -u` here — `conda activate` trips on both.

set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

# Deployment-specific locations (from `conda info --base` / `echo $HF_HOME`).
CONDA_BASE="/net/intdev/metagut/lindseylm/miniconda3"
ENV_PATH="${CONDA_BASE}/envs/evo2-sae"
HF_CACHE="/net/intdev/metagut/lindseylm/.cache"

pass=0; fail=0
ok()  { echo "  [ OK ] $*"; pass=$((pass+1)); }
bad() { echo "  [FAIL] $*"; fail=$((fail+1)); }

echo "============================================================"
echo "Evo2 restart-readiness check — running as: $(id -un)"
echo "============================================================"

# 1. conda activation -------------------------------------------------------
if [ -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if conda activate "${ENV_PATH}" 2>/dev/null && [ "${CONDA_PREFIX:-}" = "${ENV_PATH}" ]; then
        ok "activated env: ${ENV_PATH}"
    else
        bad "could not 'conda activate ${ENV_PATH}' — check rX on the env + traverse (namei -l)"
    fi
else
    bad "cannot read ${CONDA_BASE}/etc/profile.d/conda.sh — check rX on miniconda3"
fi

# 2. python stack -----------------------------------------------------------
if python -c "import torch, pandas; from evo2 import Evo2" 2>/tmp/.evo2_imp_$$; then
    cuda=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    ok "imports ok (torch, pandas, evo2) — cuda_available=${cuda}"
else
    bad "python import failed:"; sed 's/^/         /' "/tmp/.evo2_imp_$$"
fi
rm -f "/tmp/.evo2_imp_$$"

# 3. HF cache + Evo2 weights ------------------------------------------------
export HF_HOME="${HF_CACHE}"
if [ -r "${HF_CACHE}" ] && [ -d "${HF_CACHE}/hub" ]; then
    if ls "${HF_CACHE}/hub" 2>/dev/null | grep -qi evo2; then
        ok "HF cache readable + evo2 weights present (HF_HOME=${HF_CACHE})"
    else
        bad "HF cache readable but no 'evo2' under ${HF_CACHE}/hub — weights may be elsewhere"
    fi
else
    bad "cannot read HF cache ${HF_CACHE} — offline model load will fail (check rX)"
fi

# 4. repo + config ----------------------------------------------------------
if [ -r "${CONFIG}" ]; then
    ok "repo + config readable (${CONFIG})"
    # shellcheck disable=SC1090
    source "${CONFIG}"
else
    bad "cannot read ${CONFIG} — check rX on the repo"
fi

# 5. LAMBDA data ------------------------------------------------------------
if [ -n "${LAMBDA_BASE:-}" ] && [ -d "${LAMBDA_BASE}/train_val_test" ]; then
    ok "LAMBDA data readable (${LAMBDA_BASE})"
else
    bad "LAMBDA_BASE not readable: ${LAMBDA_BASE:-<unset>} — check rX (and that it's set on CBB)"
fi

# 6. output dir writable ----------------------------------------------------
if [ -n "${OUTPUT_DIR:-}" ] && [ -d "${OUTPUT_DIR}" ]; then
    probe="${OUTPUT_DIR}/.write_probe_$(id -un)_$$"
    if ( : > "${probe}" ) 2>/dev/null; then
        rm -f "${probe}"
        ok "output dir writable (${OUTPUT_DIR})"
    else
        bad "output dir NOT writable: ${OUTPUT_DIR} — check rwX"
    fi
else
    bad "OUTPUT_DIR missing/unset: ${OUTPUT_DIR:-<unset>}"
fi

# 7. GPU visible ------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${ngpu}" -gt 0 ]; then ok "nvidia-smi sees ${ngpu} GPU(s)"; else bad "nvidia-smi found no GPUs"; fi
else
    bad "nvidia-smi not found on PATH"
fi

echo "============================================================"
echo "${pass} passed, ${fail} failed"
if [ "${fail}" -eq 0 ]; then
    echo "READY — this account can restart the job. To restart:"
    echo "  source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_PATH}"
    echo "  export HF_HOME=${HF_CACHE}"
    echo "  cd ${REPO_ROOT}"
    echo "  tmux new -s evo2restart"
    echo "  PHASE=genome CUDA_VISIBLE_DEVICES=<free-gpu> bash scripts/lambda_replication/run_lambda_inference.sh 2>&1 | tee inference_restart.log"
    echo "  # (set PHASE=diag instead if it was the diagnostics pass that died)"
    echo "============================================================"
    exit 0
else
    echo "NOT READY — fix each [FAIL] above (almost always a missing 'setfacl rX'"
    echo "on the path, or a missing traverse 'x' on a parent dir: namei -l <path>)."
    echo "============================================================"
    exit 1
fi
