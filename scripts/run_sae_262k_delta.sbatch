#!/bin/bash
# SAE-only genome-wide extraction on Delta-AI (GH200), using the CORRECT checkpoint
# evo2_7b_262k (the one the Goodfire SAE was trained on). Submit once per window:
#
#   cd /work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val/Evo2_SAE_LAMBDA_assessment
#   sbatch --export=ALL,W=2k scripts/run_sae_262k_delta.sbatch
#   sbatch --export=ALL,W=4k scripts/run_sae_262k_delta.sbatch
#   sbatch --export=ALL,W=8k scripts/run_sae_262k_delta.sbatch
#
# Outputs: <evo2val>/sae_262k_out/<W>/<genome-stem>_sae_results.csv  (per genome)
# Logs:    sae262k_<W>_<jobid>.log
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=sae262k
#SBATCH --output=sae262k_%j.log

set -eo pipefail
: "${W:?set the window via  sbatch --export=ALL,W=2k ...  (2k|4k|8k)}"

# --- fixed deployment paths (see EVO2_on_delta_install.md) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment
GW=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide/$W
OUT=$WD/sae_262k_out/$W
mkdir -p "$OUT" "$WD/triton_cache"

# build the per-window input list (all genome-wide segment CSVs)
ls "$GW"/*.csv > "$OUT/inputs.txt"
echo "[$(date)] window=$W  genomes=$(wc -l < "$OUT/inputs.txt")  out=$OUT  model=evo2_7b_262k"

# container isolation + runtime env (the flags that make Evo2 work on this box)
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
E="--env PYTHONPATH=$LIB --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache"

# SAE only (no NN/LP -> no --model_dir needed); the 262k checkpoint is the fix.
apptainer exec --nv $B $E "$C" python "$A/src/batch_inference.py" \
  --input_list "$OUT/inputs.txt" \
  --output_dir "$OUT" \
  --model evo2_7b_262k \
  --run_sae \
  --feature_idx 19746 \
  --sae_max_threshold 0.5 --sae_mean_threshold 0.1 --sae_fraction_threshold 0.3

echo "[$(date)] done window=$W  -> $OUT"
