#!/bin/bash
# train_evo2_classifiers_delta.sbatch — regenerate the EVO2 LP/NN classifiers on
# the CURRENT train_val_test (53688/6908/7548), on Delta-AI (GH200).
#
# PER-WINDOW ARRAY JOB: --array=0-2 runs 2k / 4k / 8k as 3 INDEPENDENT parallel
# tasks (one GPU each), so wall-clock = the slowest single window (8k), not the
# sum. Each task is self-contained.
#
# Follows the working SAE-script pattern: load the Apptainer container and call
# the EXISTING src/evo2_embedding_analysis.py DIRECTLY with explicit Delta paths
# (no conf, no run_lambda_training.sh). Training logic is unchanged. Each task
# fits LP + 3-layer NN (+ random baseline) on FROZEN evo2_7b embeddings, writes
# linear_probe.pkl / three_layer_nn.pt / *_scaler.pkl / embedding_analysis_results.json,
# and copies the classifiers to evo2_classifiers/<w>/ (where the GW rerun + PHROG
# scripts expect them).
#
# Checkpoints: LP/NN use evo2_7b (here). SAE is zero-shot (evo2_7b_262k, no training).
#
# Submit (any env; runs in --cleanenv container):
#   cd .../evo2val && sbatch Evo2_SAE_LAMBDA_assessment/scripts/lambda_replication/train_evo2_classifiers_delta.sbatch
#   tail -f evo2train_*_*.log
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --job-name=evo2train
#SBATCH --output=evo2train_%A_%a.log
#SBATCH --error=evo2train_%A_%a.log

set -eo pipefail

WINDOWS=(2k 4k 8k)
W="${WINDOWS[$SLURM_ARRAY_TASK_ID]}"

# --- Delta EVO2 container setup (verbatim from rerun_evo2_3genomes_delta.sbatch) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
E="--env PYTHONPATH=$LIB:$A --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache"

# --- explicit Delta paths + hyperparameters (hardcoded like the SAE scripts) ---
LB=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1
OUT=$WD/evo2_train_out_$W          # per-window dir: parallel windows never collide
CLASS=$WD/evo2_classifiers
MODEL=evo2_7b
LAYER=blocks.28.mlp.l3

# The random baseline is NOT needed for the classifiers (GW/PHROG use the
# pretrained LP/NN). Default = pretrained-only (fast: extracts pretrained only).
# Set WITH_RANDOM=1 for the random-baseline pass; it reuses the cached
# embeddings_pretrained.npz in $od (so it only extracts the random half).
RB_FLAG=""
[ "${WITH_RANDOM:-0}" = "1" ] && RB_FLAG="--include_random_baseline"

run() { apptainer exec --nv $B $E "$C" python "$A/src/evo2_embedding_analysis.py" "$@"; }

cd "$A"
echo "[$(date)] EVO2 classifier training — window $W (array task $SLURM_ARRAY_TASK_ID)  node $(hostname)"

csvdir="$LB/train_val_test/$W"
od="$OUT/embedding"
mkdir -p "$od"
[ -f "$csvdir/train.csv" ] || { echo "  ERROR: missing $csvdir/train.csv"; exit 1; }
echo "  csv: $csvdir   out: $od"

run --csv_dir "$csvdir" --output_dir "$od" \
    --model "$MODEL" --layer "$LAYER" \
    --batch_size 1 --pooling mean --seed 42 \
    --nn_epochs 100 --nn_hidden_dim 256 --nn_lr 0.001 \
    ${RB_FLAG}

# copy classifiers to where rerun_evo2_3genomes_delta.sbatch + the PHROG script expect them
mkdir -p "$CLASS/$W"
for f in linear_probe.pkl linear_probe_scaler.pkl three_layer_nn.pt three_layer_nn_scaler.pkl; do
  [ -f "$od/$f" ] && cp -f "$od/$f" "$CLASS/$W/" || echo "  note: $od/$f not present"
done
echo "[$(date)] DONE window $W -> classifiers in $CLASS/$W/"
