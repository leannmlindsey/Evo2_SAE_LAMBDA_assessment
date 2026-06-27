#!/bin/bash
# train_evo2_classifiers_delta.sbatch — regenerate the EVO2 LP/NN classifiers on
# the CURRENT train_val_test (53688/6908/7548), on Delta-AI (GH200).
#
# Follows the SAME proven pattern as rerun_evo2_3genomes_delta.sbatch: load the
# Apptainer container and call the EXISTING python entry point DIRECTLY with
# explicit Delta paths. It does NOT use lambda_replication.conf or
# run_lambda_training.sh (the conf has /path/to placeholders that abort the
# driver — the working SAE scripts avoid it the same way). The training LOGIC is
# unchanged: it's the repo's src/evo2_embedding_analysis.py, just invoked
# directly with the same args run_lambda_training.sh would have built.
#
# Per window it fits the linear probe + 3-layer NN (+ random baseline) on FROZEN
# evo2_7b embeddings and writes linear_probe.pkl, three_layer_nn.pt, *_scaler.pkl,
# embedding_analysis_results.json, then copies the classifiers to
# evo2_classifiers/<w>/ (where rerun_evo2_3genomes_delta.sbatch + the PHROG
# script expect them).
#
# Checkpoints: LP/NN use evo2_7b (here). SAE is zero-shot (evo2_7b_262k, no training).
#
# Submit (any env; runs in --cleanenv container):
#   cd .../evo2val && sbatch Evo2_SAE_LAMBDA_assessment/scripts/lambda_replication/train_evo2_classifiers_delta.sbatch
#   tail -f evo2train_*.log
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=evo2train
#SBATCH --output=evo2train_%j.log
#SBATCH --error=evo2train_%j.log

set -eo pipefail

# --- Delta EVO2 container setup (verbatim from rerun_evo2_3genomes_delta.sbatch) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
# PYTHONPATH adds the repo root (for evo2_embedding_analysis.py's sibling imports)
# alongside the evo2/vortex install in $LIB.
E="--env PYTHONPATH=$LIB:$A --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache"

# --- explicit Delta paths + hyperparameters (the values run_lambda_training.sh
#     would have read from the conf; hardcoded here like the SAE scripts) ---
LB=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1
OUT=$WD/evo2_train_out
CLASS=$WD/evo2_classifiers
MODEL=evo2_7b
LAYER=blocks.28.mlp.l3

run() { apptainer exec --nv $B $E "$C" python "$A/src/evo2_embedding_analysis.py" "$@"; }

cd "$A"
echo "[$(date)] EVO2 classifier training on current train_val_test  node $(hostname)"

for w in 2k 4k 8k; do
  echo "===== train window $w ====="
  csvdir="$LB/train_val_test/$w"
  od="$OUT/$w/embedding"
  mkdir -p "$od"
  [ -f "$csvdir/train.csv" ] || { echo "  ERROR: missing $csvdir/train.csv"; exit 1; }
  echo "  csv: $csvdir   out: $od"

  run --csv_dir "$csvdir" --output_dir "$od" \
      --model "$MODEL" --layer "$LAYER" \
      --batch_size 1 --pooling mean --seed 42 \
      --nn_epochs 100 --nn_hidden_dim 256 --nn_lr 0.001 \
      --include_random_baseline

  # copy classifiers to where rerun_evo2_3genomes_delta.sbatch + the PHROG script expect them
  mkdir -p "$CLASS/$w"
  for f in linear_probe.pkl linear_probe_scaler.pkl three_layer_nn.pt three_layer_nn_scaler.pkl; do
    [ -f "$od/$f" ] && cp -f "$od/$f" "$CLASS/$w/" || echo "  note: $od/$f not present"
  done
  echo "  done $w -> classifiers in $CLASS/$w/"
done

echo "[$(date)] DONE. Classifiers in $CLASS/{2k,4k,8k}/ ; probing JSON in $OUT/<w>/embedding/."
