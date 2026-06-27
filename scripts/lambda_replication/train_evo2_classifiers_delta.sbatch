#!/bin/bash
# train_evo2_classifiers_delta.sbatch — run the repo's EXISTING Stage-1 training
# (scripts/lambda_replication/run_lambda_training.sh + lambda_replication.conf)
# inside the Delta-AI Apptainer container. This does NOT reimplement training:
# it only activates the .sif and runs the existing driver, which reads the conf
# and calls src/evo2_embedding_analysis.py per window to fit the LP + 3-layer NN
# (+ random baseline) on FROZEN evo2_7b embeddings, writing linear_probe.pkl,
# three_layer_nn.pt, *_scaler.pkl, embedding_analysis_results.json into
# <OUTPUT_DIR>/<w>/embedding/.
#
# WHY: the classifiers we have on the laptop (EVO2_FINAL_RESULTS, Feb) were
# trained on an OLD split (54659/7100/6827); the current split is 53688/6908/7548
# and the canonical Biowulf weights were never transferred. So regenerate on the
# CURRENT train_val_test.
#
# Checkpoints (per the repo docs / rerun_evo2_3genomes_delta.sbatch):
#   LP/NN classifiers  -> conf MODEL=evo2_7b   (this job)
#   SAE (zero-shot)    -> evo2_7b_262k         (no training; inference only)
#
# PREREQ: the conf on Delta must have real LAMBDA_BASE/OUTPUT_DIR (not /path/to).
# Submit:  cd .../evo2val && sbatch Evo2_SAE_LAMBDA_assessment/scripts/lambda_replication/train_evo2_classifiers_delta.sbatch
#          tail -f evo2train_*.log
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

set -uo pipefail

# --- Delta EVO2 container setup (verbatim from rerun_evo2_3genomes_delta.sbatch) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
# SKIP_CONDA_ACTIVATE=1 tells run_lambda_training.sh to use the container python
# instead of activating a (non-existent) conda env.
E="--env PYTHONPATH=$LIB --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache \
   --env SKIP_CONDA_ACTIVATE=1"

echo "[$(date)] EVO2 Stage-1 training (existing run_lambda_training.sh) inside ${C##*/}"
echo "  node: $(hostname)"

apptainer exec --nv $B $E "$C" bash "$A/scripts/lambda_replication/run_lambda_training.sh"
rc=$?

echo "[$(date)] run_lambda_training.sh exit=${rc}"
echo "Classifiers are in <OUTPUT_DIR>/<w>/embedding/ (linear_probe.pkl, three_layer_nn.pt, *_scaler.pkl)."
echo "Next: copy them to ${WD}/evo2_classifiers/<w>/ (or point rerun_evo2_3genomes_delta.sbatch's CLASS there)."
exit ${rc}
