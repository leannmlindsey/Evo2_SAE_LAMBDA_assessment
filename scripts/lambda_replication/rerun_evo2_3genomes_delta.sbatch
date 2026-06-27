#!/bin/bash
# rerun_evo2_3genomes_delta.sbatch — re-run EVO2 (LP+NN) and EVO2+SAE genome-wide
# inference for ONLY the 3 corrected genomes, on Delta-AI (GH200), against the
# CORRECTED genome_wide data. EVO2/SAE were originally run on the pre-correction
# (truncated) dataset, so these 3 genomes' predictions were incomplete.
#
# Two passes per window (different checkpoints, by design):
#   SAE   : --run_sae  --model evo2_7b_262k        (the model the Goodfire SAE needs)
#   LP/NN : --run_nn --run_lp --model evo2_7b --model_dir <classifiers>   (embeddings)
#
# The 6 RELABELED genomes do NOT need re-inference (sequences unchanged) — only the
# central re-cluster with corrected labels. Only these 3 REFETCHED genomes changed
# segments, so only they are re-inferred here.
#
# Submit:  sbatch scripts/lambda_replication/rerun_evo2_3genomes_delta.sbatch
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --job-name=evo2_3gw
#SBATCH --output=evo2_3gw_%j.log

set -eo pipefail

# --- Delta EVO2 container setup (proven; see EVO2_on_delta_install.md) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment                       # cloned assessment repo (has src/batch_inference.py)
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
E="--env PYTHONPATH=$LIB --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache"

# --- inputs ---
GW=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide      # CORRECTED data
CLASS=$WD/evo2_classifiers                              # PREREQ: transfer EVO2 LP/NN classifiers here
OUT=$WD/evo2_3genome_rerun                              # outputs land here
# The 3 refetched genome stems (segment files changed):
STEMS=(GCF_000006925.2_ASM692v2_genomic_segments \
       GCF_000007805.1_ASM780v1_genomic_segments \
       GCF_000026645.1_ASM2664v1_genomic_segments)

run() { apptainer exec --nv $B $E "$C" python "$A/src/batch_inference.py" "$@"; }

echo "[$(date)] EVO2 3-genome genome-wide re-run on corrected data"
for w in 2k 4k 8k; do
  echo "===== window $w ====="
  od="$OUT/$w"; mkdir -p "$od/_raw" "$od/evo2_lp" "$od/evo2_nn" "$od/evo2_sae"
  # build input list = the 3 corrected genome CSVs for this window
  ilist="$od/inputs.txt"; : > "$ilist"
  for stem in "${STEMS[@]}"; do
    f="$GW/$w/${stem}.csv"
    [ -f "$f" ] && echo "$f" >> "$ilist" || echo "  WARN missing input: $f"
  done
  echo "  inputs: $(wc -l < "$ilist")"

  # PASS 1 — SAE (evo2_7b_262k)
  echo "  [SAE] evo2_7b_262k ..."
  run --input_list "$ilist" --output_dir "$od/_raw" --model evo2_7b_262k --run_sae \
      --feature_idx 19746 --sae_max_threshold 0.5 --sae_mean_threshold 0.1 --sae_fraction_threshold 0.3

  # PASS 2 — LP + NN (evo2_7b + trained classifiers)
  echo "  [LP/NN] evo2_7b + classifiers $CLASS/$w ..."
  run --input_list "$ilist" --output_dir "$od/_raw" --model evo2_7b --model_dir "$CLASS/$w" \
      --run_nn --run_lp

  # rename batch_inference outputs into canonical genome_wide_<stem>_predictions.csv
  for stem in "${STEMS[@]}"; do
    mv -f "$od/_raw/${stem}_sae_results.csv"             "$od/evo2_sae/genome_wide_${stem}_predictions.csv" 2>/dev/null || true
    mv -f "$od/_raw/${stem}_nn_predictions.csv"          "$od/evo2_nn/genome_wide_${stem}_predictions.csv"  2>/dev/null || true
    mv -f "$od/_raw/${stem}_nn_predictions_metrics.json" "$od/evo2_nn/genome_wide_${stem}_predictions_metrics.json" 2>/dev/null || true
    mv -f "$od/_raw/${stem}_lp_predictions.csv"          "$od/evo2_lp/genome_wide_${stem}_predictions.csv"  2>/dev/null || true
    mv -f "$od/_raw/${stem}_lp_predictions_metrics.json" "$od/evo2_lp/genome_wide_${stem}_predictions_metrics.json" 2>/dev/null || true
  done
  echo "  done $w -> $od/{evo2_lp,evo2_nn,evo2_sae}/"
done
echo "[$(date)] DONE. Send back $OUT/ ; these replace the 3 truncated genomes in the staged EVO2/EVO2_SAE results, then re-cluster."
