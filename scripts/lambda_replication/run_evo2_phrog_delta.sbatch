#!/bin/bash
# run_evo2_phrog_delta.sbatch — EVO2 (LP+NN) and EVO2+SAE inference on the PHROG
# diagnostic set for the per-PHROG-category sensitivity table, on Delta-AI (GH200).
# Mirrors rerun_evo2_3genomes_delta.sbatch (same container setup + two passes).
#
# PHROG diagnostic = phage_annotated_segments_2k.csv (CDS segments, PHROG-annotated;
# 2k ONLY — distinct from the sliding-window FNR set). inference passes the input
# columns through, so the output carries phrog_db_category + pred_label that the
# central PHROG table groups by.
#
# Two passes per the repo's checkpoint rules (different models, by design):
#   SAE   : --run_sae  --model evo2_7b_262k     (zero-shot SAE feature 19746; NO classifier)
#   LP/NN : --run_nn --run_lp --model evo2_7b --model_dir <classifiers/2k>
#
# PREREQ for the LP/NN pass: evo2_classifiers/2k/ populated (from
# train_evo2_classifiers_delta.sbatch). The SAE pass needs no classifier and runs
# regardless — if the classifiers are missing, the SAE pass still completes and
# the LP/NN pass is skipped with a warning.
#
# Submit:  cd .../evo2val && sbatch Evo2_SAE_LAMBDA_assessment/scripts/lambda_replication/run_evo2_phrog_delta.sbatch
#          tail -f evo2_phrog_*.log
#
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=evo2_phrog
#SBATCH --output=evo2_phrog_%j.log
#SBATCH --error=evo2_phrog_%j.log

set -uo pipefail

# --- Delta EVO2 container setup (verbatim from rerun_evo2_3genomes_delta.sbatch) ---
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
A=$WD/Evo2_SAE_LAMBDA_assessment
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
E="--env PYTHONPATH=$LIB --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs --env TRITON_CACHE_DIR=$WD/triton_cache"

# --- inputs (PHROG is 2k ONLY) ---
W=2k
PHROG_CSV=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/fnr_test/${W}/phage_annotated_segments_${W}.csv
CLASS=$WD/evo2_classifiers
OUT=$WD/evo2_phrog_out
STEM=phage_annotated_segments_${W}

run() { apptainer exec --nv $B $E "$C" python "$A/src/batch_inference.py" "$@"; }

echo "[$(date)] EVO2 PHROG-category inference (2k only)"
[ -f "$PHROG_CSV" ] || { echo "ERROR: PHROG input not found: $PHROG_CSV"; exit 1; }

od="$OUT/$W"; mkdir -p "$od/_raw" "$od/evo2_lp" "$od/evo2_nn" "$od/evo2_sae"
ilist="$od/inputs.txt"; echo "$PHROG_CSV" > "$ilist"
echo "  input: $PHROG_CSV ($(($(wc -l < "$PHROG_CSV") - 1)) segments)"

# PASS 1 — SAE (evo2_7b_262k; zero-shot, no classifier needed)
echo "  [SAE] evo2_7b_262k ..."
run --input_list "$ilist" --output_dir "$od/_raw" --model evo2_7b_262k --run_sae \
    --feature_idx 19746 --sae_max_threshold 0.5 --sae_mean_threshold 0.1 --sae_fraction_threshold 0.3

# PASS 2 — LP + NN (evo2_7b + trained classifiers) — only if classifiers exist
if [ -f "$CLASS/$W/linear_probe.pkl" ] && [ -f "$CLASS/$W/three_layer_nn.pt" ]; then
    echo "  [LP/NN] evo2_7b + classifiers $CLASS/$W ..."
    run --input_list "$ilist" --output_dir "$od/_raw" --model evo2_7b --model_dir "$CLASS/$W" \
        --run_nn --run_lp
else
    echo "  WARNING: $CLASS/$W classifiers missing (linear_probe.pkl / three_layer_nn.pt)"
    echo "           SAE pass done; run train_evo2_classifiers_delta.sbatch first, then re-run for LP/NN."
fi

# Copy raw outputs to PHROG-canonical names (model-tag prefixed, matching the
# cross-model PHROG convention <TAG>_<input-stem>_predictions.csv). Uses cp (not
# mv) so the _raw originals stay as a safety net — adjust the targets if the EVO2
# PHROG harvest reader expects a different name/location.
cp -f "$od/_raw/${STEM}_sae_results.csv"             "$od/evo2_sae/EVO2_SAE_${STEM}_predictions.csv" 2>/dev/null || true
cp -f "$od/_raw/${STEM}_nn_predictions.csv"          "$od/evo2_nn/EVO2_NN_${STEM}_predictions.csv"   2>/dev/null || true
cp -f "$od/_raw/${STEM}_nn_predictions_metrics.json" "$od/evo2_nn/EVO2_NN_${STEM}_predictions_metrics.json" 2>/dev/null || true
cp -f "$od/_raw/${STEM}_lp_predictions.csv"          "$od/evo2_lp/EVO2_LP_${STEM}_predictions.csv"   2>/dev/null || true
cp -f "$od/_raw/${STEM}_lp_predictions_metrics.json" "$od/evo2_lp/EVO2_LP_${STEM}_predictions_metrics.json" 2>/dev/null || true

echo "[$(date)] DONE. Outputs under $od/{evo2_lp,evo2_nn,evo2_sae}/ (+ _raw/)."
echo "These feed the PHROG category-sensitivity table; fold into the staged EVO2/EVO2_SAE results at harvest."
