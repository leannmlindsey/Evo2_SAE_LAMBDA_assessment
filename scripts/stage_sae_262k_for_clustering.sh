#!/bin/bash
# Stage the evo2_7b_262k SAE genome-wide results for the paper's clustering pipeline.
# Strips the bulky 'sequence' column and renames each per-genome file into the
# canonical layout the grid-search/cluster scripts expect:
#
#   EVO2_SAE/<w>/inference/EVO2_SAE/genome_wide_<stem>_predictions.csv
#
# then tars it into ONE small file to copy back to the laptop VIS repo.
#
# Run on Delta (login node is fine — CPU only):
#   cd /work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val/Evo2_SAE_LAMBDA_assessment
#   bash scripts/stage_sae_262k_for_clustering.sh
#
# Then copy  $WD/sae_262k_staged.tar.gz  to the Mac and untar inside
#   CLAUDE_LAMBDA_VISUALIZE_RESULTS/LAMBDA_v1_results_to_visualize/   (it creates EVO2_SAE/...)

set -eo pipefail
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
SRC=$WD/sae_262k_out
STAGE=$WD/sae_262k_stage
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif

rm -rf "$STAGE"
for w in 2k 4k 8k; do
  dst="$STAGE/EVO2_SAE/$w/inference/EVO2_SAE"
  mkdir -p "$dst"
  n=0
  for f in "$SRC/$w"/*_sae_results.csv; do
    stem=$(basename "$f" _sae_results.csv)
    out="$dst/genome_wide_${stem}_predictions.csv"
    # drop 'sequence' if present; keep everything else (start,end,label,max_activation,...)
    apptainer exec --cleanenv --bind /work/hdd/bfzj/llindsey1 "$C" python -c \
"import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); d.drop(columns=['sequence'],errors='ignore').to_csv(sys.argv[2],index=False)" \
      "$f" "$out"
    n=$((n+1))
  done
  echo "  $w: staged $n genomes -> $dst"
done

cd "$WD"
tar -czf sae_262k_staged.tar.gz -C "$STAGE" EVO2_SAE
echo ""
echo "DONE -> $WD/sae_262k_staged.tar.gz  ($(du -h sae_262k_staged.tar.gz | cut -f1))"
echo "Copy that file to the Mac and untar inside LAMBDA_v1_results_to_visualize/"
