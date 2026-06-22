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
# Single container launch: loop over all windows/genomes inside one python process
# (avoids hundreds of apptainer starts -- friendlier on a shared login node).
apptainer exec --cleanenv --bind /work/hdd/bfzj/llindsey1 "$C" python - "$SRC" "$STAGE" <<'PY'
import sys, os, glob
import pandas as pd
src, stage = sys.argv[1], sys.argv[2]
for w in ("2k", "4k", "8k"):
    dst = os.path.join(stage, "EVO2_SAE", w, "inference", "EVO2_SAE")
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, w, "*_sae_results.csv")))
    for f in files:
        stem = os.path.basename(f)[:-len("_sae_results.csv")]
        out = os.path.join(dst, f"genome_wide_{stem}_predictions.csv")
        d = pd.read_csv(f)
        d.drop(columns=["sequence"], errors="ignore").to_csv(out, index=False)
    print(f"  {w}: staged {len(files)} genomes -> {dst}")
PY

cd "$WD"
tar -czf sae_262k_staged.tar.gz -C "$STAGE" EVO2_SAE
echo ""
echo "DONE -> $WD/sae_262k_staged.tar.gz  ($(du -h sae_262k_staged.tar.gz | cut -f1))"
echo "Copy that file to the Mac and untar inside LAMBDA_v1_results_to_visualize/"
