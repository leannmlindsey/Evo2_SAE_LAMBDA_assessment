#!/usr/bin/env python
"""Stage EVO2 LAMBDA inference results for a Globus transfer.

Copies ONLY the small result files into STAGE, preserving the
<W>/inference/<variant>/ tree, and DROPS the 'sequence' (DNA) column from every
*_predictions.csv so the genome-wide outputs stay small. Large intermediates
(embeddings_*.npz, *.pkl, *.pt, *.npy) are never touched.

Staged:
  *_predictions.csv          -> copied WITHOUT the 'sequence' column (LP / NN, and
                                renamed SAE under inference/evo2_sae/)
  *_sae_results.csv          -> SAE results (un-renamed, in _raw/) copied as-is
  *_metrics.json             -> copied as-is
  embedding_analysis_results.json -> copied as-is
(also picks up partial/un-renamed predictions under inference/_raw/.)
The large *_activations/ .npy dumps are NEVER staged.

Usage (run on CBB with the evo2 conda env active, so pandas is available):
  python scripts/lambda_replication/stage_results_for_globus.py SRC STAGE

  SRC   = .../Evo2_SAE_LAMBDA_assessment/results
  STAGE = a path under your CBB Globus endpoint
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

# Files we rewrite (drop the DNA 'sequence' column if present). SAE result CSVs
# have no 'sequence' column, so the drop is a harmless no-op for them.
PRED_SUFFIXES = ("_predictions.csv", "_sae_results.csv")
COPY_NAMES = {"embedding_analysis_results.json"}
COPY_SUFFIX = "_metrics.json"
DROP_COLS = ("sequence",)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python stage_results_for_globus.py SRC STAGE")
    src = Path(sys.argv[1]).resolve()
    stage = Path(sys.argv[2]).resolve()
    if not src.is_dir():
        sys.exit(f"SRC not a directory: {src}")
    if stage == src or src in stage.parents:
        sys.exit("STAGE must be outside SRC")

    n_pred = n_copy = n_err = 0
    total = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        out = stage / p.relative_to(src)
        try:
            if name.endswith(PRED_SUFFIXES):
                out.parent.mkdir(parents=True, exist_ok=True)
                df = pd.read_csv(p)
                df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
                df.to_csv(out, index=False)
                n_pred += 1
            elif name in COPY_NAMES or name.endswith(COPY_SUFFIX):
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
                n_copy += 1
            else:
                continue
            total += out.stat().st_size
        except Exception as e:  # don't let one half-written file abort the run
            n_err += 1
            print(f"  WARNING: skipped {p} ({e})")

    print(f"\nstaged {n_pred} prediction/SAE-result CSVs (sequence column dropped "
          f"where present) + {n_copy} json/metrics files")
    if n_err:
        print(f"  ({n_err} file(s) skipped — see warnings above)")
    print(f"total staged size: {total / 1048576:.1f} MB")
    print(f"destination: {stage}")


if __name__ == "__main__":
    main()
