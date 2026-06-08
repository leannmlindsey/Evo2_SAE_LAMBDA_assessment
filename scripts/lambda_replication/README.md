# Evo2 + SAE LAMBDA_v1 replication

Orchestration layer that reproduces the LAMBDA paper surfaces for **three Evo2
variants**, mirroring the ProkBERT / DNABERT-2 / NT-v2 / megaDNA
`lambda_replication/` pipelines. It only DRIVES the existing experiment code in
`src/*.py` — no model/experiment code is modified.

**This pipeline runs DIRECTLY on a single NCBI server** — there is NO SLURM
scheduler and NO environment modules. Each stage is a plain `bash` script you run
in the foreground; it activates the conda env (module-free) and calls the Python
entry point in-process. The server is **online**, so HuggingFace weights (Evo2 +
the SAE repo) download on first use — there is no prefetch / offline step.

**Evo2 is a frozen generative DNA language model and does NOT finetune.** The
three harvest-canonical variants are:

| variant    | method                                              | trained in Stage 1? |
|------------|-----------------------------------------------------|---------------------|
| `evo2_lp`  | frozen Evo2 embeddings -> **linear probe**          | yes (`linear_probe.pkl`) |
| `evo2_nn`  | frozen Evo2 embeddings -> **3-layer NN**            | yes (`three_layer_nn.pt`) |
| `evo2_sae` | **zero-shot SAE** feature f/19746 (threshold rules) | no — zero-shot |

## Two stages

**Stage 1 — train (lp + nn).** `run_lambda_training.sh` runs
`src/evo2_embedding_analysis.py` once per window, IN-PROCESS (one Evo2 load per
window, sequential). It extracts frozen Evo2 embeddings and fits BOTH the linear
probe and the 3-layer NN, saving `linear_probe.pkl` + `three_layer_nn.pt`
(+ scalers) and `embedding_analysis_results.json` (pretrained-vs-random metrics =
Surface D). `--include_random_baseline` is ALWAYS on. No seed sweep, no
`select_best_model`, no winners. The SAE variant needs no Stage 1.

**Stage 2 — inference (all 3 variants).** `run_lambda_inference.sh` runs
`src/batch_inference.py` once per window. `batch_inference` loads Evo2 ONCE and
runs `--run_sae --run_nn --run_lp` over a single input list (diagnostics +
genome-wide), then the driver RENAMES the outputs into the harvest-canonical
`inference/<variant>/` layout.

## Layout

```
scripts/lambda_replication/            # <-- this directory
  lambda_replication.conf              # all paths + hyperparameters (edit this)
  run_lambda_training.sh               # STAGE 1: evo2_embedding_analysis.py per window (lp + nn)
  run_lambda_inference.sh              # STAGE 2: batch_inference.py + rename into canonical layout
  check_training.sh                    # Stage 1 completeness + pretrained-vs-random MCC
  check_inference.sh                   # Stage 2 completeness + acc/mcc per (window x variant)
  README.md                            # this file
```

There is intentionally **no** finetune stage, **no** `select_best_model`, and
**no** local genome-wide scanner / aggregation. The native 50 kb scanner
(`nucleotide_evaluation.py`) and `run_lambda_batch.py` are NOT used —
genome-level aggregation is done CENTRALLY by the harvest pipeline.

## Environment

Both drivers activate the conda env the module-free way (no environment modules):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evo2-sae
```

The env (`evo2-sae`, set via `CONDA_ENV` in the conf) must have torch + the `evo2`
package + `huggingface_hub`. The drivers also set `PYTHONNOUSERSITE=1`, derive
`CUDA_HOME` from `nvcc` if unset, `cd` to the repo root, and put it on
`PYTHONPATH`.

## Outputs

```
$OUTPUT_DIR/
  <W>/                                 # 2k, 4k, 8k
    embedding/                         # linear_probe.pkl, linear_probe_scaler.pkl,
                                       #   three_layer_nn.pt, three_layer_nn_scaler.pkl,
                                       #   embedding_analysis_results.json
    inference/
      evo2_lp/                         # test_predictions.csv (+ _predictions_metrics.json),
                                       #   fpr_/gc_control_/fnr_predictions.csv,
                                       #   genome_wide_<stem>_predictions.csv (+ _metrics.json)
      evo2_nn/                         # same canonical names
      evo2_sae/                        # same canonical CSV names, but NO _metrics.json (see note)
  _inference_lists/                    # per-window input lists + name maps
```

### Canonical names + the rename map

`batch_inference.py` names its outputs `<basename>_<method>` in a tmp dir, where
`<basename>` is the input CSV stem. The inference driver then moves each one into
the canonical `inference/<variant>/<canon>_predictions.csv`:

| batch_inference output (`<basename>_...`)   | canonical destination                                  |
|---------------------------------------------|--------------------------------------------------------|
| `<basename>_sae_results.csv`                | `inference/evo2_sae/<canon>_predictions.csv`           |
| `<basename>_nn_predictions.csv`             | `inference/evo2_nn/<canon>_predictions.csv`            |
| `<basename>_nn_predictions_metrics.json`    | `inference/evo2_nn/<canon>_predictions_metrics.json`   |
| `<basename>_lp_predictions.csv`             | `inference/evo2_lp/<canon>_predictions.csv`            |
| `<basename>_lp_predictions_metrics.json`    | `inference/evo2_lp/<canon>_predictions_metrics.json`   |

`<canon>` is mapped from the input file (NAME_MAP, built by the driver):

| `<canon>`            | input file                                                |
|----------------------|-----------------------------------------------------------|
| `test`               | `train_val_test/<W>/test.csv`                             |
| `fpr`                | `fpr_test/<W>/bacteria_segments_<W>.csv`  (auto-derived)  |
| `gc_control`         | `shuffled_controls/<W>/test_shuffled.csv` (auto-derived)  |
| `fnr`                | `FNR_<W>` (if set + exists)                               |
| `genome_wide_<stem>` | each `GENOME_WIDE_<W>/*.csv` stem (the harvest glob key)  |

Example: `test.csv` -> `test_lp_predictions.csv` -> `inference/evo2_lp/test_predictions.csv`;
its SAE counterpart `test_sae_results.csv` -> `inference/evo2_sae/test_predictions.csv`.

### SAE schema note

The SAE CSV (`evo2_sae/*_predictions.csv`) does **NOT** have `prob_0`/`prob_1`
columns like the lp/nn CSVs. By design it carries `segment_id`, passthrough
columns, `sequence`, then `max_activation`, `mean_activation`, `fraction_firing`,
and `pred_label` (the OR of the three SAE thresholds). It is left as-is — the
schema is intentional and `evo2_sae` writes **no** `_metrics.json`. The harvest
aggregator handles the SAE schema separately.

### FPR / FNR are single-class

FPR is bacteria-only and FNR is phage-only (single-class diagnostics) — same as
the other LAMBDA repos. Inference still runs uniformly over them.

### Genome-wide

Genome-wide runs **all three variants** (`--run_sae --run_nn --run_lp`) over every
`GENOME_WIDE_<W>/*.csv` segment file, renamed to
`inference/<variant>/genome_wide_<stem>_predictions.csv` (+ `_metrics.json` for
lp/nn). The `genome_wide_` prefix is the harvest glob key.

## Full sweep

```bash
cd /path/to/Evo2_SAE_LAMBDA_assessment

# 1. edit scripts/lambda_replication/lambda_replication.conf
#    set LAMBDA_BASE + OUTPUT_DIR (and confirm MODEL, LAYER).
# 2. STAGE 1: train lp + nn (one in-process run per window)
bash scripts/lambda_replication/run_lambda_training.sh

# 3. verify Stage 1
bash scripts/lambda_replication/check_training.sh

# 4. STAGE 2: batch inference for all 3 variants
bash scripts/lambda_replication/run_lambda_inference.sh

# 5. verify Stage 2
bash scripts/lambda_replication/check_inference.sh
```

HuggingFace weights download automatically on the first Evo2/SAE load (the server
is online). The first window will be slower while the cache warms.

## evo2_40b follow-on

The default is `evo2_7b`. To run the larger 40B model, in
`lambda_replication.conf`:

1. `MODEL="evo2_40b"`,
2. set a **40B-appropriate `LAYER`** (the 7B value `blocks.28.mlp.l3` is NOT valid
   for the 40B block count — pick the corresponding 40B layer).

The 40B weights download on first use, same as 7B.

## Notes / assumptions

- **No `src/` code is modified.** The drivers call `src/evo2_embedding_analysis.py`
  and `src/batch_inference.py` with explicit paths; the canonical rename is pure
  bash.
- `src/evo2_embedding_analysis.py` reads `train.csv` + `test.csv` + `dev.csv` OR
  `val.csv` directly from `--csv_dir` (it falls back to `val.csv`), so no data
  staging is needed.
- **Environment:** module-free conda — `source "$(conda info --base)/etc/profile.d/conda.sh"`
  then `conda activate evo2-sae`. Online HF (no prefetch / offline flags).
- Set `LAMBDA_BASE` and `OUTPUT_DIR` in the conf before running; the drivers error
  out if they are still the `/path/to/...` placeholders.
