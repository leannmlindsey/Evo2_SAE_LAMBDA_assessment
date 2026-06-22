# Evo2 LAMBDA inference — handoff / status note

**Context:** Evo2 prophage inference is running on the **CBB server** (`dirbiogpu11`),
one **H200 GPU**, no scheduler. It produces three result "variants" — `evo2_lp`
(linear probe), `evo2_nn` (3-layer NN), and `evo2_sae` (zero-shot SAE feature). It
runs the three window sizes **2k → 4k → 8k**, one at a time. **This is a multi-day
run** — it will not finish in a single day. That is expected.

Project dir on the server:
```
/net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment
```

---

## 1. How to check progress

```bash
cd /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment
bash scripts/lambda_replication/check_evo2_status.sh
```

The output has three parts. **Read them differently** — this is the part that
caused confusion, so it matters:

### a) LIVE PROGRESS — this is the real-time progress meter
```
############ LIVE PROGRESS (_raw, current window) ############
  2k   ~4 / 86 inputs processed    [sae:4  nn:3  lp:3]
  4k   not started (no _raw)  [expected inputs: ?]
  8k   not started (no _raw)  [expected inputs: ?]
```
**`~4 / 86 inputs processed` is the number to watch.** It climbs as the run works
through the 86 input files of the current window. The `[sae/nn/lp]` counts being
uneven just means a file is mid-processing (normal).

### b) STAGE 1 — the trained models (already done)
All lines should be `[ ok ]` for 2k/4k/8k. This is finished.

### c) STAGE 2 — the FINAL transferable results
```
    --- evo2_lp ---
      [MISS] test_predictions.csv
      ...
```
**`[MISS]` here is NORMAL while the run is going.** Stage 2 checks the *final*
folders, which only fill up **after a whole window finishes** (all 86 files), when
the results get renamed into place. So expect `[MISS]` for a long time even though
work is happening — judge progress by **LIVE PROGRESS (a)**, not by Stage 2.

### The VERDICT line at the bottom
- **STILL RUNNING** → healthy, let it continue. (This is the normal state for days.)
- **ALL COMPLETE** → every result is in; go to section 3 (package + transfer).
- **STOPPED but INCOMPLETE** → it died early; see section 4.

---

## 2. Is it actually alive (vs. stuck)?

A single big input file can run for **~2 hours producing no new files** (it writes
only when each step finishes). So "no new files" does NOT mean it's stuck. Confirm
it's alive with:

```bash
nvidia-smi          # GPU should be busy (a python process, high utilization)
tail -n 5 inference_restart.log   # the progress bar should advance over ~30s
```
- GPU busy + bar moving → fine, just slow.
- GPU idle (0%) or bar frozen across two checks → genuinely stuck (see section 4).

> Watching the log with `tail -f` is safe — pressing **Ctrl-C only stops the
> `tail`, NOT the inference.** Never run `Ctrl-C` in the window where the job itself
> is running.

---

## 3. When VERDICT says ALL COMPLETE — package + transfer

One command gathers the small result files (drops the bulky DNA sequence column,
skips the huge `.npy`/`.npz`) and makes one tarball:

```bash
cd /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment
bash scripts/lambda_replication/prep_results_for_transfer.sh
```

It prints the staged folder and a tarball path, e.g.:
```
/net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/evo2_lambda_results_to_send.tar.gz
```
Then move that tarball into the Globus endpoint and transfer it to LeAnn.

---

## 4. If it dies before finishing (STOPPED but INCOMPLETE)

- **Nothing on disk is lost.** Finished files stay in `results/<W>/inference/_raw/`,
  and the packaging script in section 3 also collects from `_raw/` — so partial
  results are still transferable.
- A restart re-runs the current window from the start (there's no mid-window
  resume). Restart steps (run inside `tmux`, on GPU 7):
  ```bash
  source /net/intdev/metagut/lindseylm/miniconda3/etc/profile.d/conda.sh
  conda activate evo2-sae
  export HF_HOME=/net/intdev/metagut/lindseylm/.cache
  cd /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment
  tmux new -s evo2
  PHASE=all CUDA_VISIBLE_DEVICES=7 bash scripts/lambda_replication/run_lambda_inference.sh 2>&1 | tee inference_restart.log
  ```
  Then detach with **`Ctrl-b` then `d`** (do NOT press `Ctrl-C`).
- To confirm an account *can* restart it (env, weights, data, GPU all reachable),
  run **as that account**:
  ```bash
  bash scripts/lambda_replication/check_restart_ready.sh
  ```

---

## One-line summary for the advisor

> Run `check_evo2_status.sh`. Watch the **`~N / 86 inputs processed`** line — that's
> progress. The **`[MISS]`** lines under STAGE 2 are normal until a whole window
> finishes. When the VERDICT says **ALL COMPLETE**, run
> `prep_results_for_transfer.sh` and Globus the tarball to LeAnn. It's a multi-day
> run; "no new files for ~2 hours" is normal — check `nvidia-smi` to confirm it's
> alive.
