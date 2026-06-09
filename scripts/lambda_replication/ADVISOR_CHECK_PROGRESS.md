# Checking the Evo2 LAMBDA inference run (for the advisor)

This note is for monitoring the Evo2 prophage inference that LeAnn left running on
the **CBB server** (`dirbiogpu11`). You do **not** need her tmux session — the
status is read straight off the filesystem and the running process. Everything
below is read-only; you cannot break the run by checking on it.

The run uses **one H200 GPU** with no scheduler, so the three window sizes
(**2k → 4k → 8k**) run **one at a time, in order**. Each window takes a while; the
whole thing is expected to run for many hours. That is normal — just check in
periodically.

---

## 0. One-time setup: log in and go to the project

```bash
ssh <your-user>@dirbiogpu11        # the CBB GPU server
cd /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment
```

If `cd` fails with "Permission denied", you need read access to LeAnn's directory —
ask CBB support (or LeAnn before she leaves) to grant it, e.g.
`setfacl -R -m u:<your-user>:rX` on the path. Nothing else here will work until
`cd` succeeds.

---

## There are two ways to check — use whichever you like

| | What it tells you | Command |
|---|---|---|
| **The manifest** (source of truth for *what's done*) | Which result files exist vs. are still missing, per window, plus a VERDICT. | `bash scripts/lambda_replication/check_evo2_status.sh` |
| **The log** (source of truth for *what it's doing right now*) | Live running commentary — which file it's on, and any error message if something breaks. | `tail -f inference_restart.log` |

Use the **manifest** to answer "is it done / what's left", and the **log** to answer
"is it actively working / did it crash". They complement each other.

> **Important:** pressing **`Ctrl-C` while watching the log with `tail -f` only stops
> the *tail*, NOT the inference.** It's safe. (To stop watching the log, that's exactly
> what you want.) The inference keeps running in its own tmux session regardless.

---

## 1. The main command: the manifest

```bash
bash scripts/lambda_replication/check_evo2_status.sh
```

This prints, in a few seconds:

- whether the inference **process is still running** (and which window it's on),
- a full **file-by-file manifest** — Stage 1 (the pretrained + random embedding
  results) and Stage 2 (lp / nn / sae predictions for every window), each marked
  `[ ok ]` or `[MISS]`,
- a **`N present, M missing`** tally, and
- a one-line **VERDICT** telling you what to do next.

While the run is still going, you'll see lots of `[MISS]` for windows it hasn't
reached yet (e.g. 4k and 8k while it's still on 2k) and for genome-wide — **that's
normal**, those files just don't exist yet. The VERDICT (not the MISS count) is what
tells you the overall state.

### What the output means

A healthy "still working" check looks like this (abridged — the real output lists
every file):

```
PROCESS: RUNNING  (pid 3895990, currently on 2k window)

############ STAGE 1 — embedding (pretrained + random) ############
  == 2k/embedding ==
      [ ok ] embedding_analysis_results.json
      [ ok ] linear_probe.pkl
      [ ok ] three_layer_nn.pt
      [ ok ] test_predictions_pretrained.csv
      [ ok ] test_predictions_random.csv
      ...

############ STAGE 2 — inference (lp / nn / sae) ############
  ======== window: 2k ========
    --- evo2_lp ---
      [ ok ] test_predictions.csv
      [MISS] fnr_predictions.csv            <- not produced yet; normal while running
      [MISS] genome_wide predictions = 0/12
    ...

MANIFEST: 31 present, 58 missing (required items only)
VERDICT: STILL RUNNING  -- let it continue; missing items are simply
         not produced yet. Re-check later.
```

- **`PROCESS: RUNNING`** → it's working. Do nothing; check again later.
- Each result file shows **`[ ok ]`** (exists) or **`[MISS]`** (not there yet).
  `[ -- ]` means an optional file (no action needed either way).
- While running, lots of `[MISS]` is **expected** — those files haven't been
  produced yet. Windows finish in order: **2k, then 4k, then 8k**, and within each,
  diagnostics first, then genome-wide. The **VERDICT line**, not the MISS count, is
  what tells you the overall state.

When everything is finished you'll see every line `[ ok ]`, zero missing, and:

```
MANIFEST: 89 present, 0 missing (required items only)
VERDICT: ALL COMPLETE  -- every required result is present and the
         process has stopped. Next: stage results + transfer.
```

### The three possible verdicts

| VERDICT | Meaning | What to do |
|---|---|---|
| **STILL RUNNING** | Working normally. | Nothing. Re-check in a few hours. |
| **ALL COMPLETE** | All required files present, process exited cleanly. | Go to **Step 3** (send results to LeAnn). |
| **STOPPED but INCOMPLETE** | The process died with required files still missing. | Email LeAnn the full output — it likely needs a restart (she has the steps). Don't restart it yourself unless she asks. |

That's the whole monitoring job: **run the command, read the VERDICT.** If you want to
see *what it's doing right now* between checks, `tail -f inference_restart.log`.

---

## 2. (Optional) See the actual metrics

Once windows are complete, you can see accuracy / MCC numbers per window and
variant:

```bash
bash scripts/lambda_replication/check_inference.sh
```

This lists every (window × variant) and reports `acc=.. mcc=..` for the linear
probe (`evo2_lp`) and neural net (`evo2_nn`). The SAE variant (`evo2_sae`)
intentionally shows "no metrics by design" — that's expected, not an error.

You don't need to interpret these; LeAnn does the analysis. This is just here if
you'd like a peek at the results.

---

## 3. When the VERDICT is ALL COMPLETE: send the results to LeAnn

The full output folder is large (it includes big activation files LeAnn doesn't
need by email). There's a packaging script that copies **only the small result
files** (and strips the bulky DNA-sequence column), leaving a small folder that's
easy to transfer.

```bash
# activate the project's python env (so pandas is available)
source /net/intdev/metagut/lindseylm/miniconda3/etc/profile.d/conda.sh
conda activate evo2-sae

# package the results into a small staging folder
python scripts/lambda_replication/stage_results_for_globus.py \
    /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/results \
    ~/evo2_lambda_results_to_send
```

It prints how many files it staged and the **total size in MB** (expect tens of
MB, not gigabytes). The staged folder is `~/evo2_lambda_results_to_send`.

Then get that folder to LeAnn by whatever is easiest:

```bash
# simplest: make one compressed file you can attach/transfer
tar -czf ~/evo2_lambda_results.tar.gz -C ~ evo2_lambda_results_to_send
```

Send her `~/evo2_lambda_results.tar.gz` (Globus, scp, or email if small enough).
That single file is everything she needs.

---

## Quick reference

```bash
cd /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment

# 1. is it done? full file manifest + VERDICT (run anytime)
bash scripts/lambda_replication/check_evo2_status.sh

# 1b. what is it doing right now? (Ctrl-C only stops the tail, not the run)
tail -f inference_restart.log

# 2. (optional) the actual numbers, once windows are complete
bash scripts/lambda_replication/check_inference.sh

# 3. when ALL COMPLETE: package + send
source /net/intdev/metagut/lindseylm/miniconda3/etc/profile.d/conda.sh
conda activate evo2-sae
python scripts/lambda_replication/stage_results_for_globus.py \
    /net/intdev/metagut/lindseylm/LAMBDA_GLMS/Evo2_SAE_LAMBDA_assessment/results \
    ~/evo2_lambda_results_to_send
tar -czf ~/evo2_lambda_results.tar.gz -C ~ evo2_lambda_results_to_send
```

**If anything looks wrong** (verdict says STOPPED but INCOMPLETE, or the status
script errors out), copy the full terminal output and email it to LeAnn.
