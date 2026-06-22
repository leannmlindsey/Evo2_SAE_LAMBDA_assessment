# Running Evo2 on NCSA Delta-AI (GH200 / ARM)

How Evo2-7B was made to run on **Delta-AI** (NCSA), and how to run it again. Delta-AI
is **ARM (aarch64) Grace-Hopper GH200** nodes — a plain `pip install evo2` fails to
build flash-attn/Transformer-Engine on ARM, so we run **inside NVIDIA's NGC PyTorch
Apptainer container** (which already ships ARM builds of torch + flash-attn + TE) and
layer Evo2 on top.

Account/allocation used: **`bfzj-dtai-gh`**, partition **`ghx4`** (GH200).

---

## Where everything lives

| What | Path |
|------|------|
| Work dir | `/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val` (`$WD`) |
| Evo2/vortex install (pip `--target`) | `$WD/pylib` (`$LIB`) |
| HF model cache (evo2_7b ~14 GB) | `/work/hdd/bfzj/llindsey1/hf_cache` (`$HF`) |
| Triton JIT cache | `$WD/triton_cache` |
| pip cache | `$WD/pipcache` |
| NGC container (the one that works) | `/sw/user/NGC_containers/pytorch_26.01-py3.sif` |

**Use the newest PyTorch container (`26.01`).** The host driver is **590.48.01 /
CUDA 13.1**; the old `pytorch_24.09` container (CUDA 12.6, bundled cuda-compat) fails
with `error 803` (driver/compat mismatch). Newest container = matches the driver.

---

## Quick start (every future session)

```bash
# 1. Get an interactive GH200 node (bump --mem; model load is RAM-hungry)
srun --account=bfzj-dtai-gh --partition=ghx4 --nodes=1 --ntasks=1 \
     --cpus-per-task=8 --mem=64g --gpus-per-node=1 --time=4:00:00 --pty bash

# 2. Set the standard variables
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
LIB=$WD/pylib
HF=/work/hdd/bfzj/llindsey1/hf_cache
# the bind + env-isolation flags (reused on EVERY apptainer call):
B="--cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1"
# the runtime env Evo2 needs (HF download, no-xet, Triton libcuda, caches):
E="--env PYTHONPATH=$LIB --env HF_HOME=$HF --env HF_HUB_DISABLE_XET=1 \
   --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
   --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
   --env TRITON_CACHE_DIR=$WD/triton_cache"

# 3. Run anything inside the container, e.g. a smoke test:
apptainer exec --nv $B $E $C python -c "import torch; from evo2 import Evo2; m=Evo2('evo2_7b'); ids=torch.tensor(m.tokenizer.tokenize('ACGT'*32),dtype=torch.int).unsqueeze(0).to('cuda:0'); out,_=m(ids); print('FORWARD OK', out[0].shape)"

# Run a script:
apptainer exec --nv $B $E $C python /path/to/script.py --args ...
```

**Carry all of `$B` and `$E` on every `apptainer exec` call** — each flag fixes a
specific problem (see "Why each flag" below).

---

## One-time install (already done; redo only on a fresh `$LIB`)

Evo2 + its deps are installed into `$LIB` via `pip --target`, **`--no-deps`** so pip
reuses the container's torch/flash-attn/TE instead of rebuilding them on ARM:

```bash
PC=$WD/pipcache
apptainer exec --nv $B --env PIP_CACHE_DIR=$PC $C \
  pip install --no-deps --target $LIB evo2 vtx biopython huggingface_hub
```
Notes:
- `vtx` **is** vortex (StripedHyena). PyPI name `vtx`, **import name `vortex`**.
- `evo2/evo2/` is a thin wrapper (`models.py`, `scoring.py`) over `vortex`.
- We did NOT need to clone the Evo2 GitHub repo — `vtx` covers vortex.

---

## Why each flag (the problems we hit, in order)

1. **`error 803` (CUDA failed to initialize)** → the 2024 container's cuda-compat
   conflicts with the new 590 driver. **Fix: use `pytorch_26.01` (newest).**
2. **`ModuleNotFoundError: torch` / wrong torch from `~/.local`** → an old user-site
   torch (leftover from a prior pip-install attempt) shadows the container's, and an
   interactive `apptainer ... bash` sources `~/.bashrc` → re-activates host conda.
   **Fix: `--cleanenv --env PYTHONNOUSERSITE=1`, and run non-interactively
   (`apptainer exec ... python`), not via an interactive shell.**
3. **venv can't see container torch** → NGC installs into Debian `dist-packages`,
   which a venv's `--system-site-packages` does NOT inherit. **Fix: skip the venv;
   use `pip install --target $LIB` + `--env PYTHONPATH=$LIB`.**
4. **`hf_xet` / `XetFileInfo` ImportError on model download** → our newer
   `huggingface_hub` wants a newer `hf_xet` than the container has. **Fix:
   `--env HF_HUB_DISABLE_XET=1`** (falls back to plain HTTPS download).
5. **Triton: `libcuda.so cannot found`** → Triton (used by vortex's rotary kernel)
   locates `libcuda` via `ldconfig`, and the `--nv`-injected driver dir
   (`/.singularity.d/libs`) isn't in the container's ldconfig cache. `LD_LIBRARY_PATH`
   alone does NOT fix it. **Fix: `--env TRITON_LIBCUDA_PATH=/.singularity.d/libs`**
   (explicit override; that dir holds the host 590 driver's `libcuda.so.1`). We set
   `LD_LIBRARY_PATH` too for other CUDA libs, but `TRITON_LIBCUDA_PATH` is the key one.
6. **home quota** → model weights + Triton cache must go on `/work`, not `$HOME`.
   **Fix: `HF_HOME`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR` all under `$WD`/`/work`.**

---

## Versions that work (captured for reproducibility)

- Container: `pytorch_26.01-py3.sif` — torch `2.10.0a0…nv26.01`
- flash-attn `2.7.4.post1`, transformer-engine `2.11.0` (from the container)
- evo2 `0.5.5`, vtx (vortex) `1.1.0`, huggingface_hub `1.20.0`, biopython `1.87`
- Host driver `590.48.01`, CUDA 13.1; GPU NVIDIA GH200 120GB
- evo2_7b weights snapshot: `bda0089f92582d5baabf0f22d9fc85f3588f6b58`

> Note: this container's TE is **2.11** (Evo2 nominally expects ~2.3) and flash-attn
> **2.7.4** (vs 2.8.0.post2) — they load and run the 7B fine here. If a future
> container breaks the model build, that version skew is the first thing to check.

---

## Optional: a wrapper to avoid retyping

Save as `$WD/evo2run` and `chmod +x` it:
```bash
#!/bin/bash
C=/sw/user/NGC_containers/pytorch_26.01-py3.sif
WD=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/evo2val
exec apptainer exec --nv \
  --cleanenv --env PYTHONNOUSERSITE=1 --bind /work/hdd/bfzj/llindsey1 \
  --env PYTHONPATH=$WD/pylib --env HF_HOME=/work/hdd/bfzj/llindsey1/hf_cache \
  --env HF_HUB_DISABLE_XET=1 \
  --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64 \
  --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
  --env TRITON_CACHE_DIR=$WD/triton_cache \
  "$C" "$@"
```
Then: `./evo2run python my_script.py …`

---

## ⚠️ CRITICAL: SAE extraction must use `evo2_7b_262k`, not `evo2_7b`

The Goodfire SAE (`Evo-2-Layer-26-Mixed`, feature f/19746) was trained on the
**`evo2_7b_262k`** checkpoint and only fires correctly on it. They are *different*
weight files (`evo2_7b_262k.pt` ≠ `evo2_7b.pt`). Verified on Delta GH200:

| `--model` | prophage firing (GCA_000012265.1, blocks-26 forward) |
|-----------|------------------------------------------------------|
| `evo2_7b` (wrong) | **6% / mean 0.0001** — feature silent |
| **`evo2_7b_262k`** (correct) | **100% / mean ~0.31** — matches original oracle (100% / 0.34) |

`sae_inference.py` / `batch_inference.py` default to `--model evo2_7b`, which silently
produces a near-dead SAE signal. **Always pass `--model evo2_7b_262k` for SAE
extraction** (or change the default). This was the root cause of the re-run's
prophage under-firing — not vortex version, not ARM, not the environment.

## What this is for

Re-extracting the **EVO2 SAE prophage feature (f/19746)** to debug why a re-run
under-fired on prophage (see the SAE bug investigation). Next step after the smoke
test passes: run `validate_sae_extraction.py` against the original oracle on the
3 known-prophage genomes (forward vs generate), then re-extract 2k/4k/8k if it
reproduces the original.
