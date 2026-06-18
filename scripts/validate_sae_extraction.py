#!/usr/bin/env python3
"""
validate_sae_extraction.py — reproduce & validate the EVO2 SAE prophage extraction.

Context: a re-run of the SAE extraction produced near-zero firing on prophage
segments (fraction_firing ~0.07% vs the original ~36%), even though code, inputs,
machine, and conda env were reportedly unchanged. This script reproduces the
extraction on a fresh box and checks it against the ORIGINAL results (the oracle),
and — critically — runs BOTH activation-capture paths so we learn whether the
production `model.forward()` path is the problem vs `model.generate(cached)`:

  forward : ObservableEvo2.forward(toks, cache_activations_at=['blocks-26'])   (production)
  generate: model.generate([seq], n_tokens=1, cached_generation=True, ...)     (notebook "safe")

For each genome it compares, on PROPHAGE segments (label==1), the
normalization-independent `fraction_firing` (% of positions with feature>0) and
the per-segment firing rate, against the oracle CSV.

PASS criterion (per path): on prophage segments, mean fraction_firing within ~30%
of the oracle AND >=80% of prophage segments fire. That tells you the path
reproduces the original prophage signal.

Usage (on an x86 box with `evo2` installed, GPU):
  python scripts/validate_sae_extraction.py \
      --oracle_dir /path/to/per_segment_2k/EVO2_SAE \
      --input_csv  GCA_000012265.1_..._segments.csv [more CSVs ...] \
      --model evo2_7b --feature_idx 19746 \
      --methods forward generate --limit_segments 0

`--input_csv` files must have columns: segment_id, seq_id, start, end, label,
sequence (the same per-segment inputs used originally). The oracle dir holds the
original `*_sae_results.csv` for the same genomes (matched by assembly accession).
Use --limit_segments N for a fast smoke test (0 = all).
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import torch

# Reuse the EXACT production extraction infrastructure.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from sae_inference import ObservableEvo2, load_topk_sae, SAE_LAYER_NAME  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


def accession(name):
    m = re.search(r"(GC[AF]_\d+\.\d+|NC_\d+)", os.path.basename(name))
    return m.group(1) if m else None


def feats_forward(model, sae, seq):
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    _, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    return sae.encode(acts[SAE_LAYER_NAME][0]).cpu().detach().float().numpy()


def feats_generate(model, sae, seq):
    # The notebook's "won't crash" path: recurrent cached generation.
    _, acts = model.evo_model.generate(
        [seq], n_tokens=1, cached_generation=True, cache_activations_at=[SAE_LAYER_NAME]
    )
    return sae.encode(acts[SAE_LAYER_NAME][0]).cpu().detach().float().numpy()


CAPTURE = {"forward": feats_forward, "generate": feats_generate}


def seg_stats(feats, feature_idx):
    a = feats[:, feature_idx]
    return float(a.max()), float((a > 0).sum() / len(a)) if len(a) else 0.0


def load_oracle(oracle_dir, acc):
    for f in os.listdir(oracle_dir):
        if accession(f) == acc and f.endswith(".csv"):
            rows = list(csv.DictReader(open(os.path.join(oracle_dir, f))))
            return {(int(r["start"]), int(r["end"])): r for r in rows}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_dir", required=True)
    ap.add_argument("--input_csv", nargs="+", required=True)
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--feature_idx", type=int, default=19746)
    ap.add_argument("--methods", nargs="+", default=["forward", "generate"],
                    choices=["forward", "generate"])
    ap.add_argument("--limit_segments", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    print(f"Loading Evo2 ({args.model}) ...")
    model = ObservableEvo2(model_name=args.model)
    sae_path = hf_hub_download(repo_id="Goodfire/Evo-2-Layer-26-Mixed",
                              filename="sae-layer26-mixed-expansion_8-k_64.pt",
                              repo_type="model")
    sae = load_topk_sae(sae_path, d_hidden=model.d_hidden, device=model.device,
                        dtype=torch.bfloat16, expansion_factor=8)
    print(f"  device={model.device}  feature={args.feature_idx}  methods={args.methods}\n")

    print(f"{'genome':<18}{'path':<10}{'proph segs':>11}{'%segs fire':>12}{'mean frac_fire':>16}"
          f"{'  (oracle: %segs / mean_frac)'}")
    print("-" * 95)

    for csv_path in args.input_csv:
        acc = accession(csv_path)
        rows = list(csv.DictReader(open(csv_path)))
        oracle = load_oracle(args.oracle_dir, acc)
        if oracle is None:
            print(f"{acc:<18}  NO ORACLE in {args.oracle_dir} — skipping"); continue
        proph = [r for r in rows if str(r.get("label", "0")).strip() == "1"]
        if args.limit_segments:
            proph = proph[:args.limit_segments]
        if not proph:
            print(f"{acc:<18}  no prophage segments in input — skipping"); continue

        # oracle prophage firing (normalization-independent fraction_firing)
        o_frac, o_fire = [], 0
        for r in proph:
            k = (int(r["start"]), int(r["end"]))
            orow = oracle.get(k)
            if orow and "fraction_firing" in orow:
                ff = float(orow["fraction_firing"]); o_frac.append(ff); o_fire += (ff > 0)
        o_pct = o_fire / len(o_frac) if o_frac else float("nan")
        o_mean = float(np.mean(o_frac)) if o_frac else float("nan")

        for method in args.methods:
            cap = CAPTURE[method]
            fracs, fired = [], 0
            for r in proph:
                feats = cap(model, sae, r["sequence"])
                _, ff = seg_stats(feats, args.feature_idx)
                fracs.append(ff); fired += (ff > 0)
            pct = fired / len(fracs)
            mean = float(np.mean(fracs))
            ok = (pct >= 0.80) and (o_mean > 0) and (abs(mean - o_mean) / o_mean <= 0.30)
            tag = "PASS" if ok else "FAIL"
            print(f"{acc:<18}{method:<10}{len(proph):>11}{pct*100:>11.1f}%{mean:>16.4f}"
                  f"   (oracle: {o_pct*100:.0f}% / {o_mean:.4f})  [{tag}]")
    print("\nInterpretation:")
    print("  - A path that PASSES reproduces the original prophage firing -> use it for the re-run.")
    print("  - If 'forward' FAILS but 'generate' PASSES -> the production forward path is the bug;")
    print("    switch sae_inference.py / batch_inference.py to the generate(cached) path.")
    print("  - If BOTH fail vs the oracle -> the divergence is environmental (deps/driver), and")
    print("    this fresh box does not reproduce the original; pin package versions and retry.")


if __name__ == "__main__":
    main()
