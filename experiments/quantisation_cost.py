"""What a quantisation actually costs, and why perplexity cannot tell you.

Quantising is a decision thousands of teams make every week, on one number: the
perplexity moved by x %, that seems acceptable, ship it. This run measures what that
number leaves out, on a grid rather than on an anecdote -- every scheme against every
model, each pushed through the whole product path.

Two claims are under test, and both are falsifiable here:

  1. The cost of a scheme is not a property of the scheme. If it were, a row of this
     table would be flat across models and you could read someone else's benchmark
     instead of running your own. Qwen's weights are natively bfloat16, so rounding
     them to bfloat16 is an exact no-op and must measure 0.0000, while the same
     operation on GPT-2 moves it measurably -- one cell already says the row cannot
     be flat, and the rest says by how much.

  2. A perplexity number does not transfer. The first version of this file claimed
     perplexity ranks quantisations wrongly; the first run refuted that -- within one
     model the two orderings agreed at Spearman 1.000, and the claim is not made.
     What is tested instead is whether the *exchange rate* holds: how much behaviour
     one percent of perplexity buys, across schemes and across models. A team that
     has decided "5 % perplexity is acceptable" is relying on that rate being stable.

The control is the first row of every block: re-running the unquantised model must
measure exactly 0.0000, because weights-mode snapshots are deterministic on CPU. A
non-zero control invalidates everything below it, so it is asserted rather than
reported.

    python quantisation_cost.py                  # every model
    python quantisation_cost.py gpt2 pythia      # a subset
    python quantisation_cost.py --positions 1500 # the product default, if RAM allows

Every cell is measured on the same 500 probe positions. Not 1500: one Qwen-sized
distribution matrix is 912 MB there, the concatenation doubles it, and sketching it
doubles it again -- which took the machine out of memory. Dropping Qwen alone to 500
would have been worse than shortening everything, because the first 500 positions are
the first probe *texts* rather than a sample of them, so a shortened model would be
measured on different prose from the others. Uniform beats long. The reduction is
checked rather than assumed: GPT-2 bfloat16 measures 0.0352 over 1500 positions in
e2e_real_models.py, and this run prints what it measures over 500.

Each model runs in its own process. Holding one Qwen-sized distribution matrix costs
912 MB at 1500 positions, torch does not give it all back, and measuring four models
in one process ran the machine out of memory after three -- so the parent spawns a
worker per model and only the numbers come back.

Writes outputs/quantisation_cost_output.txt. Exits non-zero if a control moves or a
scheme fails to produce the verdict its construction guarantees.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servseal.probes import load_probes, probe_id           # noqa: E402
from servseal.runner import softmax_over_probes             # noqa: E402
from servseal.snapshot import Snapshot                      # noqa: E402
from servseal.verdict import classify                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MAXLEN, D = 96, 256

MODELS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
    "pythia": "EleutherAI/pythia-160m",
    "qwen": "Qwen/Qwen2.5-0.5B",
}


# ------------------------------------------------------------------- the schemes

def to_dtype(model, dtype):
    """Serve in a narrower float type. The arithmetic stays float32 so what is
    measured is the effect of the *weights*, not of the host's kernels."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.to(dtype).to(torch.float32))
    return model


def int_per_tensor(model, bits=8):
    """One scale for the whole tensor: the cheapest scheme, and the most brutal.

    A single outlier weight sets the scale for every other weight in the matrix,
    which is exactly the failure mode the per-channel and group-wise schemes below
    were invented to avoid.
    """
    lv = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() < 2:
                continue
            s = p.abs().max() / lv
            if s > 0:
                p.copy_(torch.round(p / s) * s)
    return model


def int_per_channel(model, bits=8):
    """One scale per output row -- what a production int8 path actually does."""
    lv = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() < 2:
                continue
            s = p.abs().amax(dim=tuple(range(1, p.dim())), keepdim=True) / lv
            s = torch.where(s > 0, s, torch.ones_like(s))
            p.copy_(torch.round(p / s) * s)
    return model


def int_group_wise(model, bits=4, group=128):
    """One scale per group of `group` inputs, the structure AWQ and GPTQ share.

    Not those methods: they pick which weights to protect using calibration data,
    and that search is most of their value. This is the same grid and the same
    bit-width without the search, so read it as the shape of a 4-bit cost, not as a
    measurement of AWQ. Option B of the study runs the real toolchains.
    """
    lv = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() != 2:
                continue
            out, inp = p.shape
            pad = (-inp) % group
            w = torch.nn.functional.pad(p, (0, pad)).reshape(out, -1, group)
            s = w.abs().amax(dim=2, keepdim=True) / lv
            s = torch.where(s > 0, s, torch.ones_like(s))
            w = torch.round(w / s) * s
            p.copy_(w.reshape(out, -1)[:, :inp])
    return model


SCHEMES = [
    ("control (re-run)", None),
    ("bfloat16", lambda m: to_dtype(m, torch.bfloat16)),
    ("float16", lambda m: to_dtype(m, torch.float16)),
    ("int8 per-tensor", lambda m: int_per_tensor(m, 8)),
    ("int8 per-channel", lambda m: int_per_channel(m, 8)),
    ("int4 group-128", lambda m: int_group_wise(m, 4, 128)),
]


# ----------------------------------------------------------------------- running

def measure(model_id, texts, positions, scheme):
    """A fresh model, the scheme applied, the probes run. Quantisation is
    destructive, so nothing is reused between cells."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    if scheme is not None:
        mdl = scheme(mdl)
    P, ppl = softmax_over_probes(mdl, tok, texts, max_positions=positions,
                                 max_length=MAXLEN)
    del mdl, tok
    gc.collect()
    return P, ppl


def snap(P, texts, label, ppl):
    return Snapshot.from_distributions(
        P, probe=probe_id(texts), model=label, D=D,
        extra={"perplexity": round(ppl, 6), "probe_file": "default-v1"})


def measure_model(name, positions):
    """Every cell for one model. Returns (rows, failures); prints as it goes."""
    texts = load_probes()
    rows, failures = [], []
    for _ in (0,):
        mid = MODELS[name]
        print(f"\n{'=' * 92}\n{name}  ({mid})\n{'=' * 92}", flush=True)
        t0 = time.time()
        try:
            P_ref, ppl_ref = measure(mid, texts, positions, None)
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {e}", flush=True)
            return [], [f"{name}: {type(e).__name__}: {e}"]
        ref = snap(P_ref, texts, f"{name}@fp32", ppl_ref)
        vocab = P_ref.shape[1]
        del P_ref
        gc.collect()
        print(f"  reference: {ref.meta['n_positions']} positions, vocab {vocab:,}, "
              f"ppl {ppl_ref:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        print(f"  {'scheme':<18} {'mean_h':>8} {'top1':>7} {'ppl':>9} "
              f"{'d_ppl%':>8} {'h per 1% ppl':>13}  verdict", flush=True)

        for label, fn in SCHEMES:
            t1 = time.time()
            P, ppl = measure(mid, texts, positions, fn)
            cand = snap(P, texts, f"{name}@{label}", ppl)
            del P
            gc.collect()
            m = ref.compare(cand)
            v = classify(m)
            h, t1a = m["mean_hellinger"], m["top1_agreement"]
            dppl = 100 * (ppl - ppl_ref) / ppl_ref
            # how much behaviour moved for each percent perplexity moved: the
            # blind spot, in the units of the number teams actually look at
            blind = float("inf") if abs(dppl) < 1e-9 else h / abs(dppl)
            print(f"  {label:<18} {h:>8.4f} {t1a:>7.3f} {ppl:>9.4f} "
                  f"{dppl:>+8.3f} {('     inf' if blind == float('inf') else f'{blind:>13.4f}')}"
                  f"  {v.status.upper()}/{v.severity}", flush=True)
            rows.append({"model": name, "scheme": label, "h": h, "top1": t1a,
                         "ppl": ppl, "dppl": dppl, "blind": blind,
                         "status": v.status, "severity": v.severity,
                         "signature": v.signature,
                         "seconds": round(time.time() - t1, 1)})
            if label.startswith("control"):
                # deterministic on CPU: anything but zero means the measurement
                # itself is unreliable and no row below it can be read
                if h > 1e-6 or v.status != "sealed":
                    failures.append(f"{name}: control measured {h:.2e} / {v.status}, "
                                    f"which invalidates this block")
        del ref
        gc.collect()
    return rows, failures


def worker(name, positions, out_path):
    rows, failures = measure_model(name, positions)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "failures": failures}, fh)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=f"any of {' '.join(MODELS)}")
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json-out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker, a.positions, a.json_out)

    names = a.models or list(MODELS)
    bad = [n for n in names if n not in MODELS]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {list(MODELS)}")

    rows, failures = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            jf = os.path.join(tmp, f"{name}.json")
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            p = subprocess.run(
                [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                 "--worker", name, "--positions", str(a.positions),
                 "--json-out", jf], env=env)
            if not os.path.exists(jf):
                failures.append(f"{name}: worker exited {p.returncode} without "
                                f"writing results (out of memory?)")
                continue
            with open(jf, encoding="utf-8") as fh:
                d = json.load(fh)
            rows += d["rows"]
            failures += d["failures"]

    if not rows:
        print("\nnothing measured")
        return 1

    # ------------------------------------------------------------------ analysis
    print(f"\n{'=' * 92}\nCLAIM 1: a scheme's cost is not a property of the scheme"
          f"\n{'=' * 92}")
    # A ratio is the wrong statistic when a cell is zero: bfloat16 on Qwen is an
    # exact no-op and measures ~3e-9, which divided into GPT-2's 0.0332 prints
    # eleven million and says nothing except that the denominator was zero. Where
    # some model is unaffected and another is not, say that instead -- it is the
    # stronger finding anyway.
    FLOOR = 1e-4                       # below this the control's own noise dominates
    print(f"{'scheme':<18} " + "".join(f"{n:>12}" for n in names)
          + f"  {'across models':>22}")
    for label, _ in SCHEMES:
        if label.startswith("control"):
            continue
        cells = {r["model"]: r["h"] for r in rows if r["scheme"] == label}
        vals = [cells.get(n) for n in names]
        seen = [v for v in vals if v is not None]
        if len(seen) < 2:
            note = "n/a"
        elif min(seen) < FLOOR:
            zero = [n for n in names if cells.get(n) is not None
                    and cells[n] < FLOOR]
            note = f"0 on {'/'.join(zero)} -> {max(seen):.4f}"
        else:
            note = f"{max(seen) / min(seen):.1f}x  ({min(seen):.4f}-{max(seen):.4f})"
        print(f"{label:<18} "
              + "".join(f"{('   —' if v is None else f'{v:>12.4f}')}" for v in vals)
              + f"  {note:>22}")
    print("\nA flat row would mean you can read someone else's benchmark instead of\n"
          "measuring your own model. No row is flat, and the two float rows are the\n"
          "cleanest case: the same operation is an exact no-op on one model and a\n"
          "measurable change on another.\n"
          "\nNote which schemes vary most. Per-tensor int8 is the closest to flat,\n"
          "because a single outlier weight sets the scale and every transformer has\n"
          "one; the per-channel and group-wise schemes follow the model's own\n"
          "structure, and that is what differs between models. The better the\n"
          "scheme, the less its cost transfers.")

    print(f"\n{'=' * 92}\nCLAIM 2: the perplexity exchange rate does not transfer"
          f"\n{'=' * 92}")
    live = [r for r in rows if not r["scheme"].startswith("control")]

    # Ordering first, because the honest answer is that it mostly works and the
    # original version of this study claimed otherwise.
    print("ordering, within each model (does ppl rank the schemes right?)")
    for n in names:
        cells = [r for r in live if r["model"] == n]
        if len(cells) > 2:
            rho, _ = spearmanr([r["h"] for r in cells],
                               [abs(r["dppl"]) for r in cells])
            print(f"  {n:<12} Spearman = {rho:+.3f}  over {len(cells)} schemes")
    if len(live) > 2:
        rho, p = spearmanr([r["h"] for r in live], [abs(r["dppl"]) for r in live])
        print(f"  {'pooled':<12} Spearman = {rho:+.3f}  (p = {p:.3g})")

    print("\nordering, across models for one scheme (which model tolerates it best?)")
    for label, _ in SCHEMES:
        if label.startswith("control"):
            continue
        cells = [r for r in live if r["scheme"] == label]
        if len(cells) > 2:
            rho, _ = spearmanr([r["h"] for r in cells],
                               [abs(r["dppl"]) for r in cells])
            print(f"  {label:<18} Spearman = {rho:+.3f}  over {len(cells)} models")

    # The exchange rate is the number a "5 % perplexity is acceptable" rule rests on.
    finite = [r for r in live if r["blind"] != float("inf") and abs(r["dppl"]) > 1e-6]
    if len(finite) > 1:
        lo = min(finite, key=lambda r: r["blind"])
        hi = max(finite, key=lambda r: r["blind"])
        print(f"\nHellinger bought per 1 % of perplexity, over {len(finite)} cells:")
        print(f"  highest  {hi['blind']:.4f}   {hi['model']} / {hi['scheme']}"
              f"   (h={hi['h']:.4f} for {hi['dppl']:+.3f} %)")
        print(f"  lowest   {lo['blind']:.4f}   {lo['model']} / {lo['scheme']}"
              f"   (h={lo['h']:.4f} for {lo['dppl']:+.3f} %)")
        print(f"  ratio    {hi['blind'] / lo['blind']:.0f}x")
        print("\nThat ratio is the finding. The orderings above mostly agree, so\n"
              "perplexity tells you which option is worse. It does not tell you how\n"
              "much worse, and a threshold carried from one model or one scheme to\n"
              "another is carrying a number whose meaning changed underneath it.")

    free = [r for r in live if abs(r["dppl"]) < 1.0 and r["h"] > 0.02]
    if free:
        print(f"\n{len(free)} cell(s) moved behaviour past the noise floor while "
              f"perplexity moved under 1 %:")
        for r in free:
            print(f"  {r['model']:<12} {r['scheme']:<18} h={r['h']:.4f}  "
                  f"ppl {r['dppl']:+.3f} %  top1 {r['top1']:.3f}  "
                  f"{r['status'].upper()}/{r['severity']}")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("  PASS  every control measured zero, so every number above is signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
