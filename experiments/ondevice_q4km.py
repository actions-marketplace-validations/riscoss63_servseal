"""The quant that actually ships on-device, and the number a failure rate compares to.

A reader of the quantisation write-up described the deployment this study had not
looked at: on device you are stuck at a 0.6B at Q4_K_M with nothing bigger to fall
back on, perplexity looks fine, and the shift wrecks the sampler -- wrong-branch tool
calls and drifting JSON, while recall@8 stays clean. Their conclusion was that rank-1
is the tell, and they offered to compare a Hellinger number against the wrong-branch
rate they observe in production.

That offer is worth more than another row, because it is the one thing this tool has
never had: an outside failure rate to correlate against.

So this measures exactly what ships -- `bartowski/Qwen_Qwen3-0.6B-GGUF`, the file
NobodyWho's own examples load -- rather than a quantisation made here. Downloading the
shipped artifact removes the question of whether my conversion matches theirs.

What to correlate against
-------------------------
Mean Hellinger is the wrong number to hand them. It is an average over the whole
distribution including the tail, and a wrong-branch tool call is a *head* event: the
model picked a different top token at a decision point. The comparable quantity is

    1 - top1_agreement  =  the fraction of probe positions where the argmax flipped

which is a rate, like theirs. Both are reported, and so is the per-position spread,
because a flip rate concentrated in a few positions means something different from
the same rate spread evenly.

    python ondevice_q4km.py                    # ~5 min, CPU only, no GPU needed
    python ondevice_q4km.py --positions 1500   # tighter, needs ~5 GB free
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

# transformers reads `gguf.__version__` to decide whether it can open a GGUF, and the
# gguf package does not define one -- so it records "N/A" and then crashes parsing it
# as a version. The metadata is there; only the attribute is missing. Set before
# transformers is imported, because that check runs once at import and is cached.
try:
    import importlib.metadata as _md
    import gguf as _gguf
    if not hasattr(_gguf, "__version__"):
        _gguf.__version__ = _md.version("gguf")
except Exception as _e:
    print(f"gguf not usable ({type(_e).__name__}); the GGUF row will fail", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servseal.probes import load_probes, probe_id           # noqa: E402
from servseal.runner import softmax_over_probes             # noqa: E402
from servseal.snapshot import Snapshot                      # noqa: E402
from servseal.verdict import classify                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL = "Qwen/Qwen3-0.6B"
GGUF_REPO = "bartowski/Qwen_Qwen3-0.6B-GGUF"
GGUF_FILE = "Qwen_Qwen3-0.6B-Q4_K_M.gguf"
MAXLEN, D = 96, 256


def measure(model, tok, texts, positions):
    model = model.to(torch.float32).to("cpu").eval()
    return softmax_over_probes(model, tok, texts, max_positions=positions,
                               max_length=MAXLEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--repo", default=GGUF_REPO)
    ap.add_argument("--file", default=GGUF_FILE)
    a = ap.parse_args()

    texts = load_probes()
    ph = probe_id(texts)
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"reference: {MODEL} float32 on CPU, {a.positions} positions", flush=True)
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    P, ppl_ref = measure(m, tok, texts, a.positions)
    ref = Snapshot.from_distributions(
        P, probe=ph, model=f"{MODEL}@fp32", D=D,
        extra={"perplexity": round(ppl_ref, 6), "probe_file": "default-v1"})
    vocab = P.shape[1]
    del m, P
    gc.collect()
    print(f"  vocab {vocab:,}  ppl {ppl_ref:.4f}  [{time.time() - t0:.0f}s]",
          flush=True)

    # the control first: re-running an unchanged model must measure exactly zero, or
    # nothing below it can be read as signal
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    P, ppl2 = measure(m, tok, texts, a.positions)
    ctl = ref.compare(Snapshot.from_distributions(
        P, probe=ph, model="control", D=D,
        extra={"perplexity": round(ppl2, 6), "probe_file": "default-v1"}))
    del m, P
    gc.collect()
    print(f"  control (re-run, nothing changed): h={ctl['mean_hellinger']:.6f}  "
          f"top1={ctl['top1_agreement']:.4f}", flush=True)
    assert ctl["mean_hellinger"] < 1e-6, "the control moved; the measurement is unsound"

    from huggingface_hub import hf_hub_download
    print(f"\ncandidate: {a.repo}/{a.file}", flush=True)
    t0 = time.time()
    path = hf_hub_download(repo_id=a.repo, filename=a.file)
    print(f"  {os.path.getsize(path) / 2**20:.0f} MB  [{time.time() - t0:.0f}s]",
          flush=True)
    q = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(path), gguf_file=os.path.basename(path),
        dtype=torch.float32)
    P, ppl = measure(q, tok, texts, a.positions)
    cand = Snapshot.from_distributions(
        P, probe=ph, model=f"{a.file}", D=D,
        extra={"perplexity": round(ppl, 6), "probe_file": "default-v1"})
    del q, P
    gc.collect()

    mm = ref.compare(cand)
    v = classify(mm)
    flip = 1 - mm["top1_agreement"]
    dppl = 100 * (ppl - ppl_ref) / ppl_ref

    print(f"\n{'=' * 78}\n{MODEL} fp32  ->  {a.file}\n{'=' * 78}")
    print(f"  mean Hellinger       {mm['mean_hellinger']:.4f}")
    print(f"  top-1 agreement      {mm['top1_agreement']:.4f}")
    print(f"  TOP-1 FLIP RATE      {flip:.4f}   <- the rate to compare a "
          f"wrong-branch rate against")
    print(f"  perplexity           {ppl_ref:.4f} -> {ppl:.4f}  ({dppl:+.2f} %)")
    print(f"  positions moved      {mm['positions_moved']} / {mm['n_positions']} "
          f"above the noise floor {mm['noise_floor']:.4f}")
    print(f"  verdict              {v.status.upper()}/{v.severity} -- {v.signature}")

    # a flip rate concentrated in a few positions is a different failure mode from
    # the same rate spread evenly, and the histogram is what distinguishes them
    hist = mm.get("hellinger_histogram")
    if hist:
        edges, counts = hist["edges"], hist["counts"]
        top = max(counts) or 1
        print("\n  how far each position moved:")
        for i, c in enumerate(counts):
            if c == 0:
                continue
            bar = "#" * max(1, round(40 * c / top))
            print(f"    {edges[i]:.2f}-{edges[i + 1]:.2f}  {c:>5}  {bar}")

    # The mean hides a bimodal distribution, so name the positions in the far mode:
    # a handful moving almost the whole way is a different failure from everything
    # moving a little, and only one of those explains a rare wrong-branch call.
    import numpy as _np
    from sqsketch.llm import compare as _row_bc
    bc = _np.clip(_row_bc(ref.positions.astype(_np.float64),
                          cand.positions.astype(_np.float64)), -1.0, 1.0)
    per_pos = _np.sqrt(_np.clip(1.0 - bc, 0.0, None))
    order = _np.argsort(-per_pos)
    flipped = ref.argmax != cand.argmax
    print(f"\n  the far mode: {(per_pos > 0.5).sum()} positions above h=0.5, "
          f"{(per_pos > 0.9).sum()} above 0.9")
    print(f"  {'pos':>5} {'h':>7} {'top1 flipped':>13}  reference token -> quantised")
    for i in order[:12]:
        a_, b_ = int(ref.argmax[i]), int(cand.argmax[i])
        print(f"  {i:>5} {per_pos[i]:>7.4f} {str(bool(flipped[i])):>13}  "
              f"{tok.decode([a_])!r} -> {tok.decode([b_])!r}")
    far = [int(i) for i in _np.flatnonzero(per_pos > 0.9)]

    rec = {"far_mode_positions": far,
           "n_above_0_5": int((per_pos > 0.5).sum()),
           "n_above_0_9": int((per_pos > 0.9).sum()),
           "per_position_hellinger": [round(float(x), 5) for x in per_pos],
           "model": MODEL, "gguf": f"{a.repo}/{a.file}", "positions": a.positions,
           "probe": ph, "vocab": vocab, "ppl_fp32": ppl_ref, "ppl_quant": ppl,
           "dppl_pct": dppl, "mean_hellinger": mm["mean_hellinger"],
           "top1_agreement": mm["top1_agreement"], "top1_flip_rate": flip,
           "positions_moved": mm["positions_moved"], "noise_floor": mm["noise_floor"],
           "control_hellinger": ctl["mean_hellinger"],
           "status": v.status, "severity": v.severity, "signature": v.signature,
           "hellinger_histogram": hist}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ondevice_q4km_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    ref.save(os.path.join(OUT, "qwen3-0.6b-fp32.seal.npz"))
    print(f"\nwrote outputs/ondevice_q4km_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
