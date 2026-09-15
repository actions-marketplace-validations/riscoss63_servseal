"""Measure what an EXL3 quantisation costs, on any machine with an Ampere GPU.

EXL3 is a QTIP variant -- trellis-coded rather than rounded to a grid, which is what
makes it worth measuring: every method in the published study is a grid method, and
four of those agreeing is weaker evidence than it looks. If the gap between what
perplexity reports and what actually moved is a property of rounding, EXL3 should not
show it. If it is a property of lossy compression, EXL3 should.

Not a Kaggle kernel, because it cannot be one. EXL3's kernels use the m16n8k16
tensor-core shape and ptxas refuses it below sm_80; Kaggle's free accelerators stop
at a T4 (sm_75). An hour on a rented A10 or L4 costs about a dollar, which is the
cheapest way to answer this.

    pip install "servseal[model]" torch
    git clone https://github.com/turboderp-org/exllamav3 && pip install ./exllamav3
    python exl3_standalone.py --model Qwen/Qwen2.5-1.5B --bpw 4.0 3.0 5.0

Every row is measured through servseal's ordinary weights-mode path: after
`patch_transformers()`, an EXL3 checkpoint loads through `AutoModelForCausalLM`, so
this compares like with like against the same float32 reference and the same probe
positions as the published table -- no sampling, no second measurement path.

If you already have EXL3 quants and a reference, `--skip-convert` takes directories
you made yourself.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from servseal.probes import load_probes, probe_id
from servseal.runner import softmax_over_probes
from servseal.snapshot import Snapshot
from servseal.verdict import classify

MAXLEN, D = 96, 256
GROUP, BITS = 128, 4


def need_ampere():
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. EXL3 needs one.")
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    if cap[0] < 8:
        sys.exit(f"sm_{cap[0]}{cap[1]}: EXL3's kernels use m16n8k16, which ptxas "
                 f"refuses below sm_80. Rent an A10, L4 or A100 -- about a dollar "
                 f"for the hour this takes.")
    # float32 matmuls must stay float32: TF32 has bfloat16's mantissa, and on Ampere
    # it is on by default, which would put the reference itself in question
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def probe(model, tok, texts, positions, device="cpu", cast_fp32=True):
    if cast_fp32:
        model = model.to(torch.float32)
    try:
        model = model.to(device)
    except (ValueError, NotImplementedError) as e:
        print(f"  the model declines .to({device}): {str(e)[:90]}")
    return softmax_over_probes(model.eval(), tok, texts, max_positions=positions,
                               max_length=MAXLEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--bpw", nargs="+", type=float, default=[4.0, 3.0, 5.0])
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--exllamav3", default="./exllamav3",
                    help="the cloned repo, for convert.py")
    ap.add_argument("--skip-convert", nargs="*", default=None, metavar="DIR",
                    help="EXL3 directories you already have, instead of converting")
    ap.add_argument("--work", default="./exl3-work")
    ap.add_argument("--out", default="exl3_standalone_results.json")
    a = ap.parse_args()

    need_ampere()
    texts = load_probes()
    probe_hash = probe_id(texts)
    tok = AutoTokenizer.from_pretrained(a.model)
    results = []

    print(f"\nreference: {a.model} float32 on CPU, {a.positions} positions")
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    P, ppl_ref = probe(m, tok, texts, a.positions, "cpu")
    ref = Snapshot.from_distributions(
        P, probe=probe_hash, model=f"{a.model}@fp32-cpu", D=D,
        extra={"perplexity": round(ppl_ref, 6), "probe_file": "default-v1"})
    vocab = P.shape[1]
    del P
    gc.collect()

    # the same reference on the GPU, so a GPU-resident candidate is not being
    # compared against another device's arithmetic without anyone saying so
    P, ppl_gpu = probe(m, tok, texts, a.positions, "cuda")
    ref_gpu = Snapshot.from_distributions(
        P, probe=probe_hash, model=f"{a.model}@fp32-gpu", D=D,
        extra={"perplexity": round(ppl_gpu, 6), "probe_file": "default-v1"})
    del m, P
    gc.collect()
    torch.cuda.empty_cache()
    hw = ref.compare(ref_gpu)
    print(f"  vocab {vocab:,}  ppl {ppl_ref:.4f} cpu / {ppl_gpu:.4f} gpu  "
          f"[{time.time() - t0:.0f}s]")
    print(f"  control, fp32 cpu vs gpu: h={hw['mean_hellinger']:.6f}  "
          f"top1={hw['top1_agreement']:.4f}   <- the floor for everything below")
    ref.save("reference_fp32_cpu.seal.npz")

    def record(label, P, ppl, against, ppl_base, note):
        m_ = against.compare(Snapshot.from_distributions(
            P, probe=probe_hash, model=f"{a.model}@{label}", D=D,
            extra={"perplexity": round(ppl, 6), "probe_file": "default-v1"}))
        v = classify(m_)
        dppl = 100 * (ppl - ppl_base) / ppl_base
        results.append({"method": label, "h": m_["mean_hellinger"],
                        "top1": m_["top1_agreement"], "ppl": ppl, "dppl": dppl,
                        "status": v.status, "severity": v.severity, "note": note})
        print(f"  {label:<18} h={m_['mean_hellinger']:.4f}  "
              f"top1={m_['top1_agreement']:.3f}  ppl {dppl:+.2f} %  "
              f"-> {v.status.upper()}/{v.severity}")
        json.dump({"model": a.model, "positions": a.positions, "probe": probe_hash,
                   "ppl_fp32_cpu": ppl_ref, "ppl_fp32_gpu": ppl_gpu,
                   "gpu": torch.cuda.get_device_name(0),
                   "control_cpu_vs_gpu": hw["mean_hellinger"], "results": results},
                  open(a.out, "w", encoding="utf-8"), indent=2)

    results.append({"method": "control: fp32 gpu vs cpu", "h": hw["mean_hellinger"],
                    "top1": hw["top1_agreement"], "ppl": ppl_gpu,
                    "dppl": 100 * (ppl_gpu - ppl_ref) / ppl_ref,
                    "status": classify(hw).status, "severity": classify(hw).severity,
                    "note": "the floor below which no verdict here means anything"})

    # The baseline the whole comparison rests on: the same 4-bit group-128 grid EXL3
    # is measured against, with none of the search. Without it there is nothing to
    # say "the method bought this much" relative to.
    print("\nbaseline: 4-bit group-128, no calibration")
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    lv = 2 ** (BITS - 1) - 1
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() != 2:
                continue
            out, inp = p.shape
            w = torch.nn.functional.pad(p, (0, (-inp) % GROUP)).reshape(out, -1, GROUP)
            s = w.abs().amax(dim=2, keepdim=True) / lv
            s = torch.where(s > 0, s, torch.ones_like(s))
            p.copy_((torch.round(w / s) * s).reshape(out, -1)[:, :inp])
    P, ppl = probe(m, tok, texts, a.positions, "cpu")
    record("naive int4 g128", P, ppl, ref, ppl_ref, "4-bit, group 128, no search")
    del m, P
    gc.collect()

    from exllamav3.integration.transformers import patch_transformers
    patch_transformers()
    print("\ntransformers patched for the EXL3 format")

    dirs = []
    if a.skip_convert is not None:
        dirs = [(os.path.basename(d.rstrip("/\\")), d) for d in a.skip_convert]
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(a.model, allow_patterns=[
            "config.json", "generation_config.json", "model*.safetensors*",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
        conv = os.path.join(a.exllamav3, "convert.py")
        if not os.path.exists(conv):
            sys.exit(f"{conv} not found -- clone the repo or pass --exllamav3")
        for b in a.bpw:
            out = os.path.abspath(f"./exl3-{b}bpw")
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(a.work, exist_ok=True)
            print(f"\nconverting to {b} bpw")
            if subprocess.run([sys.executable, conv, "-i", src, "-o", out,
                               "-w", a.work, "-b", str(b)]).returncode != 0:
                print(f"  convert.py failed at {b} bpw; skipping")
                continue
            dirs.append((f"EXL3 {b} bpw", out))

    for label, d in dirs:
        print(f"\n{label}  ({d})")
        try:
            q = AutoModelForCausalLM.from_pretrained(d, device_map="cuda")
            P, ppl = probe(q, tok, texts, a.positions, "cuda", cast_fp32=False)
            record(label, P, ppl, ref_gpu, ppl_gpu, "trellis, QTIP variant")
            del q, P
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  UNAVAILABLE: {type(e).__name__}: {str(e)[:200]}")
            results.append({"method": label, "h": None,
                            "error": f"{type(e).__name__}: {str(e)[:300]}"})

    ok = [r for r in results if r.get("h") is not None]
    base = next((r for r in ok if r["method"] == "naive int4 g128"), None)
    print(f"\n{'=' * 78}\n{a.model}, {a.positions} positions\n{'=' * 78}")
    print(f"{'method':<20} {'mean_h':>8} {'top1':>7} {'d_ppl%':>10}")
    for r in results:
        if r.get("h") is None:
            print(f"{r['method']:<20} {'-':>8} {'-':>7} {'-':>10}  {r['error'][:40]}")
        else:
            print(f"{r['method']:<20} {r['h']:>8.4f} {r['top1']:>7.3f} "
                  f"{r['dppl']:>+10.2f}")
    if base:
        print(f"\nagainst the same grid with no search "
              f"(h={base['h']:.4f}, ppl {base['dppl']:+.2f} %):")
        for r in ok:
            if r["method"] in ("naive int4 g128", "control: fp32 gpu vs cpu"):
                continue
            dh = 100 * (1 - r["h"] / base["h"])
            dp = 100 * (1 - abs(r["dppl"]) / abs(base["dppl"]))
            print(f"  {r['method']:<18} behaviour {dh:>5.0f} %   "
                  f"perplexity {dp:>5.0f} %   {abs(dh - dp):>4.0f} pts apart")
        print("\nA grid method at this size sat 18 points apart (GGUF Q4_K_M) and "
              "another\nsat 1 (bitsandbytes NF4), so the question is open: if the "
              "trellis rows\nsit apart too, the gap is about lossy compression. If "
              "not, it is about\nrounding, and the published claim needs narrowing "
              "again.")
    print(f"\nwrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
