"""Is the perplexity gap a property of grid quantisers, or of quantisation?

The first study measured four 4-bit methods and found all four reporting more repair
through perplexity than they delivered, by 15 to 27 points. A commenter on the
write-up asked for EXL3, and the question is better than a missing row: everything
measured so far is the same family underneath. AWQ and GPTQ choose *which weights to
protect* on a grid; NF4 and Q4_K_M choose *a better grid*; all four round to one.

EXL3 is a variant of QTIP -- trellis-coded, it does not round to a grid at all. So it
tests whether the gap is a property of rounding schemes or of lossy compression
generally. Four methods from one family agreeing was always weaker evidence than it
looked.

And `convert.py -b <bitrate>` takes a bitrate, so this traces a curve rather than
adding a point: does the gap widen or close as bits per weight drop?

Two things this run also fixes about the first one
--------------------------------------------------
The model moves up to Qwen2.5-1.5B. The first study ran on 0.5B, where AWQ and GPTQ
have far less to work with than they are designed for, and the ranking there should
not have been generalised -- the main caveat readers raised, and a fair one.

And EXL3 is measured through the *same path* as every other row. `patch_transformers()`
makes an EXL3 checkpoint load through `AutoModelForCausalLM`, so servseal's ordinary
weights-mode measurement applies unchanged. No sampling, no second measurement path to
reconcile, no control needed for the difference between them.

Ordering is by value, because a session that runs out of time should lose the rows
that matter least. The grid baselines come before `patch_transformers()` is ever
called, so nothing they measure can be affected by it.

Kaggle: script kernel, GPU T4 x2, internet on, no dataset (servseal from PyPI).
"""
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WORK = "/kaggle/working"
MODEL = "Qwen/Qwen2.5-1.5B"
POSITIONS = 500
MAXLEN, D = 96, 256
GROUP, BITS = 128, 4
EXL3_BPW = [4.0, 3.0, 5.0]        # headline first, then the ends of the curve


def sh(cmd, **kw):
    """A string runs through the shell, a list does not. `shell` never arrives twice:
    passing it in kw as well is what stopped llama.cpp building in an earlier run."""
    kw.pop("shell", None)
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def banner(t):
    print(f"\n{'=' * 88}\n{t}\n{'=' * 88}", flush=True)


# ----------------------------------------------------------------------- install

banner("install, before anything imports what it checks for")
# Two rules, both learned the expensive way.
#
# Order: transformers decides at *import* time whether bitsandbytes is available and
# caches the answer, so everything lands before the first transformers import.
#
# Constraints: a version pin only binds the resolver pass it takes part in. Listing
# `transformers<5` in one pip command and then running a second one lets the second
# lift it -- which is exactly what happened here, twice, from two different packages.
# Bundling every install into one command is a fix that depends on my remembering to
# bundle. A constraints file does not: pip applies it to every resolution it is given,
# including the editable install of a cloned repo and anything added later.
CONSTRAINTS = os.path.join(WORK, "constraints.txt")
with open(CONSTRAINTS, "w", encoding="utf-8") as fh:
    fh.write("transformers>=4.50,<5\n"       # the quantisers disagree with 5.x
             "numpy<2.4\n")                   # llmcompressor caps it; exllamav3 raised it


def pip(*args, **kw):
    return sh([sys.executable, "-m", "pip", "install", "-q", "-c", CONSTRAINTS,
               *args], **kw)


PKGS = ["servseal>=0.2.1", "transformers>=4.50,<5", "accelerate", "bitsandbytes",
        "gguf>=0.10.0", "datasets", "huggingface_hub", "safetensors",
        "llmcompressor"]
if pip(*PKGS).returncode != 0:
    print("the full set did not resolve; retrying without llmcompressor", flush=True)
    LLMC = 1
    pip(*PKGS[:-1], check=True)
else:
    LLMC = 0

# EXL3's matrix-multiply kernels use the m16n8k16 tensor-core shape, which ptxas
# refuses below sm_80: "Feature '.m16n8k16' requires .target sm_80 or higher". A T4
# is sm_75, so no free Kaggle accelerator can build it -- P100 is sm_60 and T4 is as
# high as it goes. Detected here rather than discovered twenty minutes into a compile
# that was always going to fail.
#
# An earlier run of this file reported "installed: True" on a T4 and I believed it.
# That was `pip install -e`, which registered the package without building the
# extension; the regular install compiles, and that is when the truth arrives.
EXL3_DIR = os.path.join(WORK, "exllamav3")
EXL3_OK = False
EXL3_WHY = ""
_cap = None
try:
    import subprocess as _sp
    _cap = _sp.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                   capture_output=True, text=True).stdout.strip().splitlines()[0]
    _major = int(float(_cap))
except Exception:
    _major = 0
if _major and _major < 8:
    EXL3_WHY = (f"compute capability {_cap}: EXL3 needs sm_80+ (Ampere). Rent an "
                f"A10/L4/A100 for an hour, or run experiments/exl3_standalone.py "
                f"wherever you have one.")
    print(f"skipping exllamav3 -- {EXL3_WHY}", flush=True)
elif not os.path.isdir(EXL3_DIR):
    sh(["git", "clone", "--depth", "1",
        "https://github.com/turboderp-org/exllamav3", EXL3_DIR])
if not EXL3_WHY and os.path.isdir(EXL3_DIR):
    # NOT editable. `pip install -e` registers the package through a .pth file, which
    # the interpreter reads at startup -- so an install done inside a running process
    # returns 0 and the very next `import` still raises ModuleNotFoundError. That is
    # exactly what happened, and the install reporting success is what made it hard
    # to see. A regular install puts the files in site-packages, already on sys.path.
    # It also builds the CUDA extension, which is the part that has to work anyway.
    EXL3_OK = pip(EXL3_DIR, cwd=EXL3_DIR).returncode == 0
    if EXL3_OK:
        try:
            import exllamav3                                   # noqa: F401
        except ModuleNotFoundError:
            sys.path.insert(0, EXL3_DIR)                       # belt and braces
            try:
                import exllamav3                               # noqa: F401,F811
                print("  imported from the clone rather than site-packages",
                      flush=True)
            except Exception as e:
                EXL3_OK = False
                print(f"  installed but unimportable: {type(e).__name__}: {e}",
                      flush=True)
print(f"exllamav3 installed and importable: {EXL3_OK}", flush=True)


# ------------------------------------------------------------------- environment

banner("environment")
import torch                                                       # noqa: E402

if not torch.cuda.is_available():
    sys.exit("No GPU. Settings -> Accelerator -> GPU T4 x2, then Save & Run All from "
             "the editor: ApiSaveKernelRequest has no accelerator field, so a pushed "
             "version reverts to Kaggle's default.")
CAP = torch.cuda.get_device_capability(0)
print(f"GPU {torch.cuda.get_device_name(0)} (sm_{CAP[0]}{CAP[1]})  "
      f"torch {torch.__version__}", flush=True)
if CAP[0] < 7:
    sys.exit(f"sm_{CAP[0]}{CAP[1]}: these kernels need sm_70+. Ask for a T4.")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import numpy as np                                                 # noqa: E402
import transformers                                                # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer       # noqa: E402
import servseal                                                    # noqa: E402
from servseal.probes import load_probes, probe_id                  # noqa: E402
from servseal.runner import softmax_over_probes                    # noqa: E402
from servseal.snapshot import Snapshot                             # noqa: E402
from servseal.verdict import classify                              # noqa: E402

print(f"servseal {servseal.__version__}  transformers {transformers.__version__}",
      flush=True)
if not transformers.__version__.startswith("4."):
    sys.exit(f"transformers {transformers.__version__}: the pin did not hold despite "
             f"{CONSTRAINTS}. Every row must be measured under one framework version, "
             f"so this stops rather than producing a table whose rows disagree about "
             f"the framework. Find which install moved it -- the pip output above "
             f"names every package it changed -- and add that bound to the "
             f"constraints file.")
print(f"  numpy {np.__version__}  (llmcompressor caps it below 2.4)", flush=True)
if EXL3_OK:
    import exllamav3
    print(f"  exllamav3 {getattr(exllamav3, '__version__', 'present')}", flush=True)

TEXTS = load_probes()
PROBE = probe_id(TEXTS)
TOK = AutoTokenizer.from_pretrained(MODEL)


# --------------------------------------------------------------------- measuring

def probe(model, device="cpu", cast_fp32=True):
    """servseal's own path, on the device the candidate belongs to.

    A quantised model may refuse to be moved -- asking is fine, insisting lost a row
    once. servseal >= 0.2.1 sends the probe ids to `model.device`, so wherever the
    model already sits is where the measurement happens.
    """
    if cast_fp32:
        model = model.to(torch.float32)
    try:
        model = model.to(device)
    except (ValueError, NotImplementedError) as e:
        print(f"  the model declines .to({device}): {str(e)[:90]}", flush=True)
    model = model.eval()
    return softmax_over_probes(model, TOK, TEXTS, max_positions=POSITIONS,
                               max_length=MAXLEN)


def snap(P, label, ppl):
    return Snapshot.from_distributions(
        P, probe=PROBE, model=f"{MODEL}@{label}", D=D,
        extra={"perplexity": round(ppl, 6), "probe_file": "default-v1"})


RESULTS = []


def attempt(label, fn):
    banner(label)
    t0 = time.time()
    try:
        P, ppl, device, note = fn()
    except Exception as e:
        print(traceback.format_exc()[-2500:], flush=True)
        RESULTS.append({"method": label, "h": None,
                        "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(f"  UNAVAILABLE: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    ref, ppl_ref = (REF_GPU, PPL_GPU) if device == "cuda" else (REF_CPU, PPL_CPU)
    m = ref.compare(snap(P, label, ppl))
    v = classify(m)
    dppl = 100 * (ppl - ppl_ref) / ppl_ref
    RESULTS.append({"method": label, "h": m["mean_hellinger"],
                    "top1": m["top1_agreement"], "ppl": ppl, "dppl": dppl,
                    "status": v.status, "severity": v.severity,
                    "signature": v.signature, "device": device, "note": note,
                    "seconds": round(time.time() - t0, 1)})
    print(f"  {label:<20} h={m['mean_hellinger']:.4f}  "
          f"top1={m['top1_agreement']:.3f}  ppl={ppl:.4f} ({dppl:+.2f} %)  "
          f"[{device}]  -> {v.status.upper()}/{v.severity}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    del P
    gc.collect()
    torch.cuda.empty_cache()
    _save()


def _save():
    with open(os.path.join(WORK, "exl3_curve_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"model": MODEL, "positions": POSITIONS, "probe": PROBE,
                   "ppl_fp32_cpu": PPL_CPU, "ppl_fp32_gpu": PPL_GPU,
                   "servseal": servseal.__version__,
                   "transformers": transformers.__version__,
                   "gpu": torch.cuda.get_device_name(0),
                   "exllamav3": EXL3_OK, "results": RESULTS}, fh, indent=2)


# -------------------------------------------------------------------- references

banner(f"references: {MODEL} float32, {POSITIONS} positions, both devices")
t0 = time.time()
_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
P_cpu, PPL_CPU = probe(_m, "cpu")
REF_CPU = snap(P_cpu, "fp32-cpu", PPL_CPU)
VOCAB = P_cpu.shape[1]
del P_cpu
gc.collect()
P_gpu, PPL_GPU = probe(_m, "cuda")
REF_GPU = snap(P_gpu, "fp32-gpu", PPL_GPU)
del _m, P_gpu
gc.collect()
torch.cuda.empty_cache()
_hw = REF_CPU.compare(REF_GPU)
print(f"  vocab {VOCAB:,}  ppl cpu {PPL_CPU:.4f}  gpu {PPL_GPU:.4f}  "
      f"[{time.time() - t0:.0f}s]", flush=True)
print(f"  hardware term, fp32 cpu vs gpu: h={_hw['mean_hellinger']:.6f}  "
      f"top1={_hw['top1_agreement']:.4f}", flush=True)
RESULTS.append({"method": "control: fp32 gpu vs cpu", "h": _hw["mean_hellinger"],
                "top1": _hw["top1_agreement"], "ppl": PPL_GPU,
                "dppl": 100 * (PPL_GPU - PPL_CPU) / PPL_CPU,
                "status": classify(_hw).status, "severity": classify(_hw).severity,
                "signature": "hardware", "device": "cuda",
                "note": "the floor below which no verdict here means anything"})
REF_CPU.save(os.path.join(WORK, "qwen15b_fp32_cpu.seal.npz"))
_save()


# --------------------------------------------------------------- the grid family

def _weights_only(fn, note):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    with torch.no_grad():
        fn(m)
    P, ppl = probe(m, "cpu")
    del m
    gc.collect()
    return P, ppl, "cpu", note


def naive_int4():
    def q(m):
        lv = 2 ** (BITS - 1) - 1
        for p in m.parameters():
            if p.dim() != 2:
                continue
            out, inp = p.shape
            w = torch.nn.functional.pad(p, (0, (-inp) % GROUP)).reshape(out, -1, GROUP)
            s = w.abs().amax(dim=2, keepdim=True) / lv
            s = torch.where(s > 0, s, torch.ones_like(s))
            p.copy_((torch.round(w / s) * s).reshape(out, -1)[:, :inp])
    return _weights_only(q, f"{BITS}-bit, group {GROUP}, no calibration")


def gguf(qtype):
    root = os.path.join(WORK, "llama.cpp")
    exe = os.path.join(root, "build", "bin", "llama-quantize")
    if not os.path.exists(exe):
        if not os.path.isdir(root):
            if sh(["git", "clone", "--depth", "1",
                   "https://github.com/ggml-org/llama.cpp", root]).returncode != 0:
                raise RuntimeError("llama.cpp clone failed")
        sh([sys.executable, "-m", "pip", "install", "-q", "-r",
            os.path.join(root, "requirements",
                         "requirements-convert_hf_to_gguf.txt")])
        if sh(f"cmake -S {root} -B {root}/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
              f"-DCMAKE_BUILD_TYPE=Release && cmake --build {root}/build "
              f"--target llama-quantize -j4").returncode:
            raise RuntimeError("llama-quantize did not build")
    from huggingface_hub import snapshot_download
    src = snapshot_download(MODEL, allow_patterns=[
        "config.json", "generation_config.json", "model*.safetensors*",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    f16 = os.path.join(WORK, "qwen15b-f16.gguf")
    if not os.path.exists(f16):
        if sh([sys.executable, os.path.join(root, "convert_hf_to_gguf.py"), src,
               "--outfile", f16, "--outtype", "f16"]).returncode != 0:
            raise RuntimeError("convert_hf_to_gguf failed")
    qf = os.path.join(WORK, f"qwen15b-{qtype}.gguf")
    if sh([exe, f16, qf, qtype]).returncode != 0:
        raise RuntimeError(f"llama-quantize {qtype} failed")
    q = AutoModelForCausalLM.from_pretrained(WORK, gguf_file=os.path.basename(qf),
                                             dtype=torch.float32)
    P, ppl = probe(q, "cpu")
    del q
    gc.collect()
    os.remove(qf)
    return P, ppl, "cpu", f"llama.cpp {qtype}, dequantised to fp32"


def bnb_nf4():
    from transformers.utils import is_bitsandbytes_available
    if not is_bitsandbytes_available():
        raise RuntimeError("transformers does not see bitsandbytes; it must be "
                           "installed before transformers is first imported")
    from transformers import BitsAndBytesConfig
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float32,
                             bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg,
                                             device_map="cuda")
    P, ppl = probe(m, "cuda", cast_fp32=False)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return P, ppl, "cuda", "NF4, double quantisation, float32 compute"


# --------------------------------------------------------------- the trellis one

_PATCHED = [False]


def exl3(bpw):
    """EXL3 at one bitrate. Trellis-coded (a QTIP variant), not a grid.

    `patch_transformers()` teaches transformers the EXL3 quantisation format, after
    which the checkpoint loads through AutoModelForCausalLM and servseal measures it
    exactly as it measures every other row. That is why the grid rows above run
    first: none of them is measured under a patched transformers.
    """
    if not EXL3_OK:
        raise RuntimeError(EXL3_WHY or "exllamav3 is not importable here; see the "
                           "install block above for whether it built or only "
                           "registered")
    from huggingface_hub import snapshot_download
    src = snapshot_download(MODEL, allow_patterns=[
        "config.json", "generation_config.json", "model*.safetensors*",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    out = os.path.join(WORK, f"exl3-{bpw}bpw")
    work = os.path.join(WORK, "exl3-work")
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    conv = os.path.join(EXL3_DIR, "convert.py")
    if not os.path.exists(conv):
        raise RuntimeError(f"{conv} is not there; the clone layout changed")
    if sh([sys.executable, conv, "-i", src, "-o", out, "-w", work,
           "-b", str(bpw)]).returncode != 0:
        raise RuntimeError(f"convert.py failed at {bpw} bpw")

    if not _PATCHED[0]:
        from exllamav3.integration.transformers import patch_transformers
        patch_transformers()
        _PATCHED[0] = True
        print("  transformers patched for the EXL3 format", flush=True)
    q = AutoModelForCausalLM.from_pretrained(out, device_map="cuda")
    P, ppl = probe(q, "cuda", cast_fp32=False)
    del q
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    return P, ppl, "cuda", f"EXL3 {bpw} bpw, trellis (QTIP variant)"


def compressed(scheme):
    if LLMC != 0:
        raise RuntimeError("llmcompressor did not install")
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.awq import AWQModifier
    out = os.path.join(WORK, f"qwen-{scheme}")
    shutil.rmtree(out, ignore_errors=True)
    ds = None
    for repo, cfg in (("Salesforce/wikitext", "wikitext-2-raw-v1"),
                      ("wikitext", "wikitext-2-raw-v1")):
        try:
            ds = load_dataset(repo, cfg, split="train")
            break
        except Exception as e:
            print(f"  {repo}: {type(e).__name__}", flush=True)
    if ds is None:
        raise RuntimeError("no calibration corpus")
    calib = ds.filter(lambda r: len(r["text"]) > 200).select(range(128))
    assert not (set(t[:80] for t in TEXTS) & set(r[:80] for r in calib["text"])), \
        "the calibration set overlaps the probe set"

    def fresh():
        return (AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
                if scheme == "awq" else
                GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]))

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             device_map="cuda")
    last = None
    for pipe in (None, "basic"):
        try:
            kw = {} if pipe is None else {"pipeline": pipe}
            oneshot(model=m, dataset=calib, recipe=fresh(), max_seq_length=512,
                    num_calibration_samples=128, output_dir=out, **kw)
            last = None
            break
        except Exception as e:
            last = e
            print(f"  {pipe or 'default'} pipeline failed: {type(e).__name__}",
                  flush=True)
            shutil.rmtree(out, ignore_errors=True)
    if last is not None:
        raise last
    del m
    gc.collect()
    torch.cuda.empty_cache()
    grid = "grid unread"
    try:
        qc = json.load(open(os.path.join(out, "config.json"),
                            encoding="utf-8")).get("quantization_config", {})
        for g in (qc.get("config_groups") or {}).values():
            w = g.get("weights", {})
            grid = f"{w.get('num_bits')}-bit, group {w.get('group_size')}"
            break
        print(f"  checkpoint says: {grid}", flush=True)
    except Exception:
        pass
    q = AutoModelForCausalLM.from_pretrained(out, device_map="cuda")
    P, ppl = probe(q, "cuda", cast_fp32=False)
    del q
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(out, ignore_errors=True)
    return P, ppl, "cuda", f"{scheme.upper()} {grid}, 128 calibration samples"


# ------------------------------------------------------------------------- run

ALL = ([("naive int4 g128", naive_int4),
        ("GGUF Q4_K_M", lambda: gguf("Q4_K_M")),
        ("bnb NF4", bnb_nf4)]
       + [(f"EXL3 {b} bpw", (lambda b=b: exl3(b))) for b in EXL3_BPW]
       + [("AWQ W4A16", lambda: compressed("awq")),
          ("GPTQ W4A16", lambda: compressed("gptq"))])

ONLY = [m.strip() for m in os.environ.get("SERVSEAL_ONLY", "").split(",") if m.strip()]
for _label, _fn in ALL:
    if ONLY and _label not in ONLY:
        continue
    attempt(_label, _fn)


# ---------------------------------------------------------------------- verdict

banner(f"THE CURVE  ({MODEL}, {POSITIONS} positions, fp32 ppl {PPL_CPU:.4f})")
ok = [r for r in RESULTS if r.get("h") is not None]
print(f"{'method':<22} {'mean_h':>8} {'top1':>7} {'d_ppl%':>10} {'dev':>5}  verdict")
for r in RESULTS:
    if r.get("h") is None:
        print(f"{r['method']:<22} {'-':>8} {'-':>7} {'-':>10} {'-':>5}  "
              f"{r['error'][:52]}")
    else:
        print(f"{r['method']:<22} {r['h']:>8.4f} {r['top1']:>7.3f} "
              f"{r['dppl']:>+10.2f} {r['device']:>5}  "
              f"{r['status'].upper()}/{r['severity']}")

naive = next((r for r in ok if r["method"] == "naive int4 g128"), None)
if naive:
    print(f"\nagainst the same 4-bit grid with no search "
          f"(h={naive['h']:.4f}, ppl {naive['dppl']:+.2f} %):")
    print(f"{'method':<18} {'behaviour':>10} {'perplexity':>12} {'apart':>8}")
    for r in ok:
        if r["method"] in ("naive int4 g128", "control: fp32 gpu vs cpu"):
            continue
        dh = 100 * (1 - r["h"] / naive["h"])
        dp = (100 * (1 - abs(r["dppl"]) / abs(naive["dppl"]))
              if naive["dppl"] else float("nan"))
        print(f"{r['method']:<18} {dh:>9.0f} % {dp:>11.0f} % {abs(dh - dp):>6.0f} pts")
    print("\nIf the trellis rows sit apart by about as much as the grid rows, the gap\n"
          "is a property of lossy compression rather than of rounding. If they do\n"
          "not, that is the more interesting answer and the first study overclaimed.")

_save()
for junk in ("llama.cpp", "exllamav3", "exl3-work"):
    shutil.rmtree(os.path.join(WORK, junk), ignore_errors=True)
for f in os.listdir(WORK):
    if f.endswith(".gguf") or f.startswith("exl3-"):
        p = os.path.join(WORK, f)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)

print(f"\n{len(ok)}/{len(RESULTS)} measured")
if not any(r["method"].startswith("EXL3") and r.get("h") is not None for r in RESULTS):
    print("  NOTE: no EXL3 row. The question this run was pushed for is unanswered;\n"
          "        read the exllamav3 install and convert output above.")
