"""What the calibration in a real quantiser buys, measured behaviourally.

Option A quantised by arithmetic alone: round to bfloat16, scale per tensor, per
channel, per group of 128. The last is the grid AWQ and GPTQ share -- 4 bits, group
128 -- without the part that makes them methods: a search, driven by calibration
data, for which weights to protect. On Qwen2.5-0.5B the naive version measured a mean
Hellinger of 0.2876 and a perplexity 47 % worse.

The question is one number wide: how much of that behavioural distance does the
search remove, and does perplexity credit it with the same gain?

Where each candidate is measured, and why there are two references
-----------------------------------------------------------------
A 4-bit checkpoint runs on its own GPU kernels; forcing it to float32 on a CPU would
measure a model nobody deploys. But the canonical reference is computed in float32 on
a CPU, because that is the only arithmetic that reproduces bit-for-bit. Comparing a
GPU candidate against a CPU reference silently folds the hardware into the verdict.

So both references are computed, the difference between them is reported as a control
row, and every candidate is compared against the reference on its own device. The
control is a measurement worth having on its own: it is the floor below which no
verdict on this machine means anything.

Methods, in descending order of how reliably they install
---------------------------------------------------------
  bitsandbytes NF4 .... what most people mean by "4-bit" on Hugging Face; in
                        transformers itself, nothing to build
  bitsandbytes int8 ... LLM.int8, the outlier-aware mixed-precision scheme
  GGUF Q4_K_M/Q8_0 .... llama.cpp k-quants, the format local deployments run
  AWQ, GPTQ ........... via llm-compressor, which is maintained; autoawq is archived
                        and gptqmodel and this transformers disagree about
                        masking_utils -- both were tried and both are recorded
  naive int4 g128 ..... Option A's baseline, recomputed here, same machine
  int8 per-channel .... the safe option, for scale

Each method installs and runs inside its own guard. A toolchain that will not build
records why and the rest continue: a rerun costs a manual accelerator change, so
partial results from one session beat a clean failure.

Kaggle: script kernel, GPU T4 x2, internet on, no dataset (servseal from PyPI).
"""
import gc
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
MODEL = "Qwen/Qwen2.5-0.5B"
POSITIONS = 500          # identical to Option A, so the two tables compose
MAXLEN, D = 96, 256
GROUP, BITS = 128, 4


def sh(cmd, **kw):
    """Run a command. A string runs through the shell, a list does not.

    `shell` is set here and must not also arrive in kw -- passing it twice is what
    stopped llama.cpp from building on the first attempt, and took both GGUF rows
    with it.
    """
    kw.pop("shell", None)
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def banner(t):
    print(f"\n{'=' * 88}\n{t}\n{'=' * 88}", flush=True)


# ------------------------------------------------------------------ environment

banner("install, before anything imports what it checks for")
# Order is the whole point of this block, and getting it wrong cost a whole session.
# transformers decides at *import* time whether bitsandbytes is available and caches
# the answer, so installing it afterwards produces "requires the latest version of
# bitsandbytes" while the package sits there installed. Everything therefore lands
# before the first transformers import, in one command so pip resolves the set
# together rather than letting each install drag the previous one's pins around.
#
#   servseal >= 0.2.1  earlier releases cannot snapshot a model on a GPU at all
#   transformers < 5   the image served 4.57.6 one session and 5.0.0 the next, and
#                      the quantisers disagree with both in different ways
#   gguf               transformers needs it to read a GGUF checkpoint; llama.cpp's
#                      own requirements.txt did not bring it
#   huggingface_hub    unpinned so pip may raise it: llm-compressor wants a symbol
#                      (_validate_relative_filename) the image's version lacks
#   llmcompressor      in the SAME command, which is the whole point: it was
#                      installed separately last session "to isolate it" and
#                      silently lifted the transformers pin to 5.14, under which
#                      its own AST tracer cannot rewrite Qwen2's forward. A pin
#                      only binds the resolver pass it takes part in.
PKGS = ["servseal>=0.2.1", "transformers>=4.50,<5", "accelerate", "bitsandbytes",
        "gguf>=0.10.0", "datasets", "huggingface_hub", "llmcompressor"]
if sh([sys.executable, "-m", "pip", "install", "-q", *PKGS]).returncode != 0:
    # llmcompressor is the least essential row; if the set will not resolve with
    # it, drop it rather than lose the six that do not need it
    print("the full set did not resolve; retrying without llmcompressor", flush=True)
    LLMC = 1
    sh([sys.executable, "-m", "pip", "install", "-q", *PKGS[:-1]], check=True)
else:
    LLMC = 0

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
# TF32 turns float32 matmuls into something with bfloat16's mantissa. Turing has no
# TF32 so this is a no-op on a T4, but it must not be left to the hardware lottery.
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
    sys.exit(f"transformers {transformers.__version__}: the pin did not hold. Every "
             f"row must be measured under one framework version, and the quantisers "
             f"need 4.x. Stopping now rather than producing a table whose rows were "
             f"measured under different transformers.")
for mod in ("bitsandbytes", "gguf", "huggingface_hub", "llmcompressor"):
    try:
        m = __import__(mod)
        print(f"  {mod:<18} {getattr(m, '__version__', 'present')}", flush=True)
    except Exception as e:
        print(f"  {mod:<18} ABSENT ({type(e).__name__})", flush=True)
from transformers.utils import is_bitsandbytes_available       # noqa: E402
print(f"  transformers sees bitsandbytes: {is_bitsandbytes_available()}", flush=True)
TEXTS = load_probes()
PROBE = probe_id(TEXTS)
TOK = AutoTokenizer.from_pretrained(MODEL)


# --------------------------------------------------------------------- measuring

def probe(model, device="cpu", cast_fp32=True):
    """The product's own measurement path, on the device the candidate belongs to.

    A quantised model may refuse to be moved -- an 8-bit bitsandbytes model raises
    "`.to` is not supported for `8-bit` models", because device_map already placed
    it and its kernels are bound to that placement. Asking is fine; insisting is
    what lost that row. servseal 0.2.1 sends the probe ids to `model.device`, so
    wherever the model already sits is where the measurement happens.
    """
    if cast_fp32:
        model = model.to(torch.float32)
    try:
        model = model.to(device)
    except (ValueError, NotImplementedError) as e:
        print(f"  the model declines .to({device}): {str(e)[:90]}", flush=True)
    model = model.eval()
    P, ppl = softmax_over_probes(model, TOK, TEXTS, max_positions=POSITIONS,
                                 max_length=MAXLEN)
    return P, ppl


def snap(P, label, ppl):
    return Snapshot.from_distributions(
        P, probe=PROBE, model=f"{MODEL}@{label}", D=D,
        extra={"perplexity": round(ppl, 6), "probe_file": "default-v1"})


RESULTS = []


def record(label, ref, ppl_ref, P, ppl, device, note=""):
    m = ref.compare(snap(P, label, ppl))
    v = classify(m)
    dppl = 100 * (ppl - ppl_ref) / ppl_ref
    row = {"method": label, "h": m["mean_hellinger"], "top1": m["top1_agreement"],
           "ppl": ppl, "dppl": dppl, "status": v.status, "severity": v.severity,
           "signature": v.signature, "device": device, "note": note}
    RESULTS.append(row)
    print(f"  {label:<22} h={row['h']:.4f}  top1={row['top1']:.3f}  "
          f"ppl={ppl:.4f} ({dppl:+.2f} %)  [{device}]  "
          f"-> {v.status.upper()}/{v.severity}", flush=True)
    return row


def attempt(label, fn):
    """One method. A toolchain that will not build must not take the rest."""
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
    record(label, ref, ppl_ref, P, ppl, device, note)
    del P
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  [{time.time() - t0:.0f}s]", flush=True)


# ------------------------------------------------------------------- references

banner(f"references: {MODEL} float32, {POSITIONS} positions, on both devices")
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
print(f"  hardware term, fp32 cpu vs fp32 gpu: h={_hw['mean_hellinger']:.6f}  "
      f"top1={_hw['top1_agreement']:.4f}", flush=True)
RESULTS.append({"method": "control: fp32 gpu vs cpu", "h": _hw["mean_hellinger"],
                "top1": _hw["top1_agreement"], "ppl": PPL_GPU,
                "dppl": 100 * (PPL_GPU - PPL_CPU) / PPL_CPU,
                "status": classify(_hw).status, "severity": classify(_hw).severity,
                "signature": "hardware", "device": "cuda",
                "note": "the floor below which no verdict here means anything"})
REF_CPU.save(os.path.join(WORK, "qwen_fp32_cpu.seal.npz"))


# ------------------------------------------------------------------- the methods

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


def int8_per_channel():
    def q(m):
        for p in m.parameters():
            if p.dim() < 2:
                continue
            s = p.abs().amax(dim=tuple(range(1, p.dim())), keepdim=True) / 127
            s = torch.where(s > 0, s, torch.ones_like(s))
            p.copy_(torch.round(p / s) * s)
    return _weights_only(q, "8-bit, one scale per output channel")


def bnb(kind):
    """bitsandbytes, through transformers. Runs on its own kernels, so on the GPU."""
    from transformers.utils import is_bitsandbytes_available
    if not is_bitsandbytes_available():
        raise RuntimeError("transformers does not see bitsandbytes; it must be "
                           "installed before transformers is first imported")
    from transformers import BitsAndBytesConfig
    if kind == "nf4":
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float32,
                                 bnb_4bit_use_double_quant=True)
        note = "NF4, double quantisation, float32 compute"
    else:
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        note = "LLM.int8, outlier-aware mixed precision"
    m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=cfg,
                                             device_map="cuda")
    P, ppl = probe(m, "cuda", cast_fp32=False)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return P, ppl, "cuda", note


def compressed(scheme):
    """AWQ or GPTQ via llm-compressor, which is maintained; autoawq is archived."""
    if LLMC != 0:
        raise RuntimeError("llmcompressor did not install (see the install block)")
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    # The deprecated path first, deliberately. `modifiers.transform.awq.AWQModifier`
    # is not a renamed class but a different one: it rejects targets, scheme and
    # ignore outright ("Extra inputs are not permitted"). The old one takes them and
    # only warns, so the warning is the cheaper problem.
    try:
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.transform.awq import AWQModifier

    out = os.path.join(WORK, f"qwen-{scheme}")
    shutil.rmtree(out, ignore_errors=True)
    # Calibration text, and deliberately NOT the probe set: calibrating a quantiser
    # on the same text the verdict is measured over would be fitting the method to
    # its own exam, and would flatter exactly the number this run exists to report.
    #
    # Several ids are tried because `datasets` now requires a full namespace/name
    # and the bare "wikitext" that worked for years raises HfUriError -- which is
    # what cost these two rows last session.
    ds = None
    for repo, cfg in (("Salesforce/wikitext", "wikitext-2-raw-v1"),
                      ("wikitext", "wikitext-2-raw-v1"),
                      ("stas/openwebtext-10k", None)):
        try:
            ds = (load_dataset(repo, cfg, split="train") if cfg
                  else load_dataset(repo, split="train"))
            print(f"  calibration from {repo}", flush=True)
            break
        except Exception as e:
            print(f"  {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    if ds is None:
        raise RuntimeError("no calibration corpus could be loaded")
    calib = ds.filter(lambda r: len(r["text"]) > 200).select(range(128))
    assert not (set(t[:80] for t in TEXTS) & set(r[:80] for r in calib["text"])), \
        "the calibration set overlaps the probe set"
    def fresh():
        """A modifier per attempt: oneshot initialises it, and a second pass over
        the same object raises "Cannot initialize a modifier that has already been
        initialized" -- so the fallback pipeline never got a chance to run."""
        return (AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
                if scheme == "awq" else
                GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]))

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             device_map="cuda")
    # The default pipeline rewrites the model's forward through an AST pass to trace
    # it layer by layer. That pass is version-sensitive -- it failed on Qwen2 with
    # "name 'Union' is not defined", its own recompiled source missing the module's
    # typing imports. "basic" skips tracing entirely and keeps the whole model
    # resident, which a 0.5B on a T4 can afford, so a tracer that will not run is a
    # slower path rather than a missing row.
    last = None
    for pipe in (None, "basic"):
        try:
            kw = {} if pipe is None else {"pipeline": pipe}
            oneshot(model=m, dataset=calib, recipe=fresh(), max_seq_length=512,
                    num_calibration_samples=128, output_dir=out, **kw)
            print(f"  quantised with the "
                  f"{pipe or 'default (sequential)'} pipeline", flush=True)
            last = None
            break
        except Exception as e:
            last = e
            print(f"  {pipe or 'default'} pipeline failed: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            shutil.rmtree(out, ignore_errors=True)
    if last is not None:
        raise last
    del m
    gc.collect()
    torch.cuda.empty_cache()
    # What grid did it actually use? The naive baseline is 4-bit group-128 by
    # construction; a W4A16 preset is *documented* as the same, and the comparison
    # rests on it, so it is read back from the checkpoint rather than asserted in a
    # sentence. Whatever it says goes into the row.
    grid = "grid unread"
    try:
        cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
        qc = cfg.get("quantization_config", {})
        groups = qc.get("config_groups") or {}
        for g in groups.values():
            w = g.get("weights", {})
            grid = (f"{w.get('num_bits')}-bit, group {w.get('group_size')}, "
                    f"{'symmetric' if w.get('symmetric') else 'asymmetric'}")
            break
        print(f"  checkpoint says: {grid}", flush=True)
        if str(w.get("group_size")) != str(GROUP):
            print(f"  NOTE: group {w.get('group_size')} != the baseline's {GROUP}; "
                  f"the 'same grid' comparison is not exact for this row",
                  flush=True)
    except Exception as e:
        print(f"  could not read the checkpoint's grid: {type(e).__name__}",
              flush=True)

    q = AutoModelForCausalLM.from_pretrained(out, device_map="cuda")
    P, ppl = probe(q, "cuda", cast_fp32=False)
    del q
    gc.collect()
    torch.cuda.empty_cache()
    return P, ppl, "cuda", f"{scheme.upper()} {grid}, 128 calibration samples"


def gguf(qtype):
    """llama.cpp k-quants, read back through transformers' GGUF dequantiser.

    Dequantised to float32 on the CPU, which is what the reference is, so this row
    measures the quantisation and not llama.cpp's kernels.
    """
    root = os.path.join(WORK, "llama.cpp")
    exe = os.path.join(root, "build", "bin", "llama-quantize")
    if not os.path.exists(exe):
        if not os.path.isdir(root):
            if sh(["git", "clone", "--depth", "1",
                   "https://github.com/ggml-org/llama.cpp", root]).returncode != 0:
                raise RuntimeError("llama.cpp clone failed")
        # only the converter's own needs; gguf itself came with the block above
        sh([sys.executable, "-m", "pip", "install", "-q", "-r",
            os.path.join(root, "requirements/requirements-convert_hf_to_gguf.txt")])
        if sh(f"cmake -S {root} -B {root}/build -DLLAMA_CURL=OFF "
              f"-DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release && "
              f"cmake --build {root}/build --target llama-quantize -j4").returncode:
            raise RuntimeError("llama-quantize did not build")
    if not os.path.exists(exe):
        raise RuntimeError(f"built, but {exe} is not there")

    from huggingface_hub import snapshot_download
    src = snapshot_download(MODEL, allow_patterns=[
        "config.json", "generation_config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    f16 = os.path.join(WORK, "qwen-f16.gguf")
    if not os.path.exists(f16):
        if sh([sys.executable, os.path.join(root, "convert_hf_to_gguf.py"), src,
               "--outfile", f16, "--outtype", "f16"]).returncode != 0:
            raise RuntimeError("convert_hf_to_gguf failed")
    qf = os.path.join(WORK, f"qwen-{qtype}.gguf")
    if sh([exe, f16, qf, qtype]).returncode != 0:
        raise RuntimeError(f"llama-quantize {qtype} failed")
    q = AutoModelForCausalLM.from_pretrained(WORK, gguf_file=os.path.basename(qf),
                                             dtype=torch.float32)
    P, ppl = probe(q, "cpu")
    del q
    gc.collect()
    return P, ppl, "cpu", f"llama.cpp {qtype}, dequantised to fp32"


# ------------------------------------------------------------------------- run

# Empty by default: every row in one session under one pin, because a table whose
# rows were measured under different framework versions is the exact defect this
# study reports about perplexity. Set SERVSEAL_ONLY="bnb int8,AWQ W4A16" in the
# editor to re-measure a subset once a coherent baseline exists.
DEFAULT_ONLY = ""
ONLY = [m.strip() for m in os.environ.get("SERVSEAL_ONLY", DEFAULT_ONLY).split(",")
        if m.strip()]
ALL = [("int8 per-channel", int8_per_channel),
       ("naive int4 g128", naive_int4),
       ("bnb NF4", lambda: bnb("nf4")),
       ("bnb int8", lambda: bnb("int8")),
       ("GGUF Q8_0", lambda: gguf("Q8_0")),
       ("GGUF Q4_K_M", lambda: gguf("Q4_K_M")),
       ("AWQ W4A16", lambda: compressed("awq")),
       ("GPTQ W4A16", lambda: compressed("gptq"))]
if ONLY:
    print(f"\nSERVSEAL_ONLY={ONLY}: running only these; the other rows are "
          f"already measured and committed", flush=True)
for _label, _fn in ALL:
    if ONLY and _label not in ONLY:
        continue
    attempt(_label, _fn)


# ---------------------------------------------------------------------- verdict

banner(f"WHAT A REAL QUANTISER BUYS  ({MODEL}, {POSITIONS} positions, "
       f"fp32 ppl {PPL_CPU:.4f})")
ok = [r for r in RESULTS if r.get("h") is not None]
print(f"{'method':<26} {'mean_h':>8} {'top1':>7} {'d_ppl%':>10} {'dev':>5}  verdict")
for r in RESULTS:
    if r.get("h") is None:
        print(f"{r['method']:<26} {'-':>8} {'-':>7} {'-':>10} {'-':>5}  "
              f"{r['error'][:60]}")
    else:
        print(f"{r['method']:<26} {r['h']:>8.4f} {r['top1']:>7.3f} "
              f"{r['dppl']:>+10.2f} {r['device']:>5}  "
              f"{r['status'].upper()}/{r['severity']}")

naive = next((r for r in ok if r["method"] == "naive int4 g128"), None)
if naive:
    print(f"\nagainst the same 4-bit grid without a search "
          f"(h={naive['h']:.4f}, ppl {naive['dppl']:+.2f} %):")
    for name in ("bnb NF4", "GGUF Q4_K_M", "AWQ W4A16", "GPTQ W4A16"):
        got = next((r for r in ok if r["method"] == name), None)
        if not got:
            continue
        dh = 100 * (1 - got["h"] / naive["h"])
        dp = 100 * (1 - abs(got["dppl"]) / abs(naive["dppl"])) if naive["dppl"] else 0
        flag = ("   <- the two do not agree" if abs(dh - dp) > 15 else "")
        print(f"  {name:<14} behaviour {dh:>+6.0f} %   perplexity {dp:>+6.0f} %{flag}")
    print("\nA method that removes 90 % of the perplexity gap and 60 % of the\n"
          "behavioural distance has not removed 90 % of the change. Which of those\n"
          "two numbers a deployment cares about is the whole question.")

with open(os.path.join(WORK, "quant_real_results.json"), "w", encoding="utf-8") as fh:
    json.dump({"model": MODEL, "positions": POSITIONS, "probe": PROBE,
               "ppl_fp32_cpu": PPL_CPU, "ppl_fp32_gpu": PPL_GPU,
               "servseal": servseal.__version__,
               "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "results": RESULTS}, fh, indent=2)
print(f"\nwrote quant_real_results.json ({len(ok)}/{len(RESULTS)} measured)")

for junk in ("qwen-awq", "qwen-gptq", "llama.cpp"):
    shutil.rmtree(os.path.join(WORK, junk), ignore_errors=True)
for f in os.listdir(WORK):
    if f.endswith(".gguf"):
        os.remove(os.path.join(WORK, f))

if not ONLY and len(ok) < 3:
    sys.exit(f"only {len(ok)} methods measured; the study needs the 4-bit row")
if ONLY and not ok:
    sys.exit("the requested methods produced nothing")
