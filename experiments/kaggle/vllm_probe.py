"""What a real inference server actually puts on the wire, and whether API mode works.

Everything measured for API mode so far ran against a server written alongside the
client, which validates the logic and proves nothing about interoperability. The open
question is narrow and decisive: when a production server answers
`/v1/completions` with `logprobs=1`, what is in `logprobs.tokens`? The raw token
(recovery is a dictionary lookup, measured collision-free), the decoded text (recovery
costs 0.01 % of the mass on English), or token ids? `servseal.endpoint` detects it
rather than assuming, and this run is where the detection meets something it did not
write.

Three deployments, all served by vLLM, all reached through the shipped CLI:

  gpt2, float32 ................. the reference itself. Must come back SEALED, or the
                                  chain has a defect somewhere between the seal and
                                  the socket.
  gpt2 + top_p 0.95 default ..... the provider switched on a nucleus filter. The
                                  client never sends top_p, so the server's own
                                  default is what is under test. Must come back
                                  CHANGED with the serving-layer signature.
  distilgpt2 .................... a smaller sibling on the same tokeniser, silently
                                  substituted. Must come back CHANGED, head-level.

The reference is snapshotted in float32 and vLLM is told to serve float32, because a
float16 deployment genuinely differs from a float32 reference (bfloat16 alone measures
0.036 Hellinger on GPT-2) and would make the first row fail for a real reason that
has nothing to do with the transport. Whether CPU-computed and GPU-computed float32
agree inside the acceptance band is itself unknown, and the first row measures it.

Kaggle: script kernel, GPU T4, internet on, with the servseal source as a dataset.
"""
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import re
import urllib.request
import zipfile

# Hugging Face's Xet backend 404s on this image
# (/api/models/<id>/xet-read-token/<sha>), which kills any download that resolves
# through it. Set before huggingface_hub is imported anywhere, including by vLLM.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WORK = "/kaggle/working"
KIT = "/kaggle/input/servseal-api-kit"
POSITIONS = "1500"
BUDGET = "5000"
PORT = 8000
BOOT_TIMEOUT = 900


def sh(cmd, **kw):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, **kw)


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


# ------------------------------------------------------------------ environment

banner("environment")
CAN_SERVE = True
try:
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        CAN_SERVE, WHY = False, "no GPU allocated"
    else:
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})",
              flush=True)
        if cap[0] < 7:
            CAN_SERVE = False
            WHY = (f"sm_{cap[0]}{cap[1]} (a P100); vLLM needs sm_70+. Settings -> "
                   "Accelerator -> GPU T4 x2, then Save & Run All FROM THE EDITOR. "
                   "ApiSaveKernelRequest carries only enable_gpu/enable_tpu, no "
                   "accelerator type, so every API push silently resets this to "
                   "Kaggle's default GPU -- measured, not assumed: a T4 set in the "
                   "UI came back as a P100 on the next pushed version.")
except ImportError:
    sys.exit("torch missing from the image")

if not CAN_SERVE:
    print(f"\nLIVE BATTERY WILL BE SKIPPED: {WHY}", flush=True)
    print("The static wire-format report below does not need a GPU and still "
          "answers\nthe question this run exists for.", flush=True)

# ------------------------------------------------------- what vLLM puts on the wire

banner("wire format, read from vLLM's own source (no GPU, no dataset needed)")
# Installing vLLM works on any machine; only *serving* needs sm_70+. So the question
# that gates the whole transport -- what lands in logprobs.tokens -- is answerable
# here whatever accelerator this kernel drew, by reading the code that writes it.
# The file is read rather than imported, because importing vllm initialises CUDA.
# This block depends on neither the dataset nor servseal, and runs before both.


def _vllm_root():
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    if not (spec and spec.origin):
        return None, "?"
    root = os.path.dirname(spec.origin)
    ver, vf = "?", os.path.join(root, "version.py")
    if os.path.exists(vf):
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)',
                      open(vf, encoding="utf-8", errors="replace").read())
        ver = m.group(1) if m else "?"
    return root, ver


if sh([sys.executable, "-m", "pip", "install", "-q", "vllm"]).returncode != 0:
    print("vLLM failed to install; the wire format cannot be read here.", flush=True)
    vroot = None
else:
    vroot, ver = _vllm_root()
    print(f"vllm {ver} at {vroot}", flush=True)

# Scanned recursively over the whole package, not just entrypoints/openai: vLLM moves
# this code between releases, and a narrow glob reports "nothing found" for a file
# that simply lives elsewhere -- which is a wrong answer wearing the clothes of a
# negative result.
PATTERNS = ["def format_token_id_placeholder", "def _get_decoded_token",
            "decoded_token =", "decoded_token=", "return_tokens_as_token_ids",
            "class CompletionLogProbs"]
# Ranked, not alphabetical. The first pass spent its display budget on
# benchmarks/plot.py while the two files that answer the question got a header line,
# because "sorted()" is not a relevance order.
RANK = ["entrypoints/openai/completion", "logprobs.py", "entrypoints/generate/base",
        "detoken", "entrypoints/openai"]


def score(rel):
    rel = rel.replace(os.sep, "/")
    for i, k in enumerate(RANK):
        if k in rel:
            return i
    return len(RANK)


hits, shown = 0, 0
if vroot:
    files = glob.glob(os.path.join(vroot, "**", "*.py"), recursive=True)
    files.sort(key=lambda p: (score(os.path.relpath(p, vroot)),
                              os.path.relpath(p, vroot)))
    print(f"scanning {len(files)} files under the installed package, "
          f"most relevant first", flush=True)
    for path in files:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        found = [p for p in PATTERNS if p in text]
        if not found:
            continue
        hits += 1
        rel = os.path.relpath(path, vroot)
        print(f"\n--- {rel}   {found}", flush=True)
        if shown >= 5:                      # enough to read; the list above is the map
            continue
        shown += 1
        lines = text.splitlines()
        printed = 0
        for i, ln in enumerate(lines):
            if not any(p in ln for p in PATTERNS):
                continue
            if printed >= 6:
                print("       (more in this file)", flush=True)
                break
            printed += 1
            lo, hi = max(0, i - 3), min(len(lines), i + 16)
            print("\n".join(f"{n + 1:5d}  {lines[n]}" for n in range(lo, hi)),
                  flush=True)
            print("       ...", flush=True)
if not hits:
    print("nothing matched anywhere in the package; the convention can then only "
          "come from the live probe below", flush=True)


banner("inputs")
# Report what is actually mounted rather than asserting where it should be: the mount
# name follows the dataset slug, and a source that failed to attach looks identical to
# one attached under another name until you look.
root = "/kaggle/input"
# Walked, not guessed. A dataset attached from the editor does not land at
# /kaggle/input/<slug>: it arrives nested under /kaggle/input/datasets/<owner>/<slug>,
# and a one-level check reports "not mounted" for a kit that is plainly there.
for dirpath, dirnames, filenames in os.walk(root):
    depth = dirpath[len(root):].count(os.sep)
    if depth <= 3:
        print(f"  {dirpath}: {sorted(dirnames)[:8]} {sorted(filenames)[:8]}",
              flush=True)
    if depth >= 5:
        dirnames[:] = []


def find_kit(base):
    """Any directory holding the package, or the archive the uploader made of it."""
    for dirpath, dirnames, filenames in os.walk(base):
        if "servseal.zip" in filenames:
            return dirpath
        if "servseal" in dirnames and os.path.exists(
                os.path.join(dirpath, "servseal", "__init__.py")):
            return dirpath
        if dirpath[len(base):].count(os.sep) >= 6:
            dirnames[:] = []
    return None


if not os.path.isdir(KIT) or not (
        os.path.isdir(os.path.join(KIT, "servseal"))
        or os.path.exists(os.path.join(KIT, "servseal.zip"))):
    found = find_kit(root) if os.path.isdir(root) else None
    if not found:
        sys.exit(f"the servseal source is nowhere under {root}; attach the dataset "
                 "abderrahmanesghairi/servseal-api-kit from the editor "
                 "(File -> Add input -> Datasets -> Your datasets)")
    KIT = found
print(f"kit: {KIT}", flush=True)

banner("install")
src = os.path.join(WORK, "servseal-src")
shutil.rmtree(src, ignore_errors=True)
shutil.copytree(KIT, src)
# the uploader stores directories as archives, and Kaggle sometimes expands them
# again on the way in, so accept whichever form actually arrived
pkg, zipped = os.path.join(src, "servseal"), os.path.join(src, "servseal.zip")
if not os.path.isdir(pkg):
    if not os.path.exists(zipped):
        sys.exit(f"neither servseal/ nor servseal.zip in the dataset: "
                 f"{sorted(os.listdir(src))}")
    with zipfile.ZipFile(zipped) as z:
        z.extractall(pkg)
    # an archive made from the directory itself nests one level deeper
    inner = os.path.join(pkg, "servseal")
    if os.path.isdir(inner) and os.path.exists(os.path.join(inner, "__init__.py")):
        for f in os.listdir(inner):
            shutil.move(os.path.join(inner, f), os.path.join(pkg, f))
        os.rmdir(inner)
if not os.path.exists(os.path.join(pkg, "__init__.py")):
    sys.exit(f"package looks wrong: {sorted(os.listdir(pkg))[:20]}")
print(f"package: {len(os.listdir(pkg))} files", flush=True)
sh([sys.executable, "-m", "pip", "install", "-q", "sqsketch>=0.3", "transformers"],
   check=True)
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", src], check=True)
import servseal                                                    # noqa: E402
print(f"servseal {servseal.__version__}", flush=True)


# ---------------------------------------------------------------------- serving

def free_port(p):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", p)) != 0


_UNRECOGNISED = re.compile(r"unrecognized arguments:\s*(.+)")
_CORE = {"--model", "--port"}


def _drop(args, message):
    """Remove the flags argparse just rejected, with their values. None if no news.

    vLLM renames server flags between releases -- this image dropped
    `--disable-log-requests` for `--enable-log-requests` -- and each rename used to
    cost a whole run to discover. The launcher now reads the rejection and retries
    without the offending flags, so only a flag that actually matters can stop it.
    """
    m = _UNRECOGNISED.search(message)
    if not m:
        return None
    bad = {t for t in m.group(1).split() if t.startswith("--")}
    if not bad or bad & _CORE:
        return None
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in bad:
            nxt = args[i + 1] if i + 1 < len(args) else None
            skip = bool(nxt) and not nxt.startswith("--")
            continue
        out.append(a)
    return (out, sorted(bad)) if out != args else None


def _launch(model, extra, log):
    logf = open(os.path.join(WORK, log), "w")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(PORT), "--dtype", "float32",
           "--max-model-len", "256", "--gpu-memory-utilization", "0.55",
           *(extra or [])]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT:
        if proc.poll() is not None:
            logf.close()
            tail = open(os.path.join(WORK, log)).read()[-4000:]
            raise RuntimeError(f"vLLM exited with {proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as r:
                json.load(r)
                print(f"vLLM up in {time.time() - t0:.0f}s", flush=True)
                return proc, logf
        except Exception:
            time.sleep(3)
    proc.kill()
    logf.close()
    raise RuntimeError("vLLM did not become ready")


def start(model, extra=None, log="vllm.log"):
    """Launch vLLM's OpenAI server and wait until it answers /v1/models.

    Retries while argparse keeps naming flags it does not know, then once more with
    no optional flags at all. A flag this image has renamed must not cost the cases
    that do not depend on it -- and if `--generation-config` is what goes, the model
    directory may still be read on its own, which the verdict shows either way.
    """
    args = list(extra or [])
    for _ in range(3):
        try:
            return _launch(model, args, log)
        except RuntimeError as e:
            dropped = _drop(args, str(e))
            if not dropped:
                break
            args, bad = dropped
            print(f"vLLM rejected {bad}; retrying without them", flush=True)
    if args:
        print("retrying with no optional flags at all", flush=True)
        return _launch(model, [], log)
    return _launch(model, args, log)


def stop(proc, logf):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()
    t0 = time.time()
    while not free_port(PORT) and time.time() - t0 < 60:
        time.sleep(1)


def raw_wire(model):
    """The one thing this run exists to find out: what comes back, literally."""
    body = json.dumps({"model": model, "prompt": [[464, 3139, 286, 4881]],
                       "max_tokens": 1, "temperature": 1.0, "logprobs": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# -------------------------------------------------------------------- reference

if not CAN_SERVE:
    banner("STOPPING BEFORE THE LIVE BATTERY")
    print(f"reason: {WHY}", flush=True)
    print("", flush=True)
    print("The wire format above came from vLLM's source and is the finding this "
          "run was\npushed for. What stays unmeasured is interoperability end to "
          "end: that a sealed\nreference verifies against a vLLM socket, and that a "
          "provider-side top-p default\nand a substituted sibling come back CHANGED "
          "with the right signature.\nRe-run this kernel on a T4 for those three "
          "rows.", flush=True)
    sys.exit(0)

banner("reference snapshot (float32, with API bands)")
ref = os.path.join(WORK, "gpt2.seal.npz")
t0 = time.time()
r = sh([sys.executable, "-m", "servseal.cli", "snapshot", "gpt2", "-o", ref,
        "--positions", POSITIONS, "--api-budget", BUDGET],
       capture_output=True, text=True)
print(r.stdout + r.stderr, flush=True)
if r.returncode != 0:
    sys.exit("snapshot failed")
print(f"[{time.time() - t0:.0f}s]", flush=True)

def build_topp_variant():
    """A copy of gpt2 whose own generation_config turns on nucleus sampling.

    The provider default the client must never override, and deliberately not a
    client-side argument. Built lazily inside the battery: doing it up front meant a
    download failure took the two cases with it that need no download at all.

    The snapshot step already pulled gpt2, so the cache is tried first and the network
    is only a fallback.
    """
    from huggingface_hub import snapshot_download

    # Only what a server loads. Unrestricted, the gpt2 repo brings its onnx, tf,
    # flax, rust and tflite exports too -- 2.7 GB per copy, which landed in the
    # kernel's output archive and made it 5 GB to download.
    keep = ["config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    try:
        base = snapshot_download("gpt2", allow_patterns=keep, local_files_only=True)
        print("gpt2 taken from the local cache", flush=True)
    except Exception as e:
        print(f"cache miss ({type(e).__name__}), downloading", flush=True)
        base = snapshot_download("gpt2", allow_patterns=keep)
    tweaked = os.path.join(WORK, "gpt2-topp")
    shutil.rmtree(tweaked, ignore_errors=True)
    shutil.copytree(base, tweaked, symlinks=False)
    gc = os.path.join(tweaked, "generation_config.json")
    cfg = json.load(open(gc)) if os.path.exists(gc) else {}
    cfg.update({"do_sample": True, "top_p": 0.95, "temperature": 1.0})
    json.dump(cfg, open(gc, "w"), indent=2)
    print(f"generation_config.json -> {cfg}", flush=True)
    return tweaked


# ------------------------------------------------------------------------- runs

CASES = [
    ("gpt2 float32 (unchanged)", lambda: "gpt2", [], "sealed", 0),
    ("gpt2 + provider top-p 0.95", build_topp_variant,
     ["--generation-config", "auto"], "changed", 3),
    ("distilgpt2 substituted", lambda: "distilgpt2", [], "changed", 3),
]

rows = []
for name, resolve, extra, want_status, want_code in CASES:
    banner(name)
    try:
        model = resolve()
    except Exception as e:
        print(f"SKIP: could not prepare the model: {type(e).__name__}: {e}",
              flush=True)
        rows.append((name, "-", "-", f"model unavailable ({type(e).__name__})",
                     None, False))
        continue
    try:
        proc, logf = start(model, extra)
    except Exception as e:
        # the tail, not the head: argparse prints its verdict on the LAST line, and
        # slicing from the front showed 600 characters of usage text instead
        print(f"SKIP: ...{str(e)[-800:]}", flush=True)
        rows.append((name, "-", "-", "server failed", None, False))
        continue
    try:
        wire = raw_wire(model)
        ch = wire["choices"][0]
        lp = ch.get("logprobs") or {}
        print("RAW WIRE  text=" + repr(ch.get("text")), flush=True)
        print("RAW WIRE  logprobs.tokens=" + repr(lp.get("tokens")), flush=True)
        print("RAW WIRE  keys=" + repr(sorted(lp)), flush=True)

        slug = ("vllm_sealed" if want_status == "sealed"
                else "vllm_topp" if "top-p" in name else "vllm_substituted")
        rep = os.path.join(WORK, f"attestation_{slug}.html")
        js = os.path.join(WORK, f"attestation_{slug}.json")
        r = sh([sys.executable, "-m", "servseal.cli", "verify", ref,
                "--endpoint", f"http://127.0.0.1:{PORT}/v1",
                "--served-model", model, "--budget", BUDGET,
                "--tokenizer", "gpt2", "--batch", "32", "--concurrency", "8",
                "--report", rep, "--json", js], capture_output=True, text=True)
        out = r.stdout + r.stderr
        print(out, flush=True)
        got = {}
        for line in out.splitlines():
            k, _, v = line.partition(" ")
            got.setdefault(k, v.strip())
        conv = "?"
        if os.path.exists(js):
            conv = json.load(open(js))["wire"]["convention"]
        hit = r.returncode == want_code
        rows.append((name, (got.get("s1", "?").split() or ["?"])[0],
                     (got.get("s2", "?").split() or ["?"])[0],
                     got.get("signature", "?"), conv, hit))
    except Exception as e:
        print(f"CASE FAILED: {type(e).__name__}: {str(e)[:600]}", flush=True)
        rows.append((name, "-", "-", f"{type(e).__name__}", None, False))
    finally:
        stop(proc, logf)


# ---------------------------------------------------------------------- verdict

banner(f"vLLM INTEROPERABILITY  (gpt2, {BUDGET} sampled tokens, {POSITIONS} positions)")
print(f"{'deployment':30s} {'S1':>8s} {'S2':>8s} {'wire':>9s} {'':5s}signature")
for name, s1, s2, sig, conv, hit in rows:
    print(f"{name:30s} {s1:>8s} {s2:>8s} {str(conv):>9s} "
          f"{'OK' if hit else 'MISS':>5s} {sig}")

# Neither the copied model nor the installed source rides home: Kaggle archives
# everything left in /kaggle/working, and the first run of this shipped a 5 GB
# download for three attestations of nine kilobytes each.
shutil.rmtree(os.path.join(WORK, "gpt2-topp"), ignore_errors=True)
shutil.rmtree(os.path.join(WORK, "servseal-src"), ignore_errors=True)

print("\nartifacts in /kaggle/working:", flush=True)
for f in sorted(glob.glob(os.path.join(WORK, "attestation_*"))):
    print("  " + os.path.basename(f), flush=True)

bad = [n for n, *_, h in rows if not h]
if bad:
    print(f"\n  FAIL  {len(bad)} case(s) did not match ground truth: {bad}")
    print("  The transport is not proven against vLLM. Read the RAW WIRE lines "
          "above\n  first: a convention this client does not know is the likeliest "
          "cause.")
    sys.exit(1)
print("\n  PASS  the shipped CLI verifies a real vLLM endpoint: sealed when it serves\n"
      "        the reference, and caught when the provider filters the tail or swaps\n"
      "        the model, with no logprobs of the reference available to it.")
