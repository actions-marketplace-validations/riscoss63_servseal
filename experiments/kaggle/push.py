"""Push and collect the two Kaggle kernels this repository runs on a GPU.

    --which vllm   the API-mode interoperability probe. It predates the release, so
                   servseal travels to it as a private dataset.
    --which quant  the real-toolchain quantisation study. Takes servseal from PyPI
                   and needs no dataset at all.

    python push.py --which quant --dry-run
    python push.py --which quant             # push and run
    python push.py --which quant --status
    python push.py --which quant --fetch     # log and artifacts into outputs/

The accelerator type is not in the API: ApiSaveKernelRequest carries enable_gpu and
enable_tpu only, so every push here reverts the kernel to Kaggle's default GPU (a
P100). Anything needing a T4 must be started from the editor after the push.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(HERE, "_build")
OUT = os.path.join(os.path.dirname(HERE), "outputs")

USER = "abderrahmanesghairi"
DATASET = f"{USER}/servseal-api-kit"
KERNEL = f"{USER}/servseal-vllm-probe"
KERNEL_QUANT = f"{USER}/servseal-quant-real"
KERNEL_EXL3 = f"{USER}/servseal-exl3-curve"

PAYLOAD = ["servseal", "pyproject.toml", "README.md", "LICENSE"]

# which kernel each --which selects, and the script it carries
TARGET = {"vllm": (KERNEL, "vllm_probe.py"),
          "quant": (KERNEL_QUANT, "quant_real.py"),
          "exl3": (KERNEL_EXL3, "exl3_curve.py")}


def api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def build():
    """Copy just the package and its metadata; no caches, no experiment outputs."""
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    for item in PAYLOAD:
        s = os.path.join(REPO, item)
        if not os.path.exists(s):
            print(f"  skip (absent): {item}")
            continue
        d = os.path.join(BUILD, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))
        else:
            shutil.copy2(s, d)
    json.dump({"title": "servseal-api-kit", "id": DATASET,
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(BUILD, "dataset-metadata.json"), "w"), indent=2)
    files = [os.path.relpath(os.path.join(r, f), BUILD)
             for r, _, fs in os.walk(BUILD) for f in fs]
    size = sum(os.path.getsize(os.path.join(BUILD, f)) for f in files)
    print(f"  payload: {len(files)} files, {size / 1024:.0f} KB")
    return files


def push_dataset(a, files):
    try:
        a.dataset_status(DATASET)
        exists = True
    except Exception:
        exists = False
    if exists:
        print(f"  versioning {DATASET}")
        a.dataset_create_version(BUILD, version_notes=f"api mode {time.strftime('%F %T')}",
                                 dir_mode="zip", quiet=False)
    else:
        print(f"  creating {DATASET}")
        a.dataset_create_new(BUILD, dir_mode="zip", public=False, quiet=False)


def push_kernel(a, which="vllm"):
    meta = {
        "id": TARGET[which][0],
        "title": TARGET[which][0].split("/")[1],
        "code_file": TARGET[which][1],
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        # the quantisation kernel takes servseal from PyPI, so it needs no dataset
        # only the vllm probe predates the release and needs the source shipped
        "dataset_sources": [DATASET] if which == "vllm" else [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    stage = os.path.join(HERE, "_kernel")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    shutil.copy2(os.path.join(HERE, meta["code_file"]), stage)
    json.dump(meta, open(os.path.join(stage, "kernel-metadata.json"), "w"), indent=2)
    print(f"  pushing {meta['id']} (private, GPU, internet"
          + (f", dataset {DATASET})" if which == "vllm" else ", no dataset)"))
    print(a.kernels_push(stage))


def status(a, which="vllm"):
    s = a.kernels_status(TARGET[which][0])
    print(json.dumps(s if isinstance(s, dict) else s.__dict__, default=str, indent=2))


def fetch(a, which="vllm"):
    os.makedirs(OUT, exist_ok=True)
    k = TARGET[which][0]
    dest = os.path.join(OUT, "kaggle_" + k.split("/")[1])
    # Archive whatever is there before replacing it. kernels_output only ever
    # returns the *latest* version, so an overwritten result cannot be fetched
    # again -- and a run whose evidence is gone cannot be cited.
    if os.path.isdir(dest) and os.listdir(dest):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        keep = f"{dest}.{stamp}"
        shutil.move(dest, keep)
        print(f"  previous output archived to {os.path.basename(keep)}")
    os.makedirs(dest, exist_ok=True)
    a.kernels_output(k, path=dest, force=True, quiet=False)
    for f in sorted(os.listdir(dest)):
        print("  " + f)
    log = os.path.join(dest, k.split("/")[1] + ".log")
    if os.path.exists(log):
        try:
            entries = json.load(open(log, encoding="utf-8"))
            text = "\n".join(e.get("data", "") for e in entries)
            flat = os.path.join(OUT, k.split("/")[1].replace("servseal-", "")
                                .replace("-", "_") + "_output.txt")
            open(flat, "w", encoding="utf-8").write(text)
            print(f"\n  log flattened to outputs/{os.path.basename(flat)} "
                  f"({len(text)} chars)")
        except Exception as e:
            print(f"  could not flatten the log: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--kernel-only", action="store_true")
    p.add_argument("--which", default="vllm", choices=list(TARGET),
                   help="which kernel to push, watch or fetch")
    a_ = p.parse_args()

    if a_.status:
        return status(api(), a_.which)
    if a_.fetch:
        return fetch(api(), a_.which)

    print("building payload")
    files = build()
    if a_.dry_run:
        for f in sorted(files)[:40]:
            print("   ", f)
        tgt = TARGET[a_.which][0]
        print(f"\nwould push kernel {tgt} ({TARGET[a_.which][1]}, private)"
              + (f", and dataset {DATASET}" if a_.which == "vllm" else ""))
        return 0
    a = api()
    if not a_.kernel_only and a_.which == "vllm":
        push_dataset(a, files)
        print("  waiting for the dataset to finish processing")
        time.sleep(20)
    push_kernel(a, a_.which)
    tgt = TARGET[a_.which][0]
    w = "" if a_.which == "vllm" else f" --which {a_.which}"
    print(f"\nwatch: https://www.kaggle.com/code/{tgt}")
    print(f"then:  python push.py{w} --status   /   python push.py{w} --fetch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
