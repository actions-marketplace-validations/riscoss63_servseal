"""Package the working tree as a Kaggle dataset and push the probe kernel.

The kernel needs a servseal that PyPI does not have yet -- API mode is unreleased --
so the source travels as a dataset rather than as a version number. Everything here is
private by default and matches the layout the other kernels in this account use.

    python push.py --dry-run     # build the payload, print what would be sent
    python push.py               # create/version the dataset, then push the kernel
    python push.py --status      # where the last run got to
    python push.py --fetch       # pull the log and artifacts into outputs/
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

PAYLOAD = ["servseal", "pyproject.toml", "README.md", "LICENSE"]


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


def push_kernel(a):
    meta = {
        "id": KERNEL,
        "title": "servseal-vllm-probe",
        "code_file": "vllm_probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    stage = os.path.join(HERE, "_kernel")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    shutil.copy2(os.path.join(HERE, "vllm_probe.py"), stage)
    json.dump(meta, open(os.path.join(stage, "kernel-metadata.json"), "w"), indent=2)
    print(f"  pushing {KERNEL} (private, GPU, internet)")
    print(a.kernels_push(stage))


def status(a):
    s = a.kernels_status(KERNEL)
    print(json.dumps(s if isinstance(s, dict) else s.__dict__, default=str, indent=2))


def fetch(a):
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "kaggle_vllm_probe")
    os.makedirs(dest, exist_ok=True)
    a.kernels_output(KERNEL, path=dest, force=True, quiet=False)
    for f in sorted(os.listdir(dest)):
        print("  " + f)
    log = os.path.join(dest, "servseal-vllm-probe.log")
    if os.path.exists(log):
        try:
            entries = json.load(open(log, encoding="utf-8"))
            text = "\n".join(e.get("data", "") for e in entries)
            open(os.path.join(OUT, "vllm_probe_output.txt"), "w",
                 encoding="utf-8").write(text)
            print(f"\n  log flattened to outputs/vllm_probe_output.txt "
                  f"({len(text)} chars)")
        except Exception as e:
            print(f"  could not flatten the log: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--kernel-only", action="store_true")
    a_ = p.parse_args()

    if a_.status:
        return status(api())
    if a_.fetch:
        return fetch(api())

    print("building payload")
    files = build()
    if a_.dry_run:
        for f in sorted(files)[:40]:
            print("   ", f)
        print(f"\nwould push dataset {DATASET} and kernel {KERNEL} (both private)")
        return 0
    a = api()
    if not a_.kernel_only:
        push_dataset(a, files)
        print("  waiting for the dataset to finish processing")
        time.sleep(20)
    push_kernel(a)
    print(f"\nwatch: https://www.kaggle.com/code/{KERNEL}")
    print("then:  python push.py --status   /   python push.py --fetch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
