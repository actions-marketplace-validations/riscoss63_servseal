"""The API-mode path end to end, on a real model, through the shipped CLI.

`e2e_real_models.py` proves weights mode against ground truth. This does the same for
the mode that talks to a black box: it stands up an OpenAI-compatible endpoint backed
by GPT-2's real next-token distributions over the real probe set, seals a reference
with API bands, and runs `servseal verify --endpoint` against it -- unchanged, then
with a top-p 0.95 filter switched on at serve time.

The point is the second row. A nucleus filter leaves the most likely token unchanged at
every position and perplexity unchanged to two decimals; against a sampled endpoint
there are no logprobs to inspect and no distribution to diff. If exit 3 comes back, the
whole chain works: prefixes rebuilt to match the seal, tokens recovered from the wire,
statistics computed against sketches made months earlier, and a verdict read off bands
calibrated where the reference distributions still existed.

    python e2e_endpoint.py            # needs experiments/cache from e2e_real_models.py

Asserts every verdict against its ground truth and exits non-zero on any miss.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servseal.probes import load_probes                             # noqa: E402
from servseal.runner import api_metadata                            # noqa: E402
from servseal.snapshot import Snapshot                              # noqa: E402
from servseal.wire import position_prefixes                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "outputs")
BUDGET = 5000
MODEL = "gpt2"


def top_p(P, keep=0.95):
    srt = -np.sort(-P, 1)
    thr = srt[np.arange(len(P)), np.argmax(np.cumsum(srt, 1) >= keep, 1)][:, None]
    Q = np.where(P >= thr, P, 0.0)
    return (Q / Q.sum(1, keepdims=True)).astype(np.float32)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        cfg = self.server.cfg
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "top_p" not in body, "the client must never send top_p"
        prompts, n = body["prompt"], int(body.get("n", 1))
        choices = []
        with cfg["lock"]:
            for pi, prompt in enumerate(prompts):
                row = cfg["cum"][cfg["index"][tuple(prompt)]]
                u = cfg["rng"].random(n)
                ids = np.minimum(np.searchsorted(row, u, side="right"),
                                 len(row) - 1)
                for ci, t in enumerate(ids):
                    choices.append({
                        "index": pi * n + ci,
                        "text": cfg["tok"].decode([int(t)]),
                        "logprobs": {"tokens": [cfg["pieces"][int(t)]],
                                     "token_logprobs": [-1.0]}})
        out = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def serve(P, prefixes, tok, pieces):
    cum = np.cumsum(P.astype(np.float64), axis=1)
    cum /= cum[:, -1:]
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.cfg = {"cum": cum, "tok": tok, "pieces": pieces, "lock": threading.Lock(),
               "rng": np.random.default_rng(11),
               "index": {tuple(p): i for i, p in enumerate(prefixes)}}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def cli(*args):
    """The shipped command, in a real subprocess -- exit codes are the contract.

    PYTHONPATH carries the repository root because this runs from experiments/, where
    an uninstalled checkout would not find the package; an installed one is unaffected.
    """
    root = os.path.dirname(HERE)
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-m", "servseal.cli", *args],
                       capture_output=True, text=True, timeout=1800, env=env)
    return r.returncode, r.stdout + r.stderr


def main():
    from transformers import AutoTokenizer

    pb = os.path.join(CACHE, "gpt2_P_base.npy")
    sb = os.path.join(CACHE, "gpt2_ref.seal.npz")
    if not (os.path.exists(pb) and os.path.exists(sb)):
        print("cache missing -- run e2e_real_models.py first")
        return 1
    os.makedirs(OUT, exist_ok=True)

    P = np.asarray(np.load(pb, mmap_mode="r"))
    snap = Snapshot.load(sb)
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = load_probes(snap.meta.get("probe_file"))

    prefixes = position_prefixes(tok, texts, max_positions=snap.meta["n_positions"],
                                 max_length=snap.meta.get("max_length", 96),
                                 template=snap.meta.get("template"))
    assert len(prefixes) == snap.meta["n_positions"] == P.shape[0], (
        f"rebuilt {len(prefixes)} positions, the seal has {snap.meta['n_positions']}")
    print(f"prefixes   {len(prefixes)} rebuilt from the probe set, matching the seal")

    t0 = time.time()
    snap.meta["model_id"] = MODEL
    snap.meta["api"] = api_metadata(P, snap, tok, budgets=(BUDGET,))
    ref = os.path.join(OUT, "gpt2_api_ref.seal.npz")
    snap.save(ref)
    tc = snap.meta["api"]["text_convention"]
    b = snap.meta["api"]["bands"][str(BUDGET)]
    print(f"calibrated {time.time() - t0:.0f}s   corrupted mass "
          f"{100 * tc['wrong_mass']:.4f} %   "
          f"s1 band {b['exact']['s1'][0]:.4f}-{b['exact']['s1'][1]:.4f}")

    pieces = tok.convert_ids_to_tokens(list(range(snap.meta["vocab_size"])))
    cases = [("unchanged", P, 0, "sealed"),
             ("serve: top-p 0.95", top_p(P), 3, "changed")]

    rows, ok = [], True
    for name, Q, want_code, want_status in cases:
        # named for what they attest, and marked local: the committed vllm_* files
        # next to these came from a real server, and the two must not be confused
        slug = "sealed" if want_status == "sealed" else "topp"
        rep = os.path.join(OUT, f"local_e2e_{slug}.html")
        js = os.path.join(OUT, f"local_e2e_{slug}.json")
        srv, url = serve(Q, prefixes, tok, pieces)
        try:
            code, out = cli("verify", ref, "--endpoint", url, "--served-model", MODEL,
                            "--budget", str(BUDGET), "--batch", "32",
                            "--concurrency", "8", "--report", rep, "--json", js)
        finally:
            srv.shutdown()
        # the attestation is the deliverable, so its production is part of the battery
        html = pathlib.Path(rep).read_text(encoding="utf-8")
        rec = json.loads(pathlib.Path(js).read_text(encoding="utf-8"))
        assert html.startswith("<!doctype html>") and "sampled endpoint" in html
        assert rec["mode"] == "endpoint" and rec["verdict"]["status"] == want_status
        assert "127.0.0.1" in html and str(srv.server_address[1]) in html
        assert "mean Hellinger" not in html, "weights-mode page rendered for a sample"
        got = {}
        for line in out.splitlines():
            k, _, v = line.partition(" ")
            got[k] = v.strip()
        num = {k: (got.get(k, "?").split() or ["?"])[0] for k in ("s1", "s2")}
        hit = code == want_code and got.get("verdict", "").lower().startswith(
            want_status)
        ok = ok and hit
        rows.append((name, num["s1"], num["s2"], want_status,
                     got.get("signature", ""), code, hit))
        print(f"\n--- {name} ---\n{out.rstrip()}")

    print(f"\n{'=' * 78}\nAPI MODE, END TO END (gpt2, {BUDGET} sampled tokens, "
          f"{len(prefixes)} positions)\n{'=' * 78}")
    print(f"{'deployment event':20s} {'S1':>8s} {'S2':>8s} {'exit':>5s} {'':4s}"
          f"signature")
    for name, s1, s2, verdict, sig, code, hit in rows:
        print(f"{name:20s} {s1:>8s} {s2:>8s} {code:>5d} {'OK' if hit else 'MISS':>4s} "
              f"{sig}")

    print()
    if not ok:
        print("  FAIL  a verdict did not match its ground truth")
        return 1
    print("  PASS  the sampled endpoint is sealed when unchanged and caught when a\n"
          "        serving-layer filter is switched on -- with no logprobs inspected,\n"
          "        no distribution available, and the reference months old.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
