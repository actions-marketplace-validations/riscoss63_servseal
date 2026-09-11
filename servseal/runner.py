"""Running a live model over the probes. The only module that imports torch.

The measured object is the full softmax at every probe position, float32, greedy-free:
no sampling is involved in weights-mode snapshots, so on CPU the snapshot of a given
model is bit-deterministic -- an unchanged deployment attests at Hellinger distance
exactly zero, and anything above the threshold is signal, not luck.

Perplexity over the probes is recorded into the metadata not because it is a good
change detector -- the point of this tool is that it is not (a top-p 0.95 filter moves
the distribution by 0.149 while moving perplexity by 0.00) -- but because showing the
two numbers side by side in the report is the honest way to make that argument.
"""
from __future__ import annotations

import time

import numpy as np

from .probes import load_probes, probe_id
from .snapshot import Snapshot

__all__ = ["snapshot_model", "softmax_over_probes", "api_metadata"]

_DTYPES = {"float32": None, "bfloat16": "bfloat16", "float16": "float16"}


def softmax_over_probes(model, tokenizer, texts, *, max_positions=1500,
                        max_length=96, template=None):
    """Full next-token distributions (positions x vocab, float32) plus realised-token
    log-probabilities for perplexity. One forward pass per probe text."""
    import torch

    # The probes must reach the model wherever it lives. Snapshotting on CPU is the
    # recommendation and was the only path exercised, so a model on a GPU used to
    # fail here on the first embedding lookup -- `device=` has been a parameter of
    # snapshot_model since the start and did not work for anything but "cpu".
    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device

    rows, logprobs = [], []
    for t in texts:
        if sum(r.shape[0] for r in rows) >= max_positions:
            break
        text = template.replace("{text}", t) if template else t
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.to(device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        p = torch.softmax(logits, -1).cpu().numpy().astype(np.float32)
        rows.append(p)
        tg = ids[0, 1:].cpu().numpy()
        n = min(len(tg), p.shape[0] - 1)
        logprobs.append(np.log(np.clip(p[np.arange(n), tg[:n]], 1e-300, None)))
    P = np.concatenate(rows)[:max_positions]
    ppl = float(np.exp(-np.mean(np.concatenate(logprobs)))) if logprobs else float("nan")
    return P, ppl


def api_metadata(P, snap, tokenizer, *, budgets=(5000,), reps=200, alpha=0.05,
                 seed=1234):
    """Everything API mode needs, computed while the reference distributions exist.

    This is the one moment it can be done. `calibrate_bands` needs P to simulate the
    unchanged endpoint, and `corrupted_mass` needs P to weight the text convention's
    ambiguity by the mass that will actually be sampled -- and P is gone the instant
    the snapshot is written. Verification against a black box later has the sketches
    and these numbers, and nothing else.

    Both band sets are stored because the endpoint's convention is not knowable now:
    `exact` for an endpoint that returns logprobs (measured collision-free, nothing to
    invert) and `text` for one that returns a bare completion, calibrated through the
    round trip so its systematic component is absorbed rather than assumed away.
    """
    from .sampler import calibrate_bands
    from .wire import WireMap, corrupted_mass

    wm = WireMap(tokenizer, P.shape[1])
    out = {"wiremap": wm.stats(), "text_convention": corrupted_mass(P, wm),
           "bands": {}}
    for b in budgets:
        out["bands"][str(int(b))] = {
            "exact": calibrate_bands(P, snap, total=int(b), reps=reps, alpha=alpha,
                                     seed=seed),
            "text": calibrate_bands(P, snap, total=int(b), reps=reps, alpha=alpha,
                                    seed=seed, wiremap=wm),
        }
    return out


def snapshot_model(model_id, *, probes=None, D=256, seed=0, max_positions=1500,
                   max_length=96, template=None, dtype="float32", device="cpu",
                   label=None, return_P=False, api_budgets=None):
    """Load a model, run the probes, return the Snapshot (and optionally the raw P)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
    texts = load_probes(probes)
    tok = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = getattr(torch, dtype) if _DTYPES[dtype] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    if dtype != "float32":
        # measure in float32 arithmetic what the narrowed weights produce, which is the
        # deployment-relevant object; keeping reduced-precision *arithmetic* as well
        # would measure the host's kernel quirks along with the model
        model = model.to(torch.float32)
    model = model.to(device).eval()

    t0 = time.perf_counter()
    P, ppl = softmax_over_probes(model, tok, texts, max_positions=max_positions,
                                 max_length=max_length, template=template)
    snap = Snapshot.from_distributions(
        P, probe=probe_id(texts), model=label or str(model_id), D=D, seed=seed,
        template=template, dtype=dtype,
        extra={"perplexity": round(ppl, 4), "probe_file": probes or "default-v1",
               "model_id": str(model_id),
               "max_length": int(max_length),
               "snapshot_seconds": round(time.perf_counter() - t0, 1)})
    if api_budgets:
        snap.meta["api"] = api_metadata(P, snap, tok, budgets=api_budgets)
    return (snap, P) if return_P else snap
