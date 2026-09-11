"""Can a sampled token be put back on its reference id? The gate on API mode.

Weights mode reads the full softmax and indexes it by token id. API mode cannot: the
endpoint returns *text*. `servseal.sampler.statistics` needs ids, so between the wire
and the statistic there is a step that does not exist yet -- recovering the reference
vocabulary's id from the string the endpoint sent back. Everything measured for API
mode so far (power 1.00 at 5000 samples) assumed that step is free. This run prices it.

Three ways it can go wrong, and only the third is dangerous:

  exact   decode(i) re-encodes to [i].                    The step is free.
  split   decode(i) re-encodes to several tokens, or none. Detectable: the sample is
          dropped, costing budget but never accuracy.
  wrong   decode(i) re-encodes to a single *different* id. Silent corruption: the
          statistic is computed against a token the endpoint never emitted.

The headline is not the share of the vocabulary that collides -- that number is large
and almost entirely irrelevant, because the colliding entries are byte fragments a
model in normal text never emits. What matters is mass: the share of the reference
distribution that sits on a token the round trip would move. And what matters more is
the end of the pipeline: how far (S1, S2) shift once the round trip is applied, next
to the width of the acceptance bands those statistics are judged against. A systematic
shift that is small against the bands is noise; one that is comparable to them turns
the false-positive rate from 5% into whatever it likes.

Battery 4 therefore also tests the fix. The round trip is deterministic, so calibrating
*through* it should absorb any systematic component entirely, leaving only whatever
variance and power loss it adds -- which calibration cannot repair and which is the
real question.

Two wire conventions are measured because the answer depends on the endpoint:
  piece    what `logprobs` usually carries (the raw token, markers and all)
  decoded  what a plain text field carries

    python token_recovery.py                 # all three cached models
    python token_recovery.py gpt2            # one of: gpt2 pythia qwen
    python token_recovery.py --no-power      # skip the top-p power check

Needs experiments/cache from e2e_real_models.py / e2e_qwen.py, and the tokenisers
(local HF cache is enough; no weights are loaded and nothing runs on a GPU).
Peak memory is one float64 array of the cached P's shape -- about 1.8 GB for qwen.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servseal.sampler import detect, sample_stream, statistics        # noqa: E402
from servseal.snapshot import Snapshot                                # noqa: E402
from servseal.wire import WireMap                                     # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
TOKENISERS = {"gpt2": "gpt2",
              "pythia": "EleutherAI/pythia-160m",
              "qwen": "Qwen/Qwen2.5-0.5B"}

BUDGET = 5000                      # the budget the README sells
REPS_CAL, REPS_NULL, REPS_ALT = 200, 100, 100
ALPHA = 0.05
TOP_P = 0.95
POS_CHUNK = 32

# A shift worth this much of the band half-width is treated as material: at that size
# it moves the false-positive rate enough to matter, and the run says so.
SHIFT_BUDGET = 0.25


# --------------------------------------------------------------------- wire forms

def wire_forms(tok, vocab):
    """(piece, decoded) for every id the model can emit.

    `vocab` is the model's output width, which can exceed the tokeniser (pythia pads
    50277 to 50304). Padded slots get None and are reported separately -- the model
    still puts mass on them, and no endpoint can ever return one.
    """
    n_tok = len(tok)
    ids = list(range(min(vocab, n_tok)))
    pieces = tok.convert_ids_to_tokens(ids)
    decoded = []
    for s in range(0, len(ids), 4096):
        decoded.extend(tok.batch_decode([[i] for i in ids[s:s + 4096]]))
    pad = vocab - len(ids)
    pieces += [None] * pad
    decoded += [None] * pad
    return pieces, decoded, pad


def show(s):
    """ASCII-safe repr: colliding wire forms are routinely unprintable by design."""
    return ascii(s)


def collisions(forms):
    """id groups sharing one wire form, ignoring the padded slots."""
    groups = defaultdict(list)
    for i, s in enumerate(forms):
        if s is not None:
            groups[s].append(i)
    return {s: ids for s, ids in groups.items() if len(ids) > 1}


def classify(tok, decoded, vocab):
    """The product's resolution rule, applied to every id.

    The table comes from `servseal.wire.WireMap` rather than a copy of it: this run is
    the price list for the deployed resolver, so measuring a re-implementation would
    let the two drift and the number stop meaning anything.

    The rule is tokeniser-only by necessity. At verify time the snapshot holds sketches
    and argmax ids, never the reference distributions, so 'resolve the collision in
    favour of whichever candidate the reference thinks likelier' is not available.

    Returns remap (id -> id, or -1 when unresolvable) and four disjoint index sets.
    """
    wm = WireMap(tok, vocab)
    arange = np.arange(vocab)
    padded = np.flatnonzero(np.array([d is None for d in decoded]))
    exact = np.flatnonzero(wm.remap == arange)
    wrong = np.flatnonzero((wm.remap >= 0) & (wm.remap != arange))
    split = np.setdiff1d(np.flatnonzero(wm.remap < 0), padded, assume_unique=True)
    assert exact.size + wrong.size + split.size + padded.size == vocab
    return wm.remap, exact, wrong, split, padded


# ------------------------------------------------------------------------- mass

def mass_on(P, idx_sets, argmax):
    """Share of the reference mass each index set carries, and the argmax exposure.

    Mass is what decides whether a collision is ever seen: a pair of ids that collide
    and each hold 1e-12 of the mass will not be sampled in this century. The argmax
    line is separate because S2 compares against `snapshot.argmax` directly, so an
    ambiguous argmax corrupts that coordinate at full strength.
    """
    n_pos = P.shape[0]
    out = {k: 0.0 for k in idx_sets}
    worst = {k: 0.0 for k in idx_sets}
    for s in range(0, n_pos, POS_CHUNK):
        blk = np.asarray(P[s:s + POS_CHUNK], dtype=np.float64)
        blk /= blk.sum(1, keepdims=True)
        for k, idx in idx_sets.items():
            if idx.size == 0:
                continue
            per_pos = blk[:, idx].sum(1)
            out[k] += float(per_pos.sum())
            worst[k] = max(worst[k], float(per_pos.max()))
    out = {k: v / n_pos for k, v in out.items()}
    hit = {k: int(np.isin(argmax, idx).sum()) for k, idx in idx_sets.items()}
    return out, worst, hit


# ---------------------------------------------------------------------- sampling

def build_cum(P, top_p=None):
    """Row-wise normalised CDF, chunked so P is never held dense.

    Identical arithmetic to `sample_stream`'s per-row cumsum, which is asserted below
    rather than assumed: the product's sampler is the reference, this is only faster.
    """
    n_pos, vocab = P.shape
    cum = np.empty((n_pos, vocab), dtype=np.float64)
    for s in range(0, n_pos, POS_CHUNK):
        blk = np.asarray(P[s:s + POS_CHUNK], dtype=np.float64)
        if top_p is not None:
            srt = -np.sort(-blk, 1)
            k = np.argmax(np.cumsum(srt, 1) >= top_p, 1)
            blk = np.where(blk >= srt[np.arange(len(blk)), k][:, None], blk, 0.0)
        c = np.cumsum(blk, axis=1)
        cum[s:s + POS_CHUNK] = c / c[:, -1:]
    return cum


def draw(cum, total, rng):
    """Bit-identical to servseal.sampler.sample_stream, with the CDF precomputed."""
    n_pos, vocab = cum.shape
    per = np.full(n_pos, total // n_pos, dtype=np.int64)
    per[: total - per.sum()] += 1
    pos_out, tok_out = [], []
    for i in range(n_pos):
        if per[i] == 0:
            continue
        ids = np.minimum(np.searchsorted(cum[i], rng.random(per[i]), side="right"),
                         vocab - 1)
        pos_out.append(np.full(per[i], i, dtype=np.int64))
        tok_out.append(ids.astype(np.int64))
    return np.concatenate(pos_out), np.concatenate(tok_out)


def through_wire(pos, tok, remap):
    """What the statistic actually receives once the tokens have been round-tripped."""
    mapped = remap[tok]
    keep = mapped >= 0
    return pos[keep], mapped[keep], int((~keep).sum())


def bands_from(s1s, s2s, alpha=ALPHA):
    """The product's band construction (servseal.sampler.calibrate), on given draws."""
    q = alpha / 4.0
    return {"s1": (float(np.quantile(s1s, q)), float(np.quantile(s1s, 1 - q))),
            "s2": (float(np.quantile(s2s, q)), float(np.quantile(s2s, 1 - q))),
            "total": BUDGET, "reps": len(s1s), "alpha": alpha}


def half_width(b, key):
    lo, hi = b[key]
    return (hi - lo) / 2.0


# ----------------------------------------------------------------------- batteries

def run_model(name, do_power=True):
    from transformers import AutoTokenizer

    seal = os.path.join(CACHE, f"{name}_ref.seal.npz")
    pbase = os.path.join(CACHE, f"{name}_P_base.npy")
    if not (os.path.exists(seal) and os.path.exists(pbase)):
        print(f"  [{name}] cache missing -- run e2e_real_models.py / e2e_qwen.py first")
        return None

    snap = Snapshot.load(seal)
    vocab = snap.meta["vocab_size"]
    tok = AutoTokenizer.from_pretrained(TOKENISERS[name])
    P = np.load(pbase, mmap_mode="r")
    assert P.shape == (snap.meta["n_positions"], vocab), "cache does not match the seal"

    print(f"\n{'=' * 78}\n{name}  ({snap.meta['model']}, vocab {vocab:,}, "
          f"{P.shape[0]} positions)\n{'=' * 78}")

    # -- 1. how much of the vocabulary is not uniquely named on the wire -----------
    pieces, decoded, pad = wire_forms(tok, vocab)
    col_p, col_d = collisions(pieces), collisions(decoded)
    amb_p = sum(len(v) for v in col_p.values())
    amb_d = sum(len(v) for v in col_d.values())
    print("\n1. WIRE FORMS")
    print(f"   {'convention':10s} {'distinct':>9s} {'colliding ids':>14s} {'groups':>8s}")
    print(f"   {'piece':10s} {len(set(p for p in pieces if p is not None)):>9,} "
          f"{amb_p:>14,} {len(col_p):>8,}")
    print(f"   {'decoded':10s} {len(set(d for d in decoded if d is not None)):>9,} "
          f"{amb_d:>14,} {len(col_d):>8,}")
    if pad:
        print(f"   {pad} padded slots have no wire form at all "
              f"(model width {vocab:,} > tokeniser {len(tok):,})")
    if col_d:
        worst = sorted(col_d.items(), key=lambda kv: -len(kv[1]))[:3]
        for s, ids in worst:
            print(f"   largest group: {show(s):>16s} <- {len(ids)} ids {ids[:6]}"
                  f"{' ...' if len(ids) > 6 else ''}")

    # -- 2. what the product's rule does to every id -------------------------------
    remap, exact, wrong, split, padded = classify(tok, decoded, vocab)
    print("\n2. ROUND TRIP  (decode -> re-encode, the rule available at verify time)")
    for lbl, idx in (("exact", exact), ("split", split), ("wrong", wrong)):
        print(f"   {lbl:6s} {idx.size:>8,}  {100 * idx.size / vocab:5.2f} % of vocabulary")

    # What the ambiguity is made of decides which traffic this mode is safe on: the
    # three causes below are reached by different text, so the mean mass measured on
    # an English probe set does not transfer to code or to diacritical scripts.
    bad = np.concatenate([wrong, split]) if wrong.size or split.size else np.array([])
    causes = defaultdict(list)
    for i in bad:
        s = decoded[i] or ""
        if "�" in s:
            causes["byte fragment (multi-byte UTF-8 split)"].append(i)
        elif s.strip() == "" and s != "":
            causes["whitespace run (code, indentation)"].append(i)
        elif any(0x0300 <= ord(c) <= 0x036F for c in s):
            causes["combining mark (NFC/NFD normalisation)"].append(i)
        elif any(ord(c) > 127 for c in s):
            causes["non-ASCII text (precomposed)"].append(i)
        else:
            causes["ASCII, cause not classified"].append(i)
    print("   what the ambiguity is made of, and therefore which text reaches it:")
    for cause, ids in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        print(f"     {len(ids):>7,}  {100 * len(ids) / max(len(bad), 1):5.1f} %  {cause}"
              f"   e.g. {show((decoded[ids[0]] or '')[:12])}")

    # -- 3. the same thing weighted by the mass that is actually sampled -----------
    sets = {"split": split, "wrong": wrong, "padded": padded}
    mass, worst_pos, argmax_hit = mass_on(P, sets, snap.argmax)
    print("\n3. MASS  (share of the reference distribution, which is what gets sampled)")
    print(f"   {'class':8s} {'mean mass':>11s} {'worst position':>15s} {'argmax hits':>12s}")
    for k in ("split", "wrong", "padded"):
        print(f"   {k:8s} {100 * mass[k]:>10.4f} % {100 * worst_pos[k]:>14.3f} % "
              f"{argmax_hit[k]:>7,} /{len(snap.argmax):,}")
    corrupt = mass["wrong"]
    print(f"\n   silently corrupted mass = {100 * corrupt:.4f} %   "
          f"(dropped, recoverable: {100 * mass['split']:.4f} %)")

    # -- 4. what it costs the statistics the verdict is made of --------------------
    rng = np.random.default_rng(20260911)
    t0 = time.time()
    cum = build_cum(P)

    # the fast sampler is only allowed to stand in once it is shown to be the same one
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = draw(cum[:8], 40, r1)
    b = sample_stream(np.asarray(P[:8], dtype=np.float64), 40, r2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), \
        "the fast sampler is not the product's sampler"

    cal_clean, cal_wire, dropped = [], [], []
    for _ in range(REPS_CAL):
        pos, tk = draw(cum, BUDGET, rng)
        cal_clean.append(statistics(pos, tk, snap))
        wp, wt, nd = through_wire(pos, tk, remap)
        cal_wire.append(statistics(wp, wt, snap))
        dropped.append(nd)
    cal_clean, cal_wire = np.array(cal_clean), np.array(cal_wire)

    b_clean = bands_from(cal_clean[:, 0], cal_clean[:, 1])
    b_wire = bands_from(cal_wire[:, 0], cal_wire[:, 1])

    null_clean, null_wire = [], []
    for _ in range(REPS_NULL):
        pos, tk = draw(cum, BUDGET, rng)
        null_clean.append(statistics(pos, tk, snap))
        wp, wt, _ = through_wire(pos, tk, remap)
        null_wire.append(statistics(wp, wt, snap))
    null_clean, null_wire = np.array(null_clean), np.array(null_wire)

    d1 = float(null_wire[:, 0].mean() - null_clean[:, 0].mean())
    d2 = float(null_wire[:, 1].mean() - null_clean[:, 1].mean())
    h1, h2 = half_width(b_clean, "s1"), half_width(b_clean, "s2")
    fp_naive = float(np.mean([detect(a_, b_, b_clean) for a_, b_ in null_wire]))
    fp_recal = float(np.mean([detect(a_, b_, b_wire) for a_, b_ in null_wire]))

    print("\n4. END TO END  "
          f"(budget {BUDGET:,}, {BUDGET / P.shape[0]:.1f} samples per position)")
    print(f"   tokens dropped as unresolvable : {np.mean(dropped):.1f} / {BUDGET} "
          f"({100 * np.mean(dropped) / BUDGET:.3f} %)")
    print(f"   {'statistic':10s} {'shift from the round trip':>26s} "
          f"{'band half-width':>17s} {'ratio':>8s}")
    print(f"   {'S1':10s} {d1:>+26.6f} {h1:>17.6f} {abs(d1) / h1:>8.3f}")
    print(f"   {'S2':10s} {d2:>+26.6f} {h2:>17.6f} {abs(d2) / h2:>8.3f}")
    print(f"\n   false positives, bands calibrated WITHOUT the round trip : "
          f"{fp_naive:.2f}   (budget {ALPHA:.2f})")
    print(f"   false positives, bands calibrated THROUGH the round trip  : "
          f"{fp_recal:.2f}   (budget {ALPHA:.2f})")

    power = None
    if do_power:
        del cum
        gc.collect()
        cum_alt = build_cum(P, top_p=TOP_P)
        hits = 0
        for _ in range(REPS_ALT):
            pos, tk = draw(cum_alt, BUDGET, rng)
            wp, wt, _ = through_wire(pos, tk, remap)
            hits += detect(*statistics(wp, wt, snap), b_wire)
        power = hits / REPS_ALT
        print(f"   power against top-p {TOP_P}, round trip applied, recalibrated "
              f"bands : {power:.2f}")
        del cum_alt
    else:
        del cum
    gc.collect()

    print(f"\n   [{time.time() - t0:.0f}s]")
    return {"model": name, "corrupt_mass": corrupt, "drop_mass": mass["split"],
            "argmax_wrong": argmax_hit["wrong"], "shift_ratio": max(abs(d1) / h1,
                                                                    abs(d2) / h2),
            "fp_naive": fp_naive, "fp_recal": fp_recal, "power": power}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=f"any of {' '.join(TOKENISERS)}; "
                                              f"default: all cached")
    ap.add_argument("--no-power", action="store_true")
    args = ap.parse_args()
    names = args.models or list(TOKENISERS)
    unknown = [n for n in names if n not in TOKENISERS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; choose from {list(TOKENISERS)}")

    rows = [r for r in (run_model(n, not args.no_power) for n in names) if r]
    if not rows:
        print("\nnothing measured")
        return 1

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"{'model':8s} {'corrupted':>10s} {'dropped':>9s} {'argmax':>7s} "
          f"{'shift/band':>11s} {'FP naive':>9s} {'FP recal':>9s} {'power':>6s}")
    for r in rows:
        p = "  -  " if r["power"] is None else f"{r['power']:.2f}"
        print(f"{r['model']:8s} {100 * r['corrupt_mass']:>9.4f}% "
              f"{100 * r['drop_mass']:>8.4f}% {r['argmax_wrong']:>7,} "
              f"{r['shift_ratio']:>11.3f} {r['fp_naive']:>9.2f} {r['fp_recal']:>9.2f} "
              f"{p:>6s}")

    fails = []
    for r in rows:
        if r["fp_recal"] > 2 * ALPHA:
            fails.append(f"{r['model']}: recalibration does not restore the false-"
                         f"positive rate ({r['fp_recal']:.2f} > {2 * ALPHA:.2f})")
        if r["power"] is not None and r["power"] < 0.90:
            fails.append(f"{r['model']}: the round trip costs detection power "
                         f"({r['power']:.2f} < 0.90)")
        if r["shift_ratio"] > SHIFT_BUDGET and r["fp_naive"] > 2 * ALPHA:
            fails.append(f"{r['model']}: the shift is material and uncalibrated bands "
                         f"are unusable (FP {r['fp_naive']:.2f})")

    print()
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print("\nAPI mode is not free. Read the failures above before building the "
              "transport.")
        return 1
    print("  PASS  the round trip is affordable: recalibrating through it restores "
          "the\n        false-positive budget and leaves detection power intact.")
    print("\nBuild the transport. Calibrate bands THROUGH the round trip and store "
          "them\nin the .seal.npz at snapshot time -- that is the mitigation this run "
          "measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
