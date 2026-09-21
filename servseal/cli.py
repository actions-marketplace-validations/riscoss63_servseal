"""The command line: snapshot, verify, report, probes.

Exit codes are the contract, so a pipeline can gate on them:

    0  sealed        the served model behaves as the reference
    1  tool error    bad arguments, missing file, model failed to load
    2  incomparable  the snapshots measured different things; no attestation made
    3  changed       the served model does NOT behave as the reference

Output is ASCII, one fact per line, machine-greppable; --json gives the whole record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .probes import load_probes, probe_id
from .snapshot import Snapshot
from .verdict import EXIT_ERROR, EXIT_INCOMPARABLE, classify, classify_api

__all__ = ["main"]


def _cmd_snapshot(a):
    from .runner import snapshot_model                     # torch only here
    snap = snapshot_model(a.model, probes=a.probes, D=a.D, seed=a.seed,
                          max_positions=a.positions, max_length=a.max_length,
                          template=a.template, dtype=a.dtype, label=a.label,
                          api_budgets=a.api_budget or None)
    snap.save(a.out)
    m = snap.meta
    print(f"snapshot   {a.out}")
    print(f"model      {m['model']}  dtype={m['dtype']}")
    print(f"positions  {m['n_positions']}  D={m['D']}  size={snap.nbytes() / 1024:.0f} KB")
    print(f"probe      {m['probe']}  ({m.get('probe_file')})")
    print(f"perplexity {m.get('perplexity')}")
    api = m.get("api")
    if api:
        tc = api["text_convention"]
        print(f"api        bands at budgets {', '.join(api['bands'])} "
              f"(exact + text conventions)")
        print(f"api_text   corrupted mass {100 * tc['wrong_mass']:.4f} %"
              f"  worst position {100 * tc['worst_position_wrong_mass']:.3f} %"
              + ("   OK" if tc["wrong_mass"] < 1e-3 else
                 "   HIGH -- require an endpoint that returns logprobs"))
    return 0


def _emit(verdict, record, a):
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
    if a.report:
        from .report import render_report
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(render_report(record))
        print(f"report     {a.report}")
    return verdict.exit_code


def _cmd_verify_endpoint(a):
    """Verify against something that can only be sampled. Loads no weights.

    Everything this needs that cannot be recovered from a black box -- the acceptance
    bands, and how much of the mass the text convention would move -- was computed at
    snapshot time and travels in the .seal.npz. A reference sealed without
    --api-budget cannot be used here, and says so rather than inventing a threshold.
    """
    from transformers import AutoTokenizer

    from .endpoint import sample_endpoint
    from .sampler import statistics
    from .wire import WireMap, position_prefixes

    ref = Snapshot.load(a.reference)
    api = ref.meta.get("api")
    if not api:
        print("error      this snapshot carries no API calibration; re-seal the "
              "reference with\n           --api-budget N. The bands can only be "
              "computed where the reference\n           distributions still exist.",
              file=sys.stderr)
        return EXIT_INCOMPARABLE
    key = str(int(a.budget))
    if key not in api["bands"]:
        print(f"error      no bands at budget {key}; this snapshot has "
              f"{', '.join(api['bands'])}", file=sys.stderr)
        return EXIT_INCOMPARABLE
    if not a.served_model:
        print("error      --served-model is required with --endpoint", file=sys.stderr)
        return EXIT_ERROR

    texts = load_probes(a.probes or ref.meta.get("probe_file"))
    if probe_id(texts) != ref.meta["probe"]:
        print(f"error      probe set does not match the reference "
              f"({probe_id(texts)} vs {ref.meta['probe']})", file=sys.stderr)
        return EXIT_INCOMPARABLE

    tok_id = a.tokenizer or ref.meta.get("model_id")
    if not tok_id:
        print("error      --tokenizer is required: this snapshot predates model_id "
              "in its metadata", file=sys.stderr)
        return EXIT_ERROR
    tok = AutoTokenizer.from_pretrained(tok_id)
    prefixes = position_prefixes(tok, texts, max_positions=ref.meta["n_positions"],
                                 max_length=ref.meta.get("max_length", 96),
                                 template=ref.meta.get("template"))
    if len(prefixes) != ref.meta["n_positions"]:
        print(f"error      rebuilt {len(prefixes)} positions, the reference has "
              f"{ref.meta['n_positions']}: this is not the tokeniser that sealed it",
              file=sys.stderr)
        return EXIT_INCOMPARABLE

    wm = WireMap(tok, ref.meta["vocab_size"])
    t0 = time.time()
    pos, tk, report = sample_endpoint(
        a.endpoint, a.served_model, prefixes, wm, total=int(a.budget),
        api_key=os.environ.get(a.api_key_env) if a.api_key_env else None,
        batch=a.batch, concurrency=a.concurrency)
    s1, s2 = statistics(pos, tk, ref)

    bands = api["bands"][key]["text" if report["convention"] == "text" else "exact"]
    verdict = classify_api(s1, s2, bands, report)

    print(f"verdict    {verdict.status.upper()}"
          + (f"  severity={verdict.severity}" if verdict.status == "changed" else ""))
    print(f"signature  {verdict.signature}")
    print(f"s1         {s1:.4f}  band {bands['s1'][0]:.4f}-{bands['s1'][1]:.4f}"
          f"  (distributional, sees the tail)")
    print(f"s2         {s2:.4f}  band {bands['s2'][0]:.4f}-{bands['s2'][1]:.4f}"
          f"  (how often the reference argmax is emitted)")
    print(f"wire       convention={report['convention']} "
          f"logprobs={'yes' if report['logprobs_available'] else 'no'} "
          f"unresolved={report['unresolved_fraction']:.4%}")
    print(f"sampling   {report['resolved']}/{report['returned']} tokens over "
          f"{len(prefixes)} positions, {report['requests']} requests, "
          f"{report['retries']} retries, {time.time() - t0:.0f}s")
    record = {"mode": "endpoint", "s1": s1, "s2": s2, "bands": bands,
              "endpoint": {"url": a.endpoint, "model": a.served_model},
              "wire": report, "ref_meta": ref.meta, "verdict": verdict.as_dict()}
    return _emit(verdict, record, a)


def _cmd_verify(a):
    if a.endpoint:
        return _cmd_verify_endpoint(a)
    if not a.candidate:
        print("error      give a candidate snapshot, or --endpoint URL",
              file=sys.stderr)
        return EXIT_ERROR
    ref = Snapshot.load(a.reference)
    cand = Snapshot.load(a.candidate)
    metrics = ref.compare(cand)
    verdict = classify(metrics)
    record = {"metrics": metrics, "verdict": verdict.as_dict()}

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
    print(f"verdict    {verdict.status.upper()}"
          + (f"  severity={verdict.severity}" if verdict.status == "changed" else ""))
    print(f"signature  {verdict.signature}")
    if not metrics.get("incomparable"):
        print(f"hellinger  mean={metrics['mean_hellinger']:.4f}"
              f"  max={metrics['max_hellinger']:.4f}"
              f"  floor={metrics['noise_floor']:.4f}")
        if metrics.get("top1_agreement") is not None:
            print(f"top1       agreement={metrics['top1_agreement']:.4f}")
        klb = metrics.get('mean_kl_lower_bound',
                          max(0.0, metrics['aggregate_kl_lower_bound']))
        print(f"kl_bound   mean per-position KL >= {klb:.4f}"
              f"  (certified; 0 certifies nothing, not absence of change)")
    else:
        print(f"reason     {metrics['incomparable']}")
    if a.report:
        from .report import render_report
        with open(a.report, "w", encoding="utf-8") as fh:
            fh.write(render_report(record))
        print(f"report     {a.report}")
    return verdict.exit_code


def _cmd_report(a):
    from .report import load_result, render_report
    record = load_result(a.result)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(render_report(record))
    print(f"report     {a.out}")
    return 0


def _cmd_probes(a):
    texts = load_probes(a.set)
    print(f"probe set  {a.set or 'default-v1'}")
    print(f"texts      {len(texts)}")
    print(f"probe id   {probe_id(texts)}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="servseal",
        description="Behavioural attestation for deployed language models: "
                    "is the model you serve the model you validated?")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="fingerprint a model over the probe set")
    s.add_argument("model", help="HF model id or local path")
    s.add_argument("-o", "--out", required=True, help="output .seal.npz file")
    s.add_argument("--probes", default=None, help="bundled set name or probe file")
    s.add_argument("--positions", type=int, default=1500)
    s.add_argument("--max-length", type=int, default=96)
    s.add_argument("--D", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    s.add_argument("--template", default=None,
                   help="prompt template with {text}, if the deployment applies one")
    s.add_argument("--label", default=None)
    s.add_argument("--api-budget", type=int, action="append", default=[], metavar="N",
                   help="also calibrate API-mode acceptance bands at this sampling "
                        "budget (repeatable). Only possible here, while the reference "
                        "distributions exist.")
    s.set_defaults(fn=_cmd_snapshot)

    v = sub.add_parser("verify", help="compare a candidate snapshot to a reference")
    v.add_argument("reference")
    v.add_argument("candidate", nargs="?", default=None,
                   help="a second snapshot; omit it and give --endpoint instead")
    v.add_argument("--endpoint", default=None,
                   help="OpenAI-compatible base URL (.../v1) to sample")
    v.add_argument("--served-model", default=None,
                   help="the model name that endpoint expects")
    v.add_argument("--budget", type=int, default=5000,
                   help="sampled tokens; must match a calibrated band")
    v.add_argument("--tokenizer", default=None,
                   help="HF id of the reference tokeniser (default: the snapshot's)")
    v.add_argument("--probes", default=None)
    v.add_argument("--api-key-env", default="SERVSEAL_API_KEY", metavar="VAR",
                   help="environment variable holding the bearer token")
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--concurrency", type=int, default=8)
    v.add_argument("--json", default=None, help="write the full record here")
    v.add_argument("--report", default=None, help="write an HTML attestation here")
    v.set_defaults(fn=_cmd_verify)

    r = sub.add_parser("report", help="render an HTML attestation from a verify --json")
    r.add_argument("result")
    r.add_argument("-o", "--out", required=True)
    r.set_defaults(fn=_cmd_report)

    pr = sub.add_parser("probes", help="show a probe set and its id")
    pr.add_argument("--set", default=None)
    pr.set_defaults(fn=_cmd_probes)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"error      {e}", file=sys.stderr)
        return EXIT_ERROR
    except ValueError as e:
        print(f"error      {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
