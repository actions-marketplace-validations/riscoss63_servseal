"""Sampling an endpoint you can only sample from. Stdlib only, no new dependency.

The protocol is one next token per probe position, repeated: `max_tokens=1`, the same
prefixes the reference was sketched on, and the returned tokens put back on reference
ids by `wire.WireMap`. What comes back is already shaped by whatever the provider does
at serve time, which is not a limitation of the test -- it is the object under test.

Two request fields are deliberate and neither is a default:

  temperature=1 is sent.  The reference is the raw softmax, so anything else compares
                          two different objects. If the endpoint ignores or clamps it,
                          that is a finding, not a nuisance.
  top_p is NOT sent.      Sending top_p=1.0 would switch off a server-side nucleus
                          filter -- the single change this tool exists to catch, and
                          the one that leaves perplexity and every argmax untouched.
                          Whatever the deployment does by default must reach us.

Which convention the endpoint reports tokens in is not knowable in advance, so it is
detected on the first response rather than configured:

  token_id:NNN  vLLM's --return-tokens-as-token-ids. Exact, nothing to invert.
  piece         the raw token. Measured collision-free on three tokenisers: exact.
  text          a bare completion field. Costs ~0.01 % of the sampled mass on English
                probes, more on code and non-Latin scripts -- which is why the
                snapshot carries that number for its own probe set.

Ask for logprobs. The text path works and is measured, but the first two are free.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["EndpointError", "sample_endpoint"]

_TOKEN_ID = re.compile(r"^token_id:(\d+)$")
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class EndpointError(RuntimeError):
    """The endpoint could not be sampled as the protocol requires."""


def _post(url, payload, headers, timeout, max_retries):
    """One request, with backoff on the statuses a busy endpoint really returns."""
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8")), attempt
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
            if e.code not in _RETRY_STATUS or attempt == max_retries:
                raise EndpointError(last) from None
            wait = float(e.headers.get("Retry-After") or 0) or min(30.0, 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
            if attempt == max_retries:
                raise EndpointError(last) from None
            wait = min(30.0, 2 ** attempt)
        time.sleep(wait + random.random() * 0.5)
    raise EndpointError(last or "unreachable")


def _wire_values(choice):
    """The token strings one choice carries, preferring logprobs over the text field."""
    lp = choice.get("logprobs") or {}
    toks = lp.get("tokens")
    if toks:
        return list(toks), True
    text = choice.get("text")
    return ([text] if text is not None else []), False


def _detect(values, wiremap):
    """Pick the convention on evidence: whichever resolves most of a first sample."""
    if not values:
        return "text"
    scores = {}
    ids = sum(1 for v in values if _TOKEN_ID.match(v or ""))
    scores["token_id"] = ids / len(values)
    scores["piece"] = sum(1 for v in values
                          if wiremap.from_piece(v) >= 0) / len(values)
    scores["text"] = sum(1 for v in values
                         if wiremap.from_text(v or "") >= 0) / len(values)
    return max(scores, key=scores.get)


def _resolve(value, convention, wiremap):
    if value is None:
        return -1
    if convention == "token_id":
        m = _TOKEN_ID.match(value)
        return int(m.group(1)) if m else -1
    if convention == "piece":
        return wiremap.from_piece(value)
    return wiremap.from_text(value)


def sample_endpoint(base_url, model, prefixes, wiremap, *, total=5000, api_key=None,
                    batch=16, concurrency=8, timeout=60.0, max_retries=5,
                    extra_headers=None, progress=None):
    """Sample `total` next tokens spread over `prefixes`, and put them back on ids.

    `prefixes` are token-id lists from `wire.position_prefixes`, one per snapshot
    position and in that order -- sending token ids rather than text keeps the prompt
    side exact too, so neither end of the measurement depends on detokenisation.

    Returns (pos, tok, report) with pos/tok shaped for `sampler.statistics`.
    """
    url = base_url.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra_headers or {})

    n_pos = len(prefixes)
    if n_pos == 0:
        raise EndpointError("no probe positions")
    per = np.full(n_pos, total // n_pos, dtype=np.int64)
    per[: total - int(per.sum())] += 1

    # one request carries `batch` prefixes that need the same number of completions,
    # which keeps the request count near total/(batch*n) instead of at total
    jobs = []
    for k in sorted(set(int(x) for x in per if x > 0)):
        idx = [i for i in range(n_pos) if per[i] == k]
        for s in range(0, len(idx), batch):
            jobs.append((idx[s:s + batch], k))

    state = {"convention": None, "requests": 0, "retries": 0, "returned": 0,
             "unresolved": 0, "examples": [], "done": 0}

    def run(job):
        idx, k = job
        payload = {"model": model, "prompt": [prefixes[i] for i in idx],
                   "max_tokens": 1, "temperature": 1.0, "logprobs": 1, "n": k}
        data, attempts = _post(url, payload, headers, timeout, max_retries)
        choices = data.get("choices")
        if not choices:
            raise EndpointError(f"no choices in response: {str(data)[:200]}")
        # A batch of P prompts with n completions each flattens to index = p*n + c,
        # which is how a choice is put back on its probe position. An endpoint that
        # caps or ignores `n` returns a shorter list and every index then decodes to
        # the wrong position -- silently, and the statistics would be computed against
        # the wrong reference sketches. Refuse instead: at n=1 the mapping is trivial
        # and the caller can retry there.
        if len(choices) != len(idx) * k:
            raise EndpointError(
                f"asked for {len(idx)} prompts x n={k} and got {len(choices)} "
                f"choices. This endpoint does not honour `n`; rerun with "
                f"--batch 1 and a budget equal to the position count, or use one "
                f"that does.")
        out = []
        sawlp = True
        for c in choices:
            j = int(c.get("index", 0))
            if not 0 <= j < len(idx) * k:
                raise EndpointError(f"choice index {j} outside the batch")
            vals, lp = _wire_values(c)
            sawlp = sawlp and lp
            out.extend((idx[j // k], v) for v in vals)
        return out, attempts, sawlp

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for out, attempts, sawlp in pool.map(run, jobs):
            results.extend(out)
            state["requests"] += 1
            state["retries"] += attempts
            state["logprobs"] = state.get("logprobs", True) and sawlp
            state["done"] += 1
            if progress:
                progress(state["done"], len(jobs))

    if not results:
        raise EndpointError("the endpoint returned nothing")
    state["returned"] = len(results)
    state["convention"] = _detect([v for _, v in results[:512]], wiremap)

    pos, tok = [], []
    for p, v in results:
        i = _resolve(v, state["convention"], wiremap)
        if i < 0 or i >= wiremap.vocab_size:
            state["unresolved"] += 1
            if len(state["examples"]) < 8:
                state["examples"].append(ascii(v))
            continue
        pos.append(p)
        tok.append(i)

    if not pos:
        raise EndpointError(
            f"nothing resolved: convention {state['convention']!r}, "
            f"examples {state['examples']}. Does the endpoint serve this tokeniser?")

    report = {"convention": state["convention"],
              "logprobs_available": bool(state.get("logprobs")),
              "requests": state["requests"], "retries": state["retries"],
              "returned": state["returned"], "resolved": len(pos),
              "unresolved": state["unresolved"],
              "unresolved_fraction": state["unresolved"] / state["returned"],
              "unresolved_examples": state["examples"]}
    return np.asarray(pos, dtype=np.int64), np.asarray(tok, dtype=np.int64), report
