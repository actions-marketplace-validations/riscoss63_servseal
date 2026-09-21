"""The API-mode transport: recovering ids from the wire, and sampling a live endpoint.

The endpoint tests run against a real HTTP server -- a thread serving an
OpenAI-compatible `/v1/completions` from known distributions -- rather than a mocked
client, because every defect this layer can have lives in the wire format: how a batch
of prompts flattens into `choices`, which field carries the token, what a 429 does.
A mock would assert the shape this code already assumes.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from servseal.endpoint import EndpointError, sample_endpoint
from servseal.sampler import calibrate_bands, statistics
from servseal.snapshot import Snapshot
from servseal.verdict import classify_api
from servseal.wire import position_prefixes

N_POS, VOCAB = 40, 64


# --------------------------------------------------------------------- a fake wire

class FakeWire:
    """A WireMap's surface, over a vocabulary small enough to read.

    Ids 0..VOCAB-1 have piece 't<id>'. Their text forms collide in pairs above 60, so
    the unresolvable path is exercised rather than assumed absent.
    """

    vocab_size = VOCAB

    def __init__(self):
        self.remap = np.arange(VOCAB, dtype=np.int64)
        self.remap[61] = -1
        self.remap[63] = -1

    def piece(self, i):
        return f"t{i}"

    def text(self, i):
        return "AMBIG" if self.remap[i] < 0 else f"T{i}"

    def from_piece(self, s):
        return int(s[1:]) if s and s.startswith("t") and s[1:].isdigit() else -1

    def from_text(self, s):
        return int(s[1:]) if s and s.startswith("T") and s[1:].isdigit() else -1


# ------------------------------------------------------------------- a fake server

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        cfg = self.server.cfg
        if cfg["fail_first"] > 0:
            cfg["fail_first"] -= 1
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b"slow down")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # the protocol must never switch a server-side nucleus filter off
        assert "top_p" not in body, "the client must not send top_p"
        assert body["max_tokens"] == 1 and body["temperature"] == 1.0
        prompts = body["prompt"]
        n = int(body.get("n", 1))
        rng = cfg["rng"]
        choices = []
        for pi, prompt in enumerate(prompts):
            pos = cfg["index"][tuple(prompt)]
            row = cfg["P"][pos]
            ids = rng.choice(len(row), size=n, p=row / row.sum())
            for ci, tid in enumerate(ids):
                c = {"index": pi * n + ci, "text": cfg["wire"].text(int(tid))}
                if cfg["mode"] != "text":
                    piece = (f"token_id:{int(tid)}" if cfg["mode"] == "token_id"
                             else cfg["wire"].piece(int(tid)))
                    c["logprobs"] = {"tokens": [piece], "token_logprobs": [-1.0]}
                choices.append(c)
        out = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def serve(P, prefixes, wire, mode="piece", fail_first=0):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.cfg = {"P": P, "wire": wire, "mode": mode, "fail_first": fail_first,
               "rng": np.random.default_rng(5),
               "index": {tuple(p): i for i, p in enumerate(prefixes)}}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.full(VOCAB, 0.3), size=N_POS).astype(np.float32)
    snap = Snapshot.from_distributions(P, probe="p", model="fake", D=256, seed=0)
    prefixes = [[1, 2, i] for i in range(N_POS)]
    return P, snap, prefixes, FakeWire()


def _tail_cut(P, keep=0.85):
    srt = -np.sort(-P, 1)
    thr = srt[np.arange(len(P)), np.argmax(np.cumsum(srt, 1) >= keep, 1)][:, None]
    Q = np.where(P >= thr, P, 0.0)
    return (Q / Q.sum(1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------------- wire

def test_position_prefixes_are_prefixes_in_order():
    tok = type("T", (), {"__call__": lambda s, t, **k: {"input_ids": [7, 8, 9]}})()
    got = position_prefixes(tok, ["a", "b"], max_positions=5, max_length=96)
    assert got == [[7], [7, 8], [7, 8, 9], [7], [7, 8]]


def test_position_prefixes_break_before_a_text_not_inside_it():
    """Mirrors runner.softmax_over_probes: a text is taken whole, then the cap bites."""
    tok = type("T", (), {"__call__": lambda s, t, **k: {"input_ids": [1, 2, 3]}})()
    assert len(position_prefixes(tok, ["a"] * 5, max_positions=4)) == 4


# --------------------------------------------------------------------- transport

@pytest.mark.parametrize("mode", ["piece", "token_id", "text"])
def test_every_wire_convention_is_detected_and_inverted(world, mode):
    P, snap, prefixes, wire = world
    srv, url = serve(P, prefixes, wire, mode=mode)
    try:
        pos, tok, rep = sample_endpoint(url, "m", prefixes, wire, total=400,
                                        batch=8, concurrency=4)
    finally:
        srv.shutdown()
    assert rep["convention"] == mode
    assert rep["logprobs_available"] is (mode != "text")
    assert pos.min() >= 0 and pos.max() < N_POS and tok.max() < VOCAB
    # only the deliberately ambiguous ids may be lost, and only on the text path
    assert rep["unresolved"] == 0 if mode != "text" else rep["unresolved"] >= 0
    assert len(pos) == len(tok) == rep["resolved"]


def test_batching_maps_choices_back_to_the_right_positions(world):
    """The flattening index*n+c is the one place a batch can silently scramble."""
    P, snap, prefixes, wire = world
    peaked = np.full((N_POS, VOCAB), 1e-9, dtype=np.float32)
    for i in range(N_POS):
        peaked[i, i % VOCAB] = 1.0
    peaked /= peaked.sum(1, keepdims=True)
    srv, url = serve(peaked, prefixes, wire, mode="piece")
    try:
        pos, tok, _ = sample_endpoint(url, "m", prefixes, wire, total=400,
                                      batch=7, concurrency=3)
    finally:
        srv.shutdown()
    assert np.all(tok == pos % VOCAB), "a batched choice landed on the wrong position"


def test_retries_a_429_and_reports_it(world):
    P, snap, prefixes, wire = world
    srv, url = serve(P, prefixes, wire, fail_first=3)
    try:
        _, _, rep = sample_endpoint(url, "m", prefixes, wire, total=200, batch=20,
                                    concurrency=1, max_retries=4)
    finally:
        srv.shutdown()
    assert rep["retries"] >= 3


def test_unreachable_endpoint_raises(world):
    _, _, prefixes, wire = world
    with pytest.raises(EndpointError):
        sample_endpoint("http://127.0.0.1:9/v1", "m", prefixes, wire, total=20,
                        max_retries=0)


# ------------------------------------------------------------------ end to end

def test_unchanged_endpoint_is_sealed(world):
    P, snap, prefixes, wire = world
    bands = calibrate_bands(P, snap, total=400, reps=120, seed=3)
    srv, url = serve(P, prefixes, wire, mode="piece")
    try:
        pos, tok, rep = sample_endpoint(url, "m", prefixes, wire, total=400, batch=8)
    finally:
        srv.shutdown()
    v = classify_api(*statistics(pos, tok, snap), bands, rep)
    assert v.status == "sealed", v.explanation


def test_tail_filtered_endpoint_is_caught_as_serving_layer(world):
    P, snap, prefixes, wire = world
    bands = calibrate_bands(P, snap, total=400, reps=120, seed=3)
    srv, url = serve(_tail_cut(P), prefixes, wire, mode="piece")
    try:
        pos, tok, rep = sample_endpoint(url, "m", prefixes, wire, total=400, batch=8)
    finally:
        srv.shutdown()
    v = classify_api(*statistics(pos, tok, snap), bands, rep)
    assert v.status == "changed", v.explanation


# ------------------------------------------------------------------ calibration

def test_calibrate_bands_holds_its_false_positive_budget(world):
    P, snap, _, _ = world
    bands = calibrate_bands(P, snap, total=400, reps=200, alpha=0.05, seed=1)
    from servseal.sampler import detect, sample_stream
    rng = np.random.default_rng(99)
    fp = np.mean([detect(*statistics(*sample_stream(P, 400, rng), snap), bands)
                  for _ in range(120)])
    assert fp <= 0.15, f"false positive rate {fp}"


def test_calibrate_bands_through_wire_drops_only_the_ambiguous(world):
    P, snap, _, wire = world
    b = calibrate_bands(P, snap, total=400, reps=40, seed=1, wiremap=wire)
    assert b["through_wire"] is True
    lost = P[:, [61, 63]].sum() / len(P)
    assert 0 < b["dropped_per_run"] < 400 * (lost + 0.05)


def test_bands_are_json_serialisable():
    """They travel inside the snapshot's metadata, which is JSON."""
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.full(16, 0.5), size=8).astype(np.float32)
    snap = Snapshot.from_distributions(P, probe="p", model="m")
    json.dumps(calibrate_bands(P, snap, total=80, reps=10))


# ------------------------------------------ the real tokeniser, the real resolver

@pytest.fixture(scope="module")
def gpt2_tok():
    """Skip only what needs a tokeniser, never the module.

    This was `pytest.importorskip("transformers")` at module scope, which skips
    everything after it -- so in an environment without transformers, the HTTP
    transport tests above vanished silently. That environment is the CI unit job:
    37 tests ran there instead of 66, and the ones that need nothing but stdlib and
    numpy were the ones lost.
    """
    pytest.importorskip("transformers", exc_type=ImportError)
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:                                  # offline, no cache
        pytest.skip(f"gpt2 tokeniser unavailable: {e}")


def test_piece_convention_is_collision_free_on_a_real_tokeniser(gpt2_tok):
    """The claim the transport's fast path rests on, checked rather than cited."""
    from servseal.wire import WireMap
    wm = WireMap(gpt2_tok, 50257)
    assert wm.piece_collisions == 0
    assert len(wm.piece_index) == 50257


def test_from_piece_inverts_every_id_exactly(gpt2_tok):
    from servseal.wire import WireMap
    wm = WireMap(gpt2_tok, 50257)
    pieces = gpt2_tok.convert_ids_to_tokens(list(range(50257)))
    assert all(wm.from_piece(p) == i for i, p in enumerate(pieces))


def test_text_convention_resolves_almost_everything(gpt2_tok):
    from servseal.wire import WireMap
    wm = WireMap(gpt2_tok, 50257)
    s = wm.stats()
    assert s["exact"] + s["wrong"] + s["split"] + s["padded"] == 50257
    assert s["exact_fraction"] > 0.99          # measured 0.9932 in token_recovery.py


def test_corrupted_mass_is_tiny_on_english_and_is_reported(gpt2_tok):
    from servseal.wire import WireMap, corrupted_mass
    wm = WireMap(gpt2_tok, 50257)
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.full(50257, 0.01), size=8).astype(np.float32)
    m = corrupted_mass(P, wm)
    assert set(m) == {"wrong_mass", "split_mass", "worst_position_wrong_mass"}
    assert 0.0 <= m["wrong_mass"] <= 1.0


def test_real_wiremap_round_trips_through_a_live_endpoint(gpt2_tok):
    """Real vocabulary, real resolver, real HTTP: the shipped path end to end."""
    from servseal.wire import WireMap
    vocab = 50257
    wm = WireMap(gpt2_tok, vocab)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.full(vocab, 0.02), size=24).astype(np.float32)
    snap = Snapshot.from_distributions(P, probe="p", model="gpt2-ish")
    prefixes = [[100, 200, i] for i in range(24)]
    bands = calibrate_bands(P, snap, total=600, reps=120, seed=3)

    class RealWire(FakeWire):
        vocab_size = vocab

        def __init__(self):
            self.remap = wm.remap

        def piece(self, i):
            return gpt2_tok.convert_ids_to_tokens([i])[0]

        def text(self, i):
            return gpt2_tok.decode([i])

        def from_piece(self, s):
            return wm.from_piece(s)

        def from_text(self, s):
            return wm.from_text(s)

    rw = RealWire()
    srv, url = serve(P, prefixes, rw, mode="piece")
    try:
        pos, tok, rep = sample_endpoint(url, "m", prefixes, wm, total=600, batch=6)
    finally:
        srv.shutdown()
    assert rep["convention"] == "piece" and rep["unresolved"] == 0
    v = classify_api(*statistics(pos, tok, snap), bands, rep)
    assert v.status == "sealed", v.explanation


# --------------------------------------------- the direction of the S2 excursion

BANDS = {"s1": (0.40, 0.50), "s2": (0.30, 0.40), "total": 5000}


def test_sealed_inside_both_bands():
    assert classify_api(0.45, 0.35, BANDS).status == "sealed"


def test_argmax_emitted_more_often_is_a_serving_filter():
    """A tail cut concentrates the sample on the head, so S2 goes UP.

    Measured on GPT-2 at top-p 0.95: S2 0.373 -> 0.807. Reading this the weights-mode
    way -- 'both coordinates moved, therefore the weights moved' -- names the one case
    the tool exists for as its opposite, so the direction is pinned here.
    """
    v = classify_api(0.56, 0.81, BANDS)
    assert v.status == "changed" and v.severity == "major"
    assert "serving-layer" in v.signature and "concentrated" in v.signature


def test_argmax_emitted_less_often_is_a_head_level_change():
    v = classify_api(0.56, 0.12, BANDS)
    assert v.status == "changed" and "head-level" in v.signature


def test_tail_moves_while_head_frequency_holds():
    v = classify_api(0.60, 0.35, BANDS)
    assert v.status == "changed" and "tail reshaped" in v.signature


def test_unresolved_tokens_are_flagged_in_the_explanation():
    v = classify_api(0.56, 0.81, BANDS, {"unresolved_fraction": 0.07,
                                         "convention": "text"})
    assert "did not resolve" in v.explanation and "logprobs" in v.explanation


def test_live_top_p_endpoint_is_named_a_serving_filter(world):
    """The same claim, through HTTP, from sampling rather than from a hand-set pair."""
    P, snap, prefixes, wire = world
    bands = calibrate_bands(P, snap, total=600, reps=150, seed=3)
    srv, url = serve(_tail_cut(P, keep=0.85), prefixes, wire, mode="piece")
    try:
        pos, tok, rep = sample_endpoint(url, "m", prefixes, wire, total=600, batch=8)
    finally:
        srv.shutdown()
    v = classify_api(*statistics(pos, tok, snap), bands, rep)
    assert v.status == "changed", v.explanation
    assert "serving-layer" in v.signature, f"named {v.signature!r}: {v.explanation}"


# ------------------------------------- an endpoint that does not honour the batch

class _CappedHandler(_Handler):
    """Returns one completion per prompt however many were asked for.

    A real risk, not a hypothetical: `n` is optional in the OpenAI completions API
    and a provider may cap it. The flattening index = p*n + c then decodes every
    index to the wrong probe position, and nothing about the resulting numbers looks
    wrong -- they are simply computed against other positions' sketches.
    """

    def do_POST(self):
        cfg = self.server.cfg
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        choices = []
        for pi, prompt in enumerate(body["prompt"]):
            row = cfg["P"][cfg["index"][tuple(prompt)]]
            tid = int(cfg["rng"].choice(len(row), p=row / row.sum()))
            choices.append({"index": pi, "text": cfg["wire"].text(tid),
                            "logprobs": {"tokens": [cfg["wire"].piece(tid)]}})
        out = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_an_endpoint_that_ignores_n_is_refused_not_misattributed(world):
    P, snap, prefixes, wire = world
    srv = HTTPServer(("127.0.0.1", 0), _CappedHandler)
    srv.cfg = {"P": P, "wire": wire, "mode": "piece", "fail_first": 0,
               "rng": np.random.default_rng(5),
               "index": {tuple(p): i for i, p in enumerate(prefixes)}}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        with pytest.raises(EndpointError, match="does not honour"):
            sample_endpoint(url, "m", prefixes, wire, total=400, batch=8,
                            concurrency=1)
    finally:
        srv.shutdown()
