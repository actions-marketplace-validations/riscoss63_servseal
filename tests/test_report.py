"""The attestation page, both modes. It is the deliverable, so it is checked.

A report is what leaves the tool: it goes into a ticket, a release artifact, a
compliance file. Three things therefore have to hold whatever the input -- it renders
at all, it never carries a credential out of the process, and it does not describe a
sampled endpoint with the language of a weights-mode comparison, which would claim
more than was measured.
"""
import json
import re

import pytest

from servseal.report import render_api_body, render_body, render_report

WEIGHTS = {
    "metrics": {
        "mean_hellinger": 0.1537, "median_hellinger": 0.15, "max_hellinger": 0.2513,
        "noise_floor": 0.0884, "top1_agreement": 1.0, "positions_moved": 900,
        "n_positions": 1500, "aggregate_kl_lower_bound": 0.0,
        "mean_kl_lower_bound": 0.0, "structure_mismatch": False,
        "templates_differ": False,
        "hellinger_histogram": {"edges": [i / 24 for i in range(25)],
                                "counts": [10] * 24},
        "ref_meta": {"model": "gpt2@fp32", "D": 256, "probe": "abc",
                     "perplexity": 27.83, "servseal": "0.1.2"},
        "cand_meta": {"model": "gpt2@top-p", "D": 256, "probe": "abc",
                      "perplexity": 27.83, "servseal": "0.1.2"},
    },
    "verdict": {"status": "changed", "severity": "major",
                "signature": "serving-layer sampling filter (tail-only)",
                "explanation": "The tail moved.", "exit_code": 3},
}


def api_record(status="changed", s2=0.8064, url="https://api.example.com/v1",
               convention="piece"):
    return {
        "mode": "endpoint", "s1": 0.5631, "s2": s2,
        "bands": {"s1": (0.4818, 0.4977), "s2": (0.3568, 0.3815), "total": 5000,
                  "reps": 200, "alpha": 0.05, "through_wire": convention == "text"},
        "endpoint": {"url": url, "model": "gpt2"},
        "wire": {"convention": convention, "logprobs_available": convention != "text",
                 "requests": 48, "retries": 0, "returned": 5000, "resolved": 4998,
                 "unresolved": 2, "unresolved_fraction": 0.0004},
        "ref_meta": {"model": "gpt2@fp32", "model_id": "gpt2", "D": 256,
                     "n_positions": 1500, "probe": "0622744e", "dtype": "float32",
                     "servseal": "0.1.2",
                     "api": {"text_convention": {"wrong_mass": 0.000101}}},
        "verdict": {"status": status, "severity": "major" if status == "changed"
                    else "none",
                    "signature": "serving-layer sampling filter (head concentrated)",
                    "explanation": "The head is concentrated.",
                    "exit_code": 3 if status == "changed" else 0},
    }


# ----------------------------------------------------------------- both modes

def test_weights_mode_still_renders():
    h = render_body(WEIGHTS)
    assert "mean Hellinger" in h and "0.1537" in h
    assert "<svg" in h                       # the per-position histogram


def test_endpoint_record_is_dispatched_to_the_api_page():
    h = render_body(api_record())
    assert "sampled endpoint" in h
    assert "mean Hellinger" not in h, "the weights-mode page described a sample"


def test_standalone_document_is_well_formed():
    doc = render_report(api_record())
    assert doc.startswith("<!doctype html>") and "</html>" in doc
    assert "<title>" in doc and "sampled endpoint" in doc


# -------------------------------------------------------------- what it claims

def flat(h):
    """Prose wraps in the source, so match against it with whitespace collapsed."""
    return re.sub(r"\s+", " ", h)


def test_sealed_does_not_claim_identity():
    """SEALED from a sample is a weaker statement than SEALED from the weights."""
    h = flat(render_api_body(api_record(status="sealed", s2=0.37)))
    assert "inside the reference" in h          # apostrophe is escaped in the banner
    assert "not proof of identity" in h
    assert "measured detection power at the budget spent" in h


def test_the_figure_marks_the_outlier():
    outside = render_api_body(api_record(s2=0.8064))
    inside = render_api_body(api_record(status="sealed", s2=0.3700))
    assert outside.count("var(--changed)") > inside.count("var(--changed)")


CELL = '<div class="k">probe-set exposure</div>'


def test_text_convention_surfaces_the_probe_set_exposure():
    h = render_api_body(api_record(convention="text"))
    assert CELL in h and "0.0101" in h
    assert "through the text round trip" in flat(h)


def test_piece_convention_does_not_show_an_exposure_it_does_not_have():
    """The cell, not the word: the footer explains the number in prose either way."""
    assert CELL not in render_api_body(api_record(convention="piece"))


# ------------------------------------------------------------------- hygiene

@pytest.mark.parametrize("url,leaked,kept", [
    ("https://user:s3cr3t@api.example.com:8000/v1", "s3cr3t", "api.example.com:8000"),
    ("https://api.example.com/v1?api_key=AKIAZZZ", "AKIAZZZ", "api.example.com"),
    ("http://127.0.0.1:9001/v1", "", "127.0.0.1:9001"),
])
def test_credentials_never_reach_the_page(url, leaked, kept):
    h = render_api_body(api_record(url=url))
    assert kept in h
    if leaked:
        assert leaked not in h, f"{leaked!r} survived into the attestation"


def test_verdict_text_is_escaped_not_injected():
    rec = api_record()
    rec["verdict"]["explanation"] = "<script>alert(1)</script> & more"
    h = render_api_body(rec)
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


def test_record_survives_a_json_round_trip():
    """The CLI writes --json and `servseal report` reads it back; tuples become lists."""
    rec = json.loads(json.dumps(api_record()))
    assert isinstance(rec["bands"]["s1"], list)
    assert "0.4818" in render_api_body(rec)


def test_no_external_script_or_image_is_pulled():
    h = render_report(api_record())
    assert "<script" not in h.lower()
    srcs = re.findall(r'src=["\']([^"\']+)', h)
    assert not srcs, f"the attestation must be self-contained, found {srcs}"
