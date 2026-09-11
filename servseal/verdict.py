"""Turning measurements into a verdict someone can act on.

The measurements are distances; the product is a decision. The thresholds here were not
chosen by taste: they are calibrated on real perturbations of real models, applied where
a deployment would apply them (experiments/e2e_real_models.py), and the calibration run
is committed next to this file. The bands they draw:

    mean Hellinger over the probe positions
      < UNCHANGED .......... behaviour identical to the reference (weights-mode
                             snapshots are deterministic, so identical means ~0)
      < MINOR .............. precision-level: bfloat16 rounding measures 0.036, a
                             temperature nudge to 1.05 measures 0.044
      < MODERATE ........... something structural moved
      otherwise ............ major: int8 weight quantisation measures 0.18, a top-p
                             serving filter 0.15

The *signature* uses a second, independent coordinate: how often the most likely token
changed. A serving-layer filter (top-p, min-p) reshapes the tail while leaving the
argmax untouched -- measured agreement 1.000 with mean Hellinger 0.149 -- which is
exactly the change that perplexity, output diffing and top-k logprobs cannot see.
Weight-level changes move the argmax too (int8: agreement 0.756). A substitution or a
prompt-template mismatch destroys it.

Every boundary is a named constant, and the e2e battery asserts the verdict of each
known perturbation, so a recalibration is a visible edit that breaks tests until the
documentation above is updated with it.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Verdict", "classify", "classify_api", "THRESHOLDS"]

THRESHOLDS = {
    "unchanged_mean_h": 0.010,   # below this, attest unchanged
    "minor_mean_h": 0.060,       # bf16 at 0.036 and temp 1.05 at 0.044 land here
    "moderate_mean_h": 0.120,    # top-p 0.95 (0.149) and int8 (0.18) land above
    "tail_only_top1": 0.990,     # argmax agreement above this = serving-layer signature
    "tail_only_min_h": 0.080,    # ...provided the distribution moved this much
    "substitution_top1": 0.600,  # argmax agreement below this = different model/prompt
}

# exit codes, CI-friendly: 0 attested, 3 changed, 2 not comparable, 1 tool error
EXIT_SEALED, EXIT_ERROR, EXIT_INCOMPARABLE, EXIT_CHANGED = 0, 1, 2, 3


@dataclass
class Verdict:
    status: str        # "sealed" | "changed" | "incomparable"
    severity: str      # "none" | "minor" | "moderate" | "major"
    signature: str     # short mechanical hypothesis
    explanation: str   # one paragraph a reader can act on
    exit_code: int

    def as_dict(self):
        return {"status": self.status, "severity": self.severity,
                "signature": self.signature, "explanation": self.explanation,
                "exit_code": self.exit_code}


def classify(metrics: dict, t: dict = THRESHOLDS) -> Verdict:
    """Metrics from Snapshot.compare -> a Verdict. Pure function, no I/O."""
    if metrics.get("incomparable"):
        return Verdict(
            "incomparable", "none", "not the same measurement",
            f"The snapshots differ in {metrics['incomparable']} and cannot be compared. "
            "Re-snapshot both sides with the same probe set, sketch width and seed.",
            EXIT_INCOMPARABLE)

    mh = metrics["mean_hellinger"]
    t1 = metrics.get("top1_agreement")           # None when positions are misaligned
    structure = bool(metrics.get("structure_mismatch"))
    templates_differ = bool(metrics.get("templates_differ"))

    if not structure and mh < t["unchanged_mean_h"] and (t1 is None or t1 > 0.999):
        return Verdict(
            "sealed", "none", "behaviour unchanged",
            f"Mean Hellinger distance {mh:.4f} over {metrics['n_positions']} positions "
            "is below the attestation threshold and the most likely token agrees "
            "everywhere. The deployed model behaves as the reference.",
            EXIT_SEALED)

    if mh < t["minor_mean_h"]:
        severity = "minor"
    elif mh < t["moderate_mean_h"]:
        severity = "moderate"
    else:
        severity = "major"

    if structure:
        signature = "tokenisation or template-level change"
        explanation = (
            "The two snapshots do not even cover the same token positions on identical "
            "probe texts, which means the text reaching the model changed: a chat "
            "template, a system prompt, or a different tokeniser. This is the class of "
            "bug that is routinely misdiagnosed as a bad quantisation.")
    elif t1 is not None and t1 < t["substitution_top1"]:
        signature = ("prompt/template mismatch" if templates_differ
                     else "model substitution or template mismatch")
        explanation = (
            f"The most likely token agrees at only {t1:.1%} of positions. No precision "
            "or sampling change does this; either a different model is being served, or "
            "the prompt reaching it is not the prompt you validated.")
    elif t1 is not None and t1 >= t["tail_only_top1"] and mh >= t["tail_only_min_h"]:
        signature = "serving-layer sampling filter (tail-only)"
        explanation = (
            f"The distribution moved substantially (mean Hellinger {mh:.3f}) while the "
            f"most likely token agrees at {t1:.1%} of positions: the head of the "
            "distribution is intact and the tail is reshaped. That is the signature of "
            "a sampling filter such as top-p or min-p applied at serving time -- a "
            "change invisible to greedy output diffs, to perplexity, and largely to "
            "top-k logprobs.")
    elif mh < t["minor_mean_h"]:
        signature = "precision-level (dtype, kernels) or mild sampling parameter"
        explanation = (
            f"A small, broad shift (mean Hellinger {mh:.3f}) with the most likely token "
            f"agreeing at {t1:.1%} of positions. Consistent with reduced-precision "
            "arithmetic (bfloat16 measures 0.036 on GPT-2) or a small temperature "
            "change (1.05 measures 0.044). Decide against your own tolerance; this is "
            "below the level at which weight quantisation typically lands.")
    else:
        signature = "weight-level change (quantisation, fine-tune, or related model)"
        explanation = (
            f"The distribution moved (mean Hellinger {mh:.3f}) and the most likely "
            f"token changed at {1 - t1:.1%} of positions. The weights producing the "
            "distribution are not the reference weights: quantisation, a fine-tune, a "
            "different checkpoint revision -- or a closely related model substituted "
            "for the reference, which behavioural evidence alone cannot always "
            "separate from a heavily modified one (a distilled sibling measures here).")

    return Verdict("changed", severity, signature, explanation, EXIT_CHANGED)


def classify_api(s1, s2, bands, report=None) -> Verdict:
    """Sampled-endpoint statistics -> a Verdict. Pure function, no I/O.

    Weights mode measures a distance and compares it to thresholds. API mode cannot:
    what comes back is a finite sample, so the reference is not a threshold but the
    acceptance band the *unchanged* endpoint produces at this budget, calibrated where
    the reference distributions still existed and carried in the snapshot. A verdict
    here is "outside what the unchanged endpoint does", not "further than 0.06".

    S2 does NOT mean here what `top1_agreement` means in weights mode, and reading it
    as though it did gets the headline case backwards. There, agreement is between two
    argmaxes and a nucleus filter leaves it at 1.000. Here it is the *frequency* with
    which the endpoint emits the reference argmax, and cutting the tail concentrates
    the sample on the head, so a top-p filter pushes S2 sharply **up** (measured
    0.373 -> 0.807 on GPT-2 at top-p 0.95, in experiments/e2e_endpoint.py). The
    discriminator is therefore the *direction* of the excursion, not merely that there
    is one: more argmax than the unchanged endpoint means the head was concentrated,
    which weights do not do; less means the head itself moved.
    """
    lo1, hi1 = bands["s1"]
    lo2, hi2 = bands["s2"]
    out1, out2 = s1 < lo1 or s1 > hi1, s2 < lo2 or s2 > hi2
    n = bands.get("total")
    where = (f"S1 {s1:.4f} (band {lo1:.4f}-{hi1:.4f}), "
             f"S2 {s2:.4f} (band {lo2:.4f}-{hi2:.4f}) at {n} sampled tokens")

    if not (out1 or out2):
        return Verdict(
            "sealed", "none", "behaviour within the unchanged endpoint's band",
            f"Both sampled statistics fall inside the bands the unchanged reference "
            f"produces at this budget: {where}. Absence of evidence at this budget is "
            "not proof of identity -- the power of this test is the measured power at "
            "the budget you spent, not certainty.",
            EXIT_SEALED)

    severity = "major"
    if s2 > hi2:
        signature = "serving-layer sampling filter (head concentrated)"
        explanation = (
            f"The endpoint emits the reference's most likely token *more* often than "
            f"the unchanged reference does: {where}. Mass is being moved out of the "
            "tail and onto the head -- top-p, top-k, or a serving temperature below "
            "1. Changed weights scatter the head, they do not sharpen it, so this "
            "excursion is a serving parameter and not a different model. It is also "
            "the change that leaves greedy output, perplexity and top-k logprobs "
            "untouched.")
    elif s2 < lo2:
        signature = "head-level change (weights, template, or temperature above 1)"
        explanation = (
            f"The endpoint emits the reference's most likely token *less* often than "
            f"the unchanged reference does: {where}. The head has lost mass, which a "
            "different checkpoint, a quantisation, a prompt template mismatch or a "
            "serving temperature above 1 all produce. Sampled evidence alone does not "
            "separate those; a weights-mode snapshot would.")
        if not out1:
            severity = "minor"
    else:
        signature = "tail reshaped, head frequency intact"
        explanation = (
            f"The distributional statistic left its band while the rate at which the "
            f"reference argmax is emitted did not: {where}. Something moved in the "
            "tail without changing how often the head wins -- a narrow serving filter "
            "or a small precision change. Nothing that reads only the top token can "
            "see this.")

    if report and report.get("unresolved_fraction", 0) > 0.01:
        explanation += (
            f" Caution: {report['unresolved_fraction']:.1%} of returned tokens did not "
            f"resolve to a reference id (convention {report.get('convention')!r}), "
            "which is above the level this protocol was measured at. Prefer an "
            "endpoint that returns logprobs before acting on this verdict.")
    return Verdict("changed", severity, signature, explanation, EXIT_CHANGED)
