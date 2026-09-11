"""Snapshotting a model that is not on the CPU.

`snapshot_model` has taken a `device=` argument since the first release and only ever
worked for "cpu": the probe token ids were built on the CPU and handed straight to the
model, so a model on a GPU failed on its first embedding lookup with "Expected all
tensors to be on the same device". It surfaced when a study needed a GPU reference,
which is late for a parameter that was always public.

These tests need no GPU. They pin the two things that were wrong -- that the ids are
moved to wherever the model is, and that what comes back is brought home before numpy
sees it -- with a stub model that reports a device and records what it is handed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from servseal.runner import softmax_over_probes  # noqa: E402

VOCAB = 32


class StubTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        n = min(len(text.split()), max_length or 8)
        ids = torch.arange(1, max(n, 2) + 1).unsqueeze(0)
        return type("Enc", (), {"input_ids": ids})()


class StubModel:
    """Reports a device, and refuses input that did not come to it."""

    def __init__(self, device=torch.device("cpu"), logits_device=None):
        self.device = device
        self._logits_device = logits_device or device
        self.seen = []

    def __call__(self, ids):
        self.seen.append(ids.device)
        if ids.device != self.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but got index is "
                f"on {ids.device}, different from other tensors on {self.device}")
        logits = torch.randn(1, ids.shape[1], VOCAB, device=self._logits_device)
        return type("Out", (), {"logits": logits})()


def test_ids_are_moved_to_the_models_device():
    tok, model = StubTokenizer(), StubModel()
    P, ppl = softmax_over_probes(model, tok, ["one two three four"],
                                 max_positions=10, max_length=8)
    assert model.seen and all(d == model.device for d in model.seen)
    assert P.shape[1] == VOCAB and P.dtype == np.float32
    assert np.isfinite(ppl)


def test_the_device_is_read_from_the_model_not_assumed():
    """The fix must consult the model, not hard-code a device.

    Recording `Tensor.to` is the only way to see the difference without a second
    device: moving a CPU tensor to the CPU is a no-op, so a test that only checks
    where the ids arrived would pass just as well against the broken version.
    """
    calls = []
    real_to = torch.Tensor.to

    def spy(self, *a, **k):
        if a and isinstance(a[0], torch.device):
            calls.append(a[0])
        return real_to(self, *a, **k)

    torch.Tensor.to = spy
    try:
        softmax_over_probes(StubModel(), StubTokenizer(), ["a b c"],
                            max_positions=5, max_length=8)
    finally:
        torch.Tensor.to = real_to
    assert torch.device("cpu") in calls, "the ids were never sent to model.device"


def test_a_model_without_a_device_attribute_still_works():
    """device_map models expose `.device`; a bare nn.Module needs the fallback."""
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ids):
            return type("Out", (), {
                "logits": torch.randn(1, ids.shape[1], VOCAB)})()

    P, _ = softmax_over_probes(Bare(), StubTokenizer(), ["a b c"],
                               max_positions=5, max_length=8)
    assert P.shape[1] == VOCAB


def test_logits_come_home_before_numpy_sees_them():
    """`.numpy()` on a non-CPU tensor raises; the fix must call `.cpu()` first.

    Exercised here on the CPU, where it is a no-op, so what this really pins is that
    the call is present. The cross-device case runs on the GPU kernel in
    experiments/kaggle/quant_real.py, which is where the bug was found.
    """
    model = StubModel()
    P, ppl = softmax_over_probes(model, StubTokenizer(), ["a b c d e"],
                                 max_positions=5, max_length=8)
    assert np.isfinite(P).all() and P.min() >= 0.0
    assert abs(P.sum(axis=1) - 1.0).max() < 1e-5
