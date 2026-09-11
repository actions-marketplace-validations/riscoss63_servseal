"""servseal: behavioural attestation for deployed language models.

    from servseal import Snapshot, classify
    from servseal.runner import snapshot_model         # needs torch

    ref = snapshot_model("gpt2")
    ref.save("gpt2.seal.npz")
    ...
    verdict = classify(Snapshot.load("gpt2.seal.npz").compare(snapshot_model(served)))
    print(verdict.status, "-", verdict.signature)

Against an endpoint you can only sample from, the reference must carry acceptance
bands, which can only be computed while its distributions exist:

    servseal snapshot MODEL -o ref.seal.npz --api-budget 5000     # needs the weights
    servseal verify ref.seal.npz --endpoint URL --served-model NAME   # needs neither
"""
__version__ = "0.2.0"

from .probes import load_probes, probe_id          # noqa: E402
from .snapshot import Snapshot                     # noqa: E402
from .verdict import Verdict, classify, classify_api   # noqa: E402

__all__ = ["Snapshot", "Verdict", "classify", "classify_api", "load_probes",
           "probe_id", "__version__"]
