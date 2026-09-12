# Reference seals

A sealed reference, published so you can verify an endpoint without first building
one. Sealing needs the weights and a forward pass over the probes; verifying needs
neither, and these files are the half that travels.

## `gpt2-fp32-default-v1.seal.npz`

| | |
|---|---|
| model | `gpt2`, float32, snapshotted on CPU |
| probe set | `default-v1`, hash `0622744ec759498ab157735adc7e945c` |
| positions | 1500, sketch width D=256 |
| vocabulary | 50,257 |
| perplexity over the probes | 27.8302 |
| API bands | calibrated at a 5,000-token budget, both wire conventions |
| this probe set's exposure to the text convention | 0.0101 % of the sampled mass |
| size | 1.4 MB — 1 KB per position, which is the whole point |

Point it at anything claiming to serve GPT-2:

```bash
pip install servseal[api]          # tokeniser only: no torch, no weights

servseal verify gpt2-fp32-default-v1.seal.npz \
    --endpoint http://localhost:8000/v1 --served-model gpt2 \
    --budget 5000 --report attestation.html
echo $?      # 0 sealed | 3 changed
```

It will tell you whether the behaviour is still this reference, how far it moved, and
which layer moved it. `--report` writes a self-contained HTML attestation.

If you have the weights locally instead, compare snapshot to snapshot:

```bash
pip install servseal[model]
servseal snapshot /path/to/your-gpt2 -o candidate.seal.npz
servseal verify gpt2-fp32-default-v1.seal.npz candidate.seal.npz
```

## What it cannot do

**It is GPT-2's reference, not yours.** A seal is tied to one model, one tokeniser and
one probe set; comparison across any of those is refused rather than approximated. To
verify your own deployment, seal your own model — the point of publishing this one is
that you can see the tool work end to end before spending a forward pass.

**Serve it with `--return-tokens-as-token-ids` where you can.** vLLM's default fills
`logprobs.tokens` with decoded text, which costs the 0.0101 % above; token ids cost
nothing. The verdict reports which convention it detected either way.

**The bands were calibrated on this probe set at this budget.** `--budget` must match
one the seal carries, and a verdict is a statement about the traffic these 37 probe
texts represent. For your own domain, snapshot your own probes.

## Provenance

Made by `servseal snapshot gpt2 --api-budget 5000` at version 0.2.x, and used as the
reference in `experiments/e2e_endpoint.py`, whose committed output is in
`experiments/outputs/`. The method is
[Norm-Invariance in Vector-Symbolic Encodings of Probability Distributions](https://doi.org/10.5281/zenodo.22214968).
