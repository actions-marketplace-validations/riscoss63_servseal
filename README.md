# servseal

[![tests](https://github.com/riscoss63/servseal/actions/workflows/ci.yml/badge.svg)](https://github.com/riscoss63/servseal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/servseal.svg)](https://pypi.org/project/servseal/)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Is the model you serve the model you validated?**

A deployment changes quietly: a provider turns on a sampling filter, an inference stack
quantises, a template edit ships, a checkpoint is swapped. Task benchmarks are noisy and
slow; checksums verify files, not behaviour. servseal snapshots what a model actually
*does* — the full next-token distribution over a fixed probe set, ~1 KB per position —
and later tells you whether the deployed behaviour is still the reference, **how much**
it moved, and **which layer** moved it.

```bash
pip install servseal[model]      # pulls sqsketch >= 0.3 from PyPI

servseal snapshot gpt2 -o reference.seal.npz
# ... deploy, quantise, migrate serving stacks, wait six months ...
servseal snapshot /path/to/served-model -o candidate.seal.npz
servseal verify reference.seal.npz candidate.seal.npz --report attestation.html
echo $?    # 0 sealed | 3 changed | 2 incomparable
```

## The change nothing else sees

Measured end to end through this tool on GPT-2 (1500 probe positions, 37 probe texts,
D=256 — every number below is from the committed run in `experiments/outputs/`):

| deployment event | mean Hellinger | top-1 agreement | perplexity | verdict |
|---|---:|---:|---:|---|
| unchanged (re-run) | **0.0000** | 1.000 | 27.83 → 27.83 | **SEALED** |
| serve: top-p 0.95 | **0.1537** | 1.000 | 27.83 → **27.83** | CHANGED/major — serving-layer |
| serve: temperature 1.05 | 0.0425 | 1.000 | 27.83 → 27.83 | CHANGED/minor |
| weights → bfloat16 | 0.0352 | 0.961 | 27.83 → 27.68 | CHANGED/minor — precision |
| weights → int8/tensor | 0.1684 | 0.773 | 27.83 → 32.03 | CHANGED/major — weight-level |
| wrong chat template | 0.9072 | 0.037 | 27.83 → 35.10 | CHANGED/major — template |
| distilgpt2 substituted | 0.2720 | 0.628 | 27.83 → 44.04 | CHANGED/major — weight-level |

Read the top-p row: the provider turned on nucleus sampling, the most likely token is
unchanged at **every** position, perplexity is unchanged to two decimals — greedy output
diffs, log-loss checks and eval suites see nothing — and the distribution moved by
0.15, which this tool measures exactly and labels correctly. A top-k logprob
fingerprint reports about half of it and does not converge with more memory (measured
in the sqsketch paper this tool builds on).

Read the template row the other way round: behaviour was destroyed (0.907, top-1
agreement 3.7%), while perplexity moved only from 27.8 to 35.1 — the class of bug
routinely misdiagnosed as "the quantisation made it worse". The signature separates
them: template bugs annihilate top-1 agreement; quantisation dents it; serving filters
leave it at 100%.

The same battery runs on a second architecture (pythia-160m): unchanged → SEALED at
0.0000, top-p → 0.1520, CHANGED/serving-layer. Comparing a pythia snapshot to a GPT-2
snapshot is refused (`INCOMPARABLE`, exit 2) — different tokenizer, not the same
measurement.

And on a **current architecture** — Qwen2.5-0.5B, vocabulary 151,936, a far sharper
model (median effective support ≈ 7) where a tail-blind fingerprint has the least to
miss (`experiments/e2e_qwen.py`, committed run in `experiments/outputs/`):

| deployment event | mean Hellinger | top-1 agreement | perplexity | verdict |
|---|---:|---:|---:|---|
| unchanged (re-run) | **0.0000** | 1.000 | 12.96 → 12.96 | **SEALED** |
| weights → bfloat16 (native: a no-op) | **0.0000** | 1.000 | 12.96 → 12.96 | **SEALED** — the true negative |
| serve: top-p 0.95 | 0.1446 | 1.000 | 12.96 → **12.96** | CHANGED/major — serving-layer |
| serve: temperature 1.05 | 0.0351 | 1.000 | 12.96 → 12.96 | CHANGED/minor |
| weights → int8/tensor | 0.1244 | 0.857 | → 14.80 | CHANGED/major — weight-level |
| ChatML template on a base model | 0.9459 | 0.016 | → 28.26 | CHANGED/major — template, certified KL ≥ 1.56 |
| **Qwen3-0.6B silently served** | 0.3546 | 0.622 | → 19.33 | CHANGED/major |

Two rows matter beyond the pattern repeating. The bfloat16 line is the true-negative
test a monitoring tool must pass to be trusted with a pager: Qwen's weights are
natively bf16, the "change" is an exact no-op, and the verdict is SEALED at exactly
zero. And the last line is the silent-upgrade scenario providers actually perform —
the next model generation behind the same tokenizer, caught at 0.35.

## API mode: endpoints you can only sample from

Against a black-box API the full distribution is unavailable; the endpoint returns
sampled tokens, already shaped by the provider's serving parameters — which is exactly
the object under test. Protocol: one `max_tokens=1, temperature=1` completion per probe
position, N times. Detection is two-sided and calibrated on the unchanged endpoint;
power measured on real GPT-2 distributions (`experiments/sampling_power.py`):

| sampled tokens | per position | false positives | detects top-p 0.95 | detects temp 1.05 |
|---:|---:|---:|---:|---:|
| 1,500 | 1.0 | 0.04 | 0.83 | 0.88 |
| **5,000** | **3.3** | **0.05** | **1.00** | **1.00** |
| 15,000 | 10.0 | 0.02 | 1.00 | 1.00 |

**Five thousand single-token completions detect both changes with power 1.00 at a 5%
false-positive budget** — on a commercial API, a few dollars of traffic. Two designs
failed before this one and are documented in `servseal/sampler.py`: a one-sided "BC
must drop" rule (power 0.10 — cutting the tail *concentrates* samples on the head and
pushes the statistic up) and a pooled corpus-aggregate statistic (power 0.03 — one
position's tail is another's head; pooling erases the evidence). The shipped statistic
is per-position, against the per-position sketches the snapshot already stores, and the
power sweep asserts bit-equality with the product code path before measuring.

### Running it

The bands can only be computed while the reference distributions exist, so they are
made at snapshot time and travel inside the `.seal.npz`. A reference sealed without
`--api-budget` is refused for API mode rather than given an invented threshold.

Sealing runs the model, so it needs the weights; verifying an endpoint never does, and
the two halves can live on different machines years apart.

```bash
# once, wherever the weights are
pip install servseal[model]
servseal snapshot Qwen/Qwen2.5-0.5B -o reference.seal.npz --api-budget 5000

# from anywhere, for as long as you keep the seal: tokeniser only, no torch
pip install servseal[api]
servseal verify reference.seal.npz \
    --endpoint https://your-provider/v1 --served-model qwen2.5-0.5b \
    --budget 5000 --report attestation.html
echo $?      # 0 sealed | 3 changed
```

`SERVSEAL_API_KEY` carries the bearer token. The client sends `max_tokens=1` and
`temperature=1`, and **never sends `top_p`** — sending `top_p=1.0` would switch off a
server-side nucleus filter, which is the single change this mode exists to catch.

### Measured against vLLM

Not against a mock: a real vLLM serving GPT-2 in float32, reached through the shipped
CLI over HTTP, with the reference snapshotted separately on CPU
(`experiments/kaggle/vllm_probe.py`, committed attestations in
`experiments/outputs/`). 5,000 sampled tokens over 1,500 probe positions in 48
requests, batched 32 prompts at a time with no retries.

| deployment served by vLLM | S1 (band 0.4823–0.4975) | S2 (band 0.3568–0.3815) | verdict |
|---|---:|---:|---|
| gpt2 float32, unchanged | **0.4903** | **0.3681** | **SEALED** |
| gpt2, provider `top_p 0.95` | 0.5091 | **0.3840** ↑ | CHANGED — serving-layer filter |
| distilgpt2 silently substituted | 0.4308 | **0.3099** ↓ | CHANGED — head-level |

The first row is the true negative that makes the rest worth reading: a float32
reference computed on a CPU and a float32 deployment computed on a T4 land inside the
band, so the verdict is about the deployment and not about the hardware it was sealed
on.

The two changed rows differ in the *direction* of the S2 excursion, and that is the
whole signature. S2 is not agreement between two argmaxes — it is how often the
endpoint emits the reference's most likely token. A nucleus filter cuts the tail,
which concentrates the sample on the head and pushes S2 **up**; changed weights
scatter the head and push it **down**. Reading the excursion as a magnitude rather
than a direction names the nucleus filter as a weight change, which is the one error
this tool cannot afford, so `tests/test_wire_and_endpoint.py` pins both directions.

Note how thin the top-p margin is: 0.3840 against a bound of 0.3815. Five thousand
tokens is the *minimum* for that filter at this vocabulary, not a comfortable budget.
Snapshot a second, larger budget if you intend to act on a single run.

### Getting the token back: ask for ids

The endpoint returns text; the statistics need ids in the reference vocabulary. Three
conventions are detected on the first response rather than configured, and one of them
costs nothing:

| what the endpoint returns | recovery | cost |
|---|---|---|
| `token_id:1234` (`vllm serve --return-tokens-as-token-ids`) | exact | none |
| the raw token piece | exact — measured collision-free on GPT-2, pythia-160m, Qwen2.5 | none |
| decoded text (**vLLM's default**) | re-encode; drop what does not resolve to one id | 0.01 % of the sampled mass on English probes |

**Serve with `--return-tokens-as-token-ids` where you can.** vLLM populates
`logprobs.tokens` from `logprob.decoded_token`, which is the *decoded text*, not the
piece — so the default path is the third row, and the run above measured it at 0.04 %
of returned tokens unresolved, two out of five thousand.

That 0.01 % is a property of the *probe set*, not of the tokeniser: the ambiguity
lives in byte fragments of multi-byte characters, runs of whitespace, and non-Latin
scripts (39.9 % / 52.2 % of the ambiguous entries on Qwen2.5). So it is measured on
your probes at snapshot time and stored in the seal, and `snapshot` prints it:

```
api_text   corrupted mass 0.0102 %  worst position 2.142 %   OK
```

On code or a non-Latin script it will be larger, the line says so, and the answer is
an endpoint that returns ids. The full price list — vocabulary collisions, mass
weighting, the shift in (S1, S2) against the band width, and the false-positive and
power checks that follow — is `experiments/token_recovery.py`, whose committed output
covers three tokenisers.

## What a verdict gives you

```
verdict    CHANGED  severity=major
signature  serving-layer sampling filter (tail-only)
hellinger  mean=0.1537  max=0.2513  floor=0.0884
top1       agreement=1.0000
kl_bound   mean per-position KL >= 0.0000  (certified; 0 certifies nothing, not absence of change)
report     attestation.html
```

`--report` writes a self-contained HTML attestation — verdict banner, the numbers, a
per-position histogram, both snapshot identities and the probe hash — suitable for a
ticket, a release artifact, or a compliance file.

## Use it as a CI gate

The repository doubles as a GitHub Action: seal the reference once, commit the snapshot,
and every scheduled run fails loudly the day the deployment stops behaving like it.

```yaml
name: model-fidelity
on:
  schedule: [{cron: "0 6 * * 1"}]      # every Monday
  workflow_dispatch:

jobs:
  attest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - uses: riscoss63/servseal@v0.1.2
        with:
          reference: seals/reference.seal.npz
          model: ./path-or-hf-id-of-what-you-serve
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: attestation, path: attestation.html}
```

Exit codes gate the job: `0` sealed, `3` changed (job fails, attestation attached),
`2` not comparable (a configuration error, also failing).

## What it will not do

- **Compare across tokenizers.** A fingerprint lives in one vocabulary; pythia vs GPT-2
  is refused, not approximated. (Measured and closed as a negative result in the
  underlying paper.)
- **Name the culprit with certainty.** Signatures are calibrated hypotheses, not
  proofs: a distilled sibling substituted for the reference lands in the same band as
  an aggressive quantisation (both are "the weights are not the reference weights"),
  and behavioural evidence alone cannot always split them.
- **Certify small changes.** The certified-KL line is a one-sided bound that only bites
  on gross changes at D=256: `0` certifies nothing and must never be read as "nothing
  changed". The Hellinger measurement, not the certificate, is the detector.
- **Judge quality.** servseal tells you the behaviour moved, not whether it got worse.
  A fine-tune you shipped on purpose is CHANGED/major — as it should be.
- **See what the probes never touch.** Coverage is the probe set's; snapshot your own
  traffic domain too (`--probes yourfile.txt`). Snapshots refuse comparison across
  probe sets by hash.
- **Test a model whose weights you never had.** API mode calibrates its bands by
  simulating the unchanged endpoint, which needs the reference distributions and
  therefore the weights — available exactly once, when the reference is sealed. That
  makes this a tool for open-weights deployments: your own serving stack, or a
  provider you buy Llama/Qwen/Mistral inference from. A model that was never yours to
  snapshot cannot be sealed.
- **Weights-mode determinism is CPU-grade.** Reference snapshots here are computed in
  float32 on CPU, where re-running the same model reproduces distance exactly 0.0000.
  GPU inference can be nondeterministic; snapshot on CPU for the reference of record.

## How it works

Each position's full softmax (vocabulary-sized, tail included) is compressed to D=256
numbers by a square-root sketch whose inner product estimates the Bhattacharyya
coefficient of the complete distributions, with error `sqrt(2/D)` independent of
vocabulary size — the tail is exactly where serving filters and quantisation act, and
where top-k logprobs are blind by construction. The estimator, its error bars, the
one-sided KL certificate and the negative results (cross-tokenizer, top-k saturation)
are established in the sqsketch paper: *Norm-Invariance in Vector-Symbolic Encodings of
Probability Distributions* (DOI
[10.5281/zenodo.22214968](https://doi.org/10.5281/zenodo.22214968), code
[riscoss63/sqsketch](https://github.com/riscoss63/sqsketch)).

## Audit it

Everything above is one command each, on models small enough for a laptop CPU:

```bash
pip install -e .[model,dev]
pytest                                    # 71 tests: units, CLI, HTTP transport
cd experiments
python e2e_real_models.py                 # the verdict battery, ~12 min CPU
python sampling_power.py                  # the API-mode power table, ~1 min
python token_recovery.py                  # what recovering a token costs, ~2 min
python e2e_endpoint.py                    # API mode end to end over HTTP, ~2 min
```

The battery *asserts* every verdict against its ground truth and exits non-zero on any
miss; the committed outputs in `experiments/outputs/` are what the tables above quote.
Verdict thresholds live in `servseal/verdict.py` with the calibration documented
inline; changing them breaks tests until the documentation moves with them.

## Commercial support

The tool is MIT and stays that way. What is paid for is work and vigilance, not code:

- **Deployment-fidelity audit** (fixed price): did your quantisation, serving-stack
  migration or provider switch change the model's behaviour, where, and by how much? You
  get the sealed reference, the verification runs, and a signed attestation report.
- **Continuous attestation**: your endpoints re-verified on a schedule, with alerting
  and a monthly report.

Open an issue with the `audit` label, or reach the maintainer via the contact on their
[GitHub profile](https://github.com/riscoss63).

## Licence

MIT.
