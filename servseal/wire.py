"""Between the wire and the statistic: putting a sampled token back on its id.

Weights mode indexes the full softmax by token id. An endpoint returns *text*, so API
mode needs a step weights mode never did -- recovering the reference vocabulary's id
from what came back. `experiments/token_recovery.py` prices that step; this module is
what it priced, and the two must not drift, so the experiment imports from here.

Two conventions, because endpoints differ and the cost does too:

    piece   what `logprobs` carries: the raw token, byte markers and all. Measured
            collision-free on gpt2, pythia-160m and qwen2.5-0.5b -- every id has its
            own piece, so recovery is a dictionary lookup and is exact. Ask for this.
    text    what a bare completion field carries: `decode([id])`, which is not
            injective. Measured on an English probe set it costs 0.01 % of the sampled
            mass, touches no argmax, and moves the statistics ~1000x less than the
            width of the acceptance bands -- but that number is a property of the
            *probe set*, not of the tokeniser: the ambiguity lives in byte fragments
            of multi-byte characters, runs of whitespace and non-ASCII scripts. On
            code or a non-Latin script it is larger, which is why `corrupted_mass`
            below is measured at snapshot time, when P is still in hand, and stored.

The resolution rule for the text convention is deliberately tokeniser-only. At verify
time the snapshot holds sketches and argmax ids, never the reference distributions, so
"resolve the collision in favour of whichever candidate the reference thinks likelier"
is not an option the deployed tool has. Re-encode, take the id if that yields exactly
one, drop the sample otherwise: dropping costs budget, guessing would cost accuracy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["WireMap", "corrupted_mass", "position_prefixes"]

_CHUNK = 4096
_POS_CHUNK = 32


class WireMap:
    """Both wire conventions for one tokeniser, and the rule that inverts them."""

    def __init__(self, tokenizer, vocab_size):
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        n = min(self.vocab_size, len(tokenizer))
        ids = list(range(n))

        pieces = tokenizer.convert_ids_to_tokens(ids)
        self.piece_index = {}
        self.piece_collisions = 0
        for i, p in enumerate(pieces):
            if p in self.piece_index:
                self.piece_collisions += 1
            else:
                self.piece_index[p] = i

        decoded = []
        for s in range(0, n, _CHUNK):
            decoded.extend(tokenizer.batch_decode([[i] for i in ids[s:s + _CHUNK]]))
        self.text_index = {}
        enc = []
        for s in range(0, n, _CHUNK):
            enc.extend(tokenizer(decoded[s:s + _CHUNK],
                                 add_special_tokens=False)["input_ids"])
        self.remap = np.full(self.vocab_size, -1, dtype=np.int64)
        for i, e in enumerate(enc):
            if len(e) == 1:
                self.remap[i] = e[0]
        # every id sharing a decoded form re-encodes identically, so the first wins
        for i, d in enumerate(decoded):
            self.text_index.setdefault(d, int(self.remap[i]))

        arange = np.arange(self.vocab_size)
        self.exact = int(np.sum(self.remap == arange))
        self.wrong = int(np.sum((self.remap >= 0) & (self.remap != arange)))
        self.split = int(np.sum(self.remap < 0)) - (self.vocab_size - n)
        self.padded = self.vocab_size - n

    # ------------------------------------------------------------------- inverting

    def from_piece(self, piece):
        """Exact inverse where the endpoint reports raw tokens. -1 if unknown."""
        return self.piece_index.get(piece, -1)

    def from_text(self, text):
        """Tokeniser-only inverse of `decode([id])`. -1 when it does not resolve.

        A string no id decodes to is still re-encoded rather than refused: providers
        normalise (NFC, stripped controls), so the form on the wire is not always one
        the table was built from, and the rule -- one token in, one id out -- is the
        same either way.
        """
        got = self.text_index.get(text)
        if got is not None:
            return int(got)
        enc = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        got = int(enc[0]) if len(enc) == 1 else -1
        self.text_index[text] = got
        return got

    def apply(self, tok_ids):
        """Round-trip a stream of *reference* ids, as the text convention would.

        Only used to price the convention (calibration, and the experiment): the live
        path resolves strings, never ids. Returns the mapped ids with the
        unresolvable ones marked -1.
        """
        return self.remap[np.asarray(tok_ids)]

    def stats(self):
        v = self.vocab_size
        return {"vocab_size": v, "piece_collisions": self.piece_collisions,
                "exact": self.exact, "wrong": self.wrong, "split": self.split,
                "padded": self.padded,
                "exact_fraction": round(self.exact / v, 6)}

    def __repr__(self):
        return (f"WireMap(vocab={self.vocab_size}, piece_collisions="
                f"{self.piece_collisions}, text_exact={self.exact / self.vocab_size:.4f})")


def corrupted_mass(P, wiremap):
    """Share of the reference mass the text convention would move or drop.

    Measured at snapshot time because it needs P, and reported per probe set because
    that is what it is a property of. `wrong` is the dangerous half -- the statistic
    would be computed against a token the endpoint never emitted; `split` is merely
    dropped and costs budget only.
    """
    arange = np.arange(wiremap.vocab_size)
    wrong = np.flatnonzero((wiremap.remap >= 0) & (wiremap.remap != arange))
    split = np.flatnonzero(wiremap.remap < 0)
    n_pos = P.shape[0]
    tot_w = tot_s = 0.0
    worst_w = 0.0
    for s in range(0, n_pos, _POS_CHUNK):
        blk = np.asarray(P[s:s + _POS_CHUNK], dtype=np.float64)
        blk /= blk.sum(1, keepdims=True)
        pw = blk[:, wrong].sum(1) if wrong.size else np.zeros(len(blk))
        tot_w += float(pw.sum())
        worst_w = max(worst_w, float(pw.max()))
        if split.size:
            tot_s += float(blk[:, split].sum())
    return {"wrong_mass": tot_w / n_pos, "split_mass": tot_s / n_pos,
            "worst_position_wrong_mass": worst_w}


def position_prefixes(tokenizer, texts, *, max_positions=1500, max_length=96,
                      template=None):
    """The token-id prefix the endpoint must be asked for, one per snapshot position.

    Mirrors `runner.softmax_over_probes` exactly: same tokenisation, same truncation,
    same per-text loop and the same break *before* a text rather than inside it, so
    position i here is the position i that was sketched. `tests/test_wire.py` asserts
    that agreement against the runner rather than trusting this comment.
    """
    out = []
    for t in texts:
        if len(out) >= max_positions:
            break
        text = template.replace("{text}", t) if template else t
        ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        out.extend(ids[:k + 1] for k in range(len(ids)))
    return out[:max_positions]
