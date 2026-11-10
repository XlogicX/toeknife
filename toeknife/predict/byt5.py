"""ByT5 as a predictor: byte-level, and shaped for exactly this problem.

The causal-LM adapter in `hf.py` has two flaws that are structural rather than
fixable. It works on text, so it has nothing to say about a binary stream. And
it costs a forward pass per candidate, because a byte-pair tokeniser has no way
to ask "what byte goes here" directly.

ByT5 has neither problem. Its vocabulary *is* the byte alphabet, 256 values plus
a handful of specials, so one forward pass yields a probability for every
possible byte at once rather than 256 passes. And span infilling is what it was
pretrained on: mask a run with a sentinel and the decoder produces what belongs
there. That is the hole-filling problem stated in the model's own terms, rather
than a causal model persuaded into it.

Bytes map to token ids with an offset of 3, past pad, eos and unk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .base import UNIFORM

BYTE_OFFSET = 3
SENTINEL = "<extra_id_0>"


class MissingBackend(RuntimeError):
    """torch and transformers are not installed."""


def _require():
    try:
        import torch                                          # noqa: F401
        from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa
    except ImportError as e:                                  # pragma: no cover
        raise MissingBackend(
            "this predictor needs torch and transformers: "
            "pip install 'toeknife[ml]'") from e


@dataclass
class Byt5Predictor:
    name: str = "google/byt5-small"
    order: int = 128
    lookahead: int = 48
    device: str = "cpu"
    _model: object = None
    _tok: object = None
    _sentinel: int = 0

    def load(self) -> "Byt5Predictor":
        _require()
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self._tok = AutoTokenizer.from_pretrained(self.name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.name)
        self._model.eval()
        self._model.to(self.device)
        self._sentinel = self._tok.convert_tokens_to_ids(SENTINEL)
        torch.set_grad_enabled(False)
        return self

    def distribution(self, left: bytes, right: bytes = b"") -> dict[int, float]:
        """log2 P(byte) for all 256 values, in a single forward pass.

        The hole is masked with the sentinel the model was pretrained to fill,
        and the first decoder step's logits are read off directly. No candidate
        enumeration: the vocabulary already is the answer space.
        """
        if self._model is None:
            self.load()
        import torch

        head = list(left[-self.order:])
        tail = list(right[:self.lookahead])
        ids = ([b + BYTE_OFFSET for b in head] + [self._sentinel]
               + [b + BYTE_OFFSET for b in tail] + [self._tok.eos_token_id])
        input_ids = torch.tensor([ids], device=self.device)
        # The decoder is primed with pad then the sentinel, so the next position
        # is the first byte of the masked span.
        decoder_ids = torch.tensor([[self._tok.pad_token_id, self._sentinel]],
                                   device=self.device)
        logits = self._model(input_ids=input_ids,
                             decoder_input_ids=decoder_ids).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        scale = 1.0 / math.log(2)
        return {b: logprobs[b + BYTE_OFFSET].item() * scale for b in range(256)}

    def rank(self, left: bytes, candidates, right: bytes = b"") -> list:
        dist = self.distribution(left, right)
        return sorted(((v, dist.get(v, UNIFORM)) for v in candidates),
                      key=lambda t: -t[1])

    def logprob(self, byte: int, left: bytes) -> float:
        return self.distribution(left).get(byte, UNIFORM)


def available() -> bool:
    try:
        _require()
        return True
    except MissingBackend:
        return False
