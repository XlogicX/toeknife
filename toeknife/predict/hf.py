"""A HuggingFace causal language model as a predictor.

Optional: `pip install toeknife[ml]`. Everything else in this package works
without it, and for most carved data the byte n-gram is the better tool anyway.
ziprec measured n-grams beating word-based reconstruction by three to five times
the speed at equal or better accuracy, and that was against a purpose-built
implementation.

**This is for text.** A general language model reads text; it has nothing useful
to say about an ELF header or a JPEG's entropy-coded data, and asking costs a
forward pass per candidate. The adapter declines rather than guessing when the
context does not decode as text, which sends the caller back to the byte model
that is actually suited to it.

Candidates are ranked by the log-probability of the whole sequence rather than
of the candidate byte alone. A byte-level BPE merges across the boundary, so a
candidate can change how the bytes *before* it tokenise; scoring only the
candidate's own tokens would miss that. The prefix is not constant across
candidates, so it stays in the comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .base import UNIFORM


class MissingBackend(RuntimeError):
    """torch and transformers are not installed."""


def _require():
    try:
        import torch                                        # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as e:                                # pragma: no cover
        raise MissingBackend(
            "this predictor needs torch and transformers: "
            "pip install 'toeknife[ml]'") from e


@dataclass
class HFPredictor:
    """Ranks byte candidates with a causal language model.

    `order` is how many bytes of left context to hand the model. More is better
    for the model and linearly more expensive, and past a sentence or so it
    stops paying for itself on the short contexts a carved fragment provides.
    """

    name: str = "distilgpt2"
    order: int = 192
    lookahead: int = 24
    device: str = "cpu"
    batch: int = 64
    _model: object = None
    _tok: object = None

    def load(self) -> "HFPredictor":
        _require()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tok = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForCausalLM.from_pretrained(self.name)
        self._model.eval()
        self._model.to(self.device)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        torch.set_grad_enabled(False)
        return self

    # ---- text handling ------------------------------------------------------

    @staticmethod
    def _as_text(data: bytes) -> str | None:
        """Bytes as text, or None when they are not text.

        Declining is the point: a model asked about bytes it cannot read will
        still return numbers, and those numbers are noise dressed as evidence.
        """
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            # A carved context routinely starts mid-character; trimming a few
            # leading bytes is legitimate, inventing replacements is not.
            for skip in range(1, min(4, len(data))):
                try:
                    got = data[skip:].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                # Trimming everything is not a decode.  A lone high byte would
                # otherwise come back as the empty string, which tokenises to
                # nothing and takes the forward pass down with it.
                if got:
                    return got
            return None

    def _sequence_logprob(self, texts: list[str]) -> list[float]:
        import torch
        tok = self._tok
        out: list[float] = []
        for start in range(0, len(texts), self.batch):
            chunk = texts[start:start + self.batch]
            enc = tok(chunk, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            logits = self._model(input_ids=ids, attention_mask=mask).logits
            logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            target = ids[:, 1:]
            picked = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)
            keep = mask[:, 1:].float()
            out.extend((picked * keep).sum(dim=1).tolist())
        return [v / math.log(2) for v in out]        # nats to bits

    # ---- the predictor interface -------------------------------------------

    def rank(self, left: bytes, candidates, right: bytes = b"") -> list:
        """Score each candidate byte in place.  Best first."""
        if self._model is None:
            self.load()
        head = self._as_text(left[-self.order:])
        if head is None:
            return []
        tail = self._as_text(right[:self.lookahead]) or ""
        texts, keep = [], []
        for value in candidates:
            body = self._as_text(bytes([value]))
            if not body:
                continue                    # not text, so this model cannot judge it
            text = head + body + tail
            if not text.strip("\x00"):
                continue
            texts.append(text)
            keep.append(value)
        if not texts:
            return []
        scores = self._sequence_logprob(texts)
        ranked = sorted(zip(keep, scores), key=lambda t: -t[1])
        return ranked

    def logprob(self, byte: int, left: bytes) -> float:
        """Protocol compatibility.  One forward pass per call, so prefer rank()."""
        got = self.rank(left, [byte])
        return got[0][1] if got else UNIFORM


def available() -> bool:
    try:
        _require()
        return True
    except MissingBackend:
        return False
