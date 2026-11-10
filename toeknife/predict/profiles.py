"""Working out what the data is, and assembling a predictor to suit.

Detection runs on the bytes already recovered, which is the one sample of the
plaintext that is certainly genuine. It is deliberately conservative: guessing
"UTF-8 text" wrongly turns on a constraint that rules out the right answer, and
a hard constraint applied to the wrong kind of data is worse than no constraint
at all. So each profile has to clear a margin rather than merely win.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import ASCII_TEXT, Alphabet, Utf8
from .mixture import Mixture, TablePrior, Weighted

ELF_MAGIC = b"\x7fELF"
PE_MAGIC = b"MZ"
MACHO = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"}


@dataclass
class Profile:
    name: str
    why: str
    text_like: bool = False
    constraints: list = field(default_factory=list)
    corpus_hint: str = ""


def _known_bytes(tape, limit: int = 400000) -> bytes:
    out = bytearray()
    for p in range(min(len(tape), limit)):
        v = tape.value_at(p)
        if v is not None:
            out.append(v)
    return bytes(out)


def _valid_utf8_fraction(sample: bytes) -> float:
    """How much of the sample decodes as UTF-8, ignoring a broken tail.

    Recovered bytes are not contiguous, so a strict decode of the whole thing
    fails on the first splice.  Chunking measures the property that matters:
    whether the data is UTF-8 shaped, not whether this particular sample is a
    valid document.
    """
    if not sample:
        return 0.0
    good = total = 0
    for i in range(0, len(sample), 256):
        chunk = sample[i:i + 256]
        total += 1
        # A chunk can be cut mid-character at either end, and neither is
        # evidence about the data.  Trimming both is the difference between
        # this reporting 100% and reporting roughly half.
        head = 0
        while head < 4 and head < len(chunk) and 0x80 <= chunk[head] < 0xC0:
            head += 1
        body = chunk[head:]
        try:
            body.decode("utf-8")
            good += 1
        except UnicodeDecodeError as e:
            if e.start >= len(body) - 4:
                good += 1
    return good / total


def detect(tape, trace=None) -> Profile:
    sample = _known_bytes(tape)
    if len(sample) < 32:
        return Profile("unknown", "too few bytes recovered to tell")

    printable = sum(1 for b in sample if b in ASCII_TEXT) / len(sample)
    high = sum(1 for b in sample if b >= 0x80) / len(sample)
    nul = sample.count(0) / len(sample)

    constraints = []
    if trace is not None:
        alphabet = Alphabet.from_trace(trace)
        if alphabet is not None:
            constraints.append(alphabet)

    if sample[:4] == ELF_MAGIC or b"\x7fELF" in sample[:4096]:
        return Profile("elf", "ELF magic in the recovered bytes",
                       constraints=constraints, corpus_hint="/usr/bin")
    if sample[:2] == PE_MAGIC and b"PE\x00\x00" in sample[:4096]:
        return Profile("pe", "MZ and PE signatures", constraints=constraints)
    if sample[:4] in MACHO:
        return Profile("macho", "Mach-O magic", constraints=constraints)

    if printable > 0.92 and high < 0.02:
        return Profile("ascii-text", f"{printable:.0%} of recovered bytes are "
                       f"printable ASCII", text_like=True,
                       constraints=constraints)

    if high > 0.02 and printable + high > 0.9:
        share = _valid_utf8_fraction(sample)
        if share > 0.98:
            return Profile(
                "utf-8", f"{share:.0%} of sampled chunks decode as UTF-8",
                text_like=True, constraints=constraints + [Utf8()])

    if nul > 0.3:
        return Profile("sparse-binary", f"{nul:.0%} of recovered bytes are zero",
                       constraints=constraints)
    return Profile("binary", f"{printable:.0%} printable, {high:.0%} high-bit",
                   constraints=constraints)


def sanity_check(model, tape, sample: int = 60000, floor: int = 32) -> tuple[float, str]:
    """Is this model any use on this data, or is it actively harmful?

    A model trained on the wrong kind of data does not degrade to ignorance, it
    degrades past it. Measured on held-out samples: an English-trained order-4
    model scores -3.9 bits per byte on English and -17.2 on ELF, against -8.0
    for knowing nothing at all. It is not merely unhelpful there, it is
    confidently wrong, and confidence is what the solver's threshold reads.

    Returns (mean log2 probability per byte, a verdict).
    """
    known = _known_bytes(tape, sample)
    if len(known) < floor:
        return 0.0, "too few recovered bytes to judge the model against"
    score = model.score(known)
    if score < -8.0:
        return score, (f"this model scores {score:.1f} bits per byte on the "
                       f"recovered data, worse than the -8.0 of knowing nothing. "
                       f"It is trained on the wrong kind of data and will be "
                       f"confidently wrong. Use a corpus that resembles this "
                       f"stream, or none at all")
    if score < -6.5:
        return score, (f"this model scores {score:.1f} bits per byte on the "
                       f"recovered data, barely better than the -8.0 of knowing "
                       f"nothing; a closer corpus would help a great deal")
    return score, ""


def build(session, corpus_model=None, self_train: bool = True,
          profile: Profile | None = None,
          order: int = 4) -> tuple[Mixture, Profile]:
    """Assemble the predictor for this session.

    Three sources, none sufficient alone.  A corpus model knows the language but
    not this file.  A self-trained model knows this file but has seen very
    little.  A dynamic block's table knows which bytes occur and how often but
    nothing about order.  The weights lean on whichever has the most behind it.
    """
    from .context import ContextModel
    from .solver import train_from_session

    prof = profile or detect(session.tape, session.trace)
    parts = []

    if corpus_model is not None:
        parts.append(Weighted(corpus_model, 1.0, "corpus"))

    if self_train:
        own = train_from_session(session, order=order)
        if own.trained_bytes >= 256:
            # Weighted by how much it actually saw: a model built from a few
            # hundred bytes is a rumour, not evidence.
            weight = min(1.0, own.trained_bytes / 20000)
            parts.append(Weighted(own, weight * (0.6 if corpus_model else 1.0),
                                  "self-trained"))

    prior = TablePrior.from_trace(session.trace)
    if prior is not None:
        parts.append(Weighted(prior, 0.25, "table prior"))

    if not parts:
        parts.append(Weighted(ContextModel(order=4), 1.0, "empty"))
    return Mixture(parts, prof.constraints), prof
