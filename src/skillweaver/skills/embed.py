"""A local Embedder for retrieval, and the factory that decides whether there is one.

**OFF by default, because it was measured and did not pay. Do not "finish" it by
turning it on.** all-MiniLM-L6-v2 through onnxruntime, already a dependency, so no new
package and no API key; ~14 ms for five short texts, ~0.1 s to load, ~90 MB on disk.

``scripts/bench_retrieval.py``, 48 requests over both libraries this repo carries:
recall improved (top-1 18/24 -> 21/24), RUNNABLE did not move at all (3/24 and 9/24
both ways), and precision got WORSE (irrelevant requests answered 1/6 -> 5/6, because a
cosine is almost never zero). No cut-off separates those populations - the best
irrelevant request scores 0.198 against correct paraphrases at 0.191 and 0.198 - so
there is nothing to tune. Re-measured 2026-09-20 after binding and the content gate
both moved: every number unchanged.

WHY: a candidate must still BIND and then account for the request, and both count
WORDS, so a request worded differently fails them for the reason it ranked badly.
WHAT WOULD HAVE TO CHANGE: those two gates, not this module. Then re-run the bench and
read ``runnable``.

Reachable ONLY through ``SKILLWEAVER_EMBEDDER``: weights on disk are deliberately not an
implicit yes. They are not committed; until ``scripts/fetch_embedder.py`` has run,
:func:`load_embedder` returns ``None`` WITH A REASON that reaches every candidate's
``why``. Tokenization is here rather than from ``tokenizers`` because the dependency
list is shared surface, and ``_`` is BERT punctuation, which ``snake_case`` skill names
depend on.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from skillweaver.config import Settings, settings
from skillweaver.errors import ProviderError
from skillweaver.logging_ import get_logger

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MEASURED_TIMINGS",
    "MODEL_FILE",
    "VOCAB_FILE",
    "OnnxTextEmbedder",
    "WordPiece",
    "embedder_for",
    "load_embedder",
]

log = get_logger(__name__)

MODEL_FILE = "model.onnx"
"""The two files ``scripts/fetch_embedder.py`` writes into
:attr:`~skillweaver.config.Settings.embedder_dir`."""

VOCAB_FILE = "vocab.txt"

DEFAULT_MAX_TOKENS = 256
"""Longest token sequence one text is cut to, the ``[CLS]``/``[SEP]`` pair included.

all-MiniLM-L6-v2 was trained with a 256-token window, so this is the model's own limit
rather than a budget. It also bounds the one native call here - attention is quadratic
in the sequence length. Skill text on the shipped Wikipedia library runs 60 to 130
tokens, so nothing real is truncated today."""

DEFAULT_THREADS = 1
"""onnxruntime threads for the embedding session.

Set EXPLICITLY, for the reason written up in :mod:`skillweaver.perception.ocr`: left
alone, onnxruntime sizes its pool from the core count and its workers spin. One is not a
compromise - the batch is a handful of short sentences and measured 14 ms for five of
them single-threaded."""

MEASURED_TIMINGS = {
    "session_load_ms": 92.0,
    "five_short_texts_ms": 14.1,
    "model_bytes": 90_405_214,
}
"""Measured 2026-09-19 on the demo laptop (macOS arm64, onnxruntime 1.30, 1 thread).
Quoted so the hosted-versus-local decision above can be re-checked rather than
re-argued: a whole library re-embeds in the time one round trip spends on its TLS
handshake."""

_UNK = "[UNK]"
_CLS = "[CLS]"
_SEP = "[SEP]"


# --------------------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------------------


def _is_punctuation(char: str) -> bool:
    """BERT's own definition: the ASCII symbol ranges plus every Unicode ``P``
    category. Wider than "not alphanumeric" on purpose - ``_`` is punctuation here,
    which is what splits ``add_dish_with_option`` into words the model has seen."""
    point = ord(char)
    if 33 <= point <= 47 or 58 <= point <= 64 or 91 <= point <= 96 or 123 <= point <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_cjk(char: str) -> bool:
    """Whether a character is in one of the CJK blocks BERT surrounds with spaces so
    each one becomes its own token."""
    point = ord(char)
    return (
        0x4E00 <= point <= 0x9FFF
        or 0x3400 <= point <= 0x4DBF
        or 0x20000 <= point <= 0x2A6DF
        or 0x2A700 <= point <= 0x2B73F
        or 0x2B740 <= point <= 0x2B81F
        or 0x2B820 <= point <= 0x2CEAF
        or 0xF900 <= point <= 0xFAFF
        or 0x2F800 <= point <= 0x2FA1F
    )


class WordPiece:
    """The BERT-uncased tokenizer, over a ``vocab.txt`` of one token per line.

    Faithful to the reference implementation in the ways that change token ids:
    lower-casing, NFD accent stripping, control-character removal, punctuation and CJK
    splitting, then greedy longest-match-first WordPiece with ``##`` continuations.

    Raises:
        ProviderError: if the vocabulary is missing a special token the encoder needs.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        for special in (_UNK, _CLS, _SEP):
            if special not in vocab:
                raise ProviderError(f"embedder vocabulary has no {special} token")
        self.vocab = vocab
        self.unk_id = vocab[_UNK]
        self.cls_id = vocab[_CLS]
        self.sep_id = vocab[_SEP]

    @classmethod
    def from_file(cls, path: Path) -> WordPiece:
        """Read ``vocab.txt``. Line ``n`` is token id ``n``.

        Raises:
            ProviderError: if the file cannot be read or is missing a special token.
        """
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except OSError as exc:
            raise ProviderError(f"cannot read embedder vocabulary {path}: {exc}") from exc
        # A trailing newline makes a final empty entry; a vocabulary has no empty token.
        return cls({token: i for i, token in enumerate(lines) if token})

    def split(self, text: str) -> list[str]:
        """``text`` as whitespace-and-punctuation words, lower-cased and unaccented."""
        cleaned: list[str] = []
        for char in unicodedata.normalize("NFD", text):
            if char in ("\t", "\n", "\r"):
                cleaned.append(" ")
                continue
            category = unicodedata.category(char)
            if category == "Mn" or category.startswith("C") or ord(char) == 0xFFFD:
                continue
            if _is_cjk(char) or _is_punctuation(char):
                cleaned.extend((" ", char, " "))
            else:
                cleaned.append(char)
        return "".join(cleaned).lower().split()

    def pieces(self, word: str) -> list[str]:
        """One word as WordPiece tokens, or ``["[UNK]"]`` when it cannot be spelled."""
        if len(word) > 100:  # the reference's own cut-off for a pathological "word"
            return [_UNK]
        out: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found: str | None = None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [_UNK]
            out.append(found)
            start = end
        return out

    def encode(self, text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[int]:
        """``text`` as token ids, wrapped in ``[CLS]``/``[SEP]`` and cut to
        ``max_tokens`` including them."""
        ids = [self.cls_id]
        for word in self.split(text):
            for piece in self.pieces(word):
                ids.append(self.vocab.get(piece, self.unk_id))
                if len(ids) >= max_tokens:
                    return [*ids[: max_tokens - 1], self.sep_id]
        ids.append(self.sep_id)
        return ids


# --------------------------------------------------------------------------------------
# The embedder
# --------------------------------------------------------------------------------------


class OnnxTextEmbedder:
    """all-MiniLM-L6-v2 through onnxruntime: 384 dimensions, mean-pooled over the
    attention mask and L2-normalized, which is what ``sentence-transformers`` does with
    the same checkpoint and what Embedder promises.

    The session is opened on the FIRST ``embed`` rather than in ``__init__``, because
    ``skillweaver skills ls`` builds a retriever and never ranks anything; the files are
    checked for existence up front by :func:`load_embedder`, which is cheap.
    """

    def __init__(
        self,
        model_path: Path,
        vocab_path: Path,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        threads: int = DEFAULT_THREADS,
    ) -> None:
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.max_tokens = max_tokens
        self.threads = threads
        self._session: Any | None = None
        self._tokenizer: WordPiece | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One L2-normalized vector per text, in order. Empty input gives ``[]``.

        Raises:
            ProviderError: if the model or vocabulary cannot be loaded or the model
                cannot be run.
        """
        if not texts:
            return []
        session, tokenizer = self._ready()
        import numpy as np

        encoded = [tokenizer.encode(text, self.max_tokens) for text in texts]
        width = max(len(ids) for ids in encoded)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1
        feed = {"input_ids": input_ids, "attention_mask": mask}
        if any(i.name == "token_type_ids" for i in session.get_inputs()):
            feed["token_type_ids"] = np.zeros_like(input_ids)
        try:
            hidden = session.run(None, feed)[0]
        except Exception as exc:  # onnxruntime raises its own exception types
            raise ProviderError(f"embedding model failed: {exc}") from exc
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return [[float(v) for v in row] for row in pooled / norms]

    def _ready(self) -> tuple[Any, WordPiece]:
        """The loaded session and tokenizer, opening them on first use.

        Raises:
            ProviderError: if either cannot be loaded.
        """
        if self._session is None or self._tokenizer is None:
            self._tokenizer = WordPiece.from_file(self.vocab_path)
            self._session = self._open_session()
            log.debug(
                "embedder.loaded",
                model=str(self.model_path),
                vocab=len(self._tokenizer.vocab),
                threads=self.threads,
            )
        return self._session, self._tokenizer

    def _open_session(self) -> Any:
        """Open the onnxruntime session with an explicit thread count.

        Raises:
            ProviderError: if onnxruntime is missing or the model will not load.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - onnxruntime ships with RapidOCR
            raise ProviderError(f"onnxruntime is not available: {exc}") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = self.threads
        try:
            return ort.InferenceSession(
                str(self.model_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ProviderError(f"cannot load embedding model {self.model_path}: {exc}") from exc

    def __repr__(self) -> str:
        loaded = "loaded" if self._session is not None else "not loaded yet"
        return f"OnnxTextEmbedder({self.model_path.name}, {loaded})"


# --------------------------------------------------------------------------------------
# Whether there is one
# --------------------------------------------------------------------------------------


def embedder_for(config: Settings) -> tuple[OnnxTextEmbedder | None, str]:
    """``(embedder, reason it is absent)`` for these settings - never both.

    A reason rather than ``None`` alone because a silent fall-back to word overlap is a
    feature nobody can measure: it travels into every candidate's ``why``, so a run that
    ranked on keywords says which of the three ways it got there.

    Nothing here loads a model: two ``exists`` calls.
    """
    if not config.embedder_enabled:
        return None, "turned off by SKILLWEAVER_EMBEDDER"
    directory = config.embedder_dir
    model, vocab = directory / MODEL_FILE, directory / VOCAB_FILE
    missing = [p.name for p in (model, vocab) if not p.is_file()]
    if missing:
        return None, (
            f"no local weights: {', '.join(missing)} not in {directory} "
            "(run scripts/fetch_embedder.py)"
        )
    return OnnxTextEmbedder(model, vocab), ""


@lru_cache(maxsize=1)
def load_embedder() -> tuple[OnnxTextEmbedder | None, str]:
    """:func:`embedder_for` over the process-wide settings, resolved once. Cached because
    the answer cannot change without the process's settings changing. A caller that
    moves the weights afterwards calls ``load_embedder.cache_clear()``."""
    return embedder_for(settings())


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Plain cosine of two vectors, for scripts that want to check an embedder by
    hand. Retrieval does not call this - it has its own, which clamps and reports a
    length mismatch as a :class:`~skillweaver.errors.ProviderError`."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if scale == 0.0 else dot / scale
