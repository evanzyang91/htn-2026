"""The concrete :class:`~skillweaver.contracts.Embedder` retrieval has always been
written for, and the factory that decides whether there is one.

``skills/retrieve.py`` has accepted an ``embedder`` since it was written, computes the
cosine, caches the vectors and reports which backend ranked. Nothing ever constructed
one, so every run in this project's history took the ``None`` branch and ranked on word
overlap alone. This module is that missing piece.

Local, not hosted
-----------------

The backend is **all-MiniLM-L6-v2 run locally through onnxruntime**, which is already a
dependency (RapidOCR brings it) - so this adds NO package to ``pyproject.toml``, which
is shared surface, and no API key to the demo.

The alternative considered was a hosted embedding API. Anthropic does not serve one, so
that would have meant Gemini's ``embed_content`` over ``google-genai``. It was rejected
on the project's own terms rather than on taste:

* retrieval runs on the fast path, and the whole claim of a warm run is that it answers
  in single-digit seconds with zero model calls - a network round trip before the first
  action spends the saving the feature exists to produce (the ordering suite's warm hits
  ran in 4.3 to 5.7 seconds END TO END);
* it would make a laptop with the wifi off, or a provider having a bad afternoon, a
  demo that ranks worse than it did before;
* and it puts a per-call price on a ranking that happens on every single run.

Measured here, the local read costs ~14 ms for five short texts and ~0.1 s to load the
session once, against ~90 MB on disk. See :data:`MEASURED_TIMINGS`.

What it is worth, measured
--------------------------

Less than it looks, and the number that says so is not the one this was built to move.
``scripts/bench_retrieval.py`` scores both rankings over the two libraries this
repository carries, 48 requests in two shapes:

    ranking    top-1 recall    runnable    irrelevant requests answered
    keywords      18/24         3 or 9/24            1/6
    embedder      21/24         3 or 9/24            5/6

Recall is what an embedder is for and it really does improve: three paraphrases that
word overlap ranked wrongly - *save this table to a comma separated file*, *find the
rows belonging to Acme Corp*, *answer the open thread* - now rank their own skill
first. Not one of them becomes a warm run. A candidate still has to BIND
(:func:`~skillweaver.agent.planner.bind_args`) and then account for the request
(:data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR`), and both of those are counted
in words, so a request worded differently fails them for the same reason it ranked
badly. Better ranking arrives at a door that is locked in the same language.

And the third column is the one to weigh against the first, because it moved in the
wrong direction: five of six requests that NOTHING in the library can do came back
with a candidate anyway, against one of six on keywords. A cosine is almost never
zero, so "nothing is relevant" stops being expressible as an empty list. Raising
``min_score`` is the obvious answer and does not work: on these corpora the best
irrelevant request scores 0.198 while two correct paraphrases score 0.191 and 0.198,
so no cut-off separates the populations and tuning one would be fitting the cut to
this bench. Nothing downstream ran a wrong skill - the content gate refused all of
them, which is the same gate that refuses the gains - but that is the gate's credit,
not the ranking's.

So :data:`~skillweaver.config.DEFAULT_EMBEDDER_ENABLED` is ``False``, and only that
switch can change it: the weights being on disk is deliberately NOT an implicit yes,
because a laptop that ran ``make embedder`` once must not quietly rank differently
from a clean clone. This module is kept, not deleted, because the finding is about
where the bottleneck IS: the next person to work on warm-path reuse should spend it on
binding and on the content gate, and re-run the bench with ``SKILLWEAVER_EMBEDDER=true``
once either has moved.

Re-measured after both moved, and it still does not pay
-------------------------------------------------------

Both moved on 2026-09-20: binding reads a value through the slot a proven request left
(:func:`~skillweaver.agent.planner._through_slot`), and the content gate lets relatives
that perform the same workflow vouch in their own words
(:func:`~skillweaver.agent.planner.account_of`, :mod:`skillweaver.skills.family`). The
same bench, the same 48 requests, scored through the planner's own new gate:

    ranking    top-1 recall    runnable    irrelevant requests answered
    keywords      18/24         3 or 8/24            1/6
    embedder      21/24         3 or 8/24            5/6

The embedder still converts none of its recall into a warm run and still answers five
of six irrelevant requests, so the switch stays off. One number DID move, for both
rankings alike and in the wrong direction: 9/24 became 8/24 when a suite supplies the
values. That is the new intent gate (:func:`~skillweaver.agent.planner.asks_for`)
declining *Bring up the list of records* for a skill learned as *Open the records
list ...* - ``bring`` is on no verb list, and a verb the gate does not know matches
only itself. It is a correct run lost, stated here because it is the price of refusing
*remove ... from my cart* on an add-to-cart skill, and the list was NOT extended to
win the bench back. WHY nothing else moved is the useful
part, because the new readers demonstrably do convert rewordings into warm runs on a
live site (*Buy the "..."* against a skill learned as *Add the "..." to the cart*: 0
model calls). This bench's paraphrases were written to avoid each skill's own nouns AND
its sentence shape - *look up Grace Hopper on Wikipedia and bring up her page* for
*search Wikipedia for "..." and open her article* - and the slot reader deliberately
refuses those: every content word outside the value must agree, because the alternative
is running a skill on a guess. And neither library here was admitted after signatures
existed, so neither holds a family to vouch. What the new gates reach is the request
that keeps the errand's shape and changes its verb, its value or its tail; what this
bench asks for is a request that shares nothing but meaning, and no model-free reader
in this project binds an argument out of one of those. An embedder would have to be
paired with a binder that can - which is the composer, and it already costs a model
call. The precision loss (1/6 -> 5/6) is unchanged too, and by itself still decides it.

What is deliberately NOT here
-----------------------------

The weights are not committed. ``data/`` is git-ignored except for the UI detector, and
a 90 MB file in a repository that a demo clones is a cost with no payoff when the thing
degrades honestly without it. ``scripts/fetch_embedder.py`` downloads them; until it has
been run, :func:`load_embedder` returns ``None`` **with a reason**, retrieval ranks on
word overlap exactly as it does today, and the reason is printed in the candidate's
``why``. That is what keeps ``make test`` free of any network call.

Tokenization is done here rather than by ``tokenizers``/``transformers`` for the same
reason: the dependency list is shared surface. WordPiece over a 30k ``vocab.txt`` is a
small, exactly specified algorithm, and :class:`WordPiece` implements the BERT-uncased
one including its punctuation and CJK splitting - which matters, because a skill's
searchable text is full of ``snake_case`` names and ``_`` is BERT punctuation.
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

all-MiniLM-L6-v2 was trained with a 256-token window and its position table stops at
512, so this is the model's own limit rather than a budget. It also bounds the one
native call in this module: attention is quadratic in the sequence length, so a
skill whose docstring runs long costs a fixed ceiling rather than an open one. Skill
text measured on the shipped Wikipedia library runs 60 to 130 tokens, so nothing real
is truncated today; a longer docstring loses its tail, which is the part a summary
already repeats.
"""

DEFAULT_THREADS = 1
"""onnxruntime threads for the embedding session.

Set EXPLICITLY, for the reason written up in :mod:`skillweaver.perception.ocr`: left
alone, onnxruntime sizes its pool from the core count and its workers spin. One thread
is not a compromise here - the batch is a handful of short sentences and the session
measured 14 ms for five of them single-threaded, so there is nothing to parallelize
that the pool would not spend more time waking up for.
"""

MEASURED_TIMINGS = {
    "session_load_ms": 92.0,
    "five_short_texts_ms": 14.1,
    "model_bytes": 90_405_214,
}
"""Measured 2026-09-19 on the demo laptop (macOS arm64, onnxruntime 1.30, 1 thread).

Quoted so the hosted-versus-local decision in the module docstring can be re-checked
rather than re-argued: a whole library re-embeds in the time one network round trip
spends on its TLS handshake.
"""

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
    splitting, then greedy longest-match-first WordPiece with ``##`` continuations and
    ``[UNK]`` for anything the vocabulary cannot spell.

    Args:
        vocab: token to id, in file order.

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
    attention mask and L2-normalized, which is what ``sentence-transformers`` does
    with the same checkpoint and what :class:`~skillweaver.contracts.Embedder`
    promises.

    The session is opened on the FIRST ``embed`` rather than in ``__init__``, because
    ``skillweaver skills ls`` builds a retriever and never ranks anything; the files
    are checked for existence up front by :func:`load_embedder`, which is cheap.

    Args:
        model_path: the ``model.onnx`` to run.
        vocab_path: the ``vocab.txt`` beside it.
        max_tokens: sequence ceiling; see :data:`DEFAULT_MAX_TOKENS`.
        threads: onnxruntime thread count; see :data:`DEFAULT_THREADS`.
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

    A reason is returned rather than ``None`` alone because a silent fall-back to word
    overlap is a feature nobody can measure: the reason travels into every candidate's
    ``why``, so a run that ranked on keywords says which of the three ways it got there
    - turned off, weights never fetched, or files half present.

    Nothing here loads a model: it is two ``exists`` calls, so a command that never
    ranks anything pays nothing.
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
    """:func:`embedder_for` over the process-wide settings, resolved once.

    Cached because it is called wherever a retriever is built and the answer cannot
    change without the process's settings changing; tests that move the weights call
    ``load_embedder.cache_clear()``.
    """
    return embedder_for(settings())


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Plain cosine of two vectors, for scripts that want to check an embedder by
    hand. Retrieval does not call this - it has its own, which clamps and reports a
    length mismatch as a :class:`~skillweaver.errors.ProviderError`."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if scale == 0.0 else dot / scale
