"""``SkillRetriever``: which stored skills are worth showing the planner.

Two signals, deliberately:

*Cosine* over an :class:`~skillweaver.contracts.Embedder`'s vectors of each skill's
summary and docstring. It understands that "pay a bill" and "settle an invoice" are
the same errand.

*Token overlap* between the task and the skill's name, summary, docstring, parameter
names and - measurably the most useful of them - the sentence the skill was learned
from. It is crude, but it needs no model, no network and no API key - so the library
is still useful on a laptop with the wifi off, and the demo still works when a
provider is down. With no embedder configured this signal is the whole score.

The learned sentence (``provenance.task_text``) is in there because a summary is
written by a model to describe a skill, while the learned sentence is what a PERSON
actually typed to get it; asking for the same errand again tends to reuse the
person's words, not the model's. Measured over the four skills the live Wikipedia
suite builds, adding it left the top-ranked skill unchanged on all six tasks and
widened the margin to the runner-up on four of them - most sharply on a verbatim
repeat, which now scores a clean 1.0.

They are blended (``lexical_weight``, 0.3 by default) rather than switched between,
because the cheap signal is a good sanity check on the expensive one: a skill the
embedder likes but that shares no word with the task is usually a near-miss.

Which of the two actually ranked is always stated
-------------------------------------------------

An embedder can be absent (nobody fetched the weights, someone turned it off) or break
mid-run, and in both cases this class keeps working on token overlap alone. That is the
right behaviour and a silent version of it would be a defect: every measurement of what
the embedder is worth compares two runs whose ONLY difference is supposed to be the
embedder, and a run that quietly fell back is indistinguishable from one that never had
one. So ``why`` names the state in words - ``cosine 0.62 ...``, ``no embedder (<why
not>): keyword and token overlap only`` - and ``ranked_by`` says the same thing to code.

A failing embedder is a fall-back only when the caller asked for one
(``degrade_on_error``). Injected directly it still raises
:class:`~skillweaver.errors.ProviderError`, which is what
:class:`~skillweaver.contracts.SkillRetriever` promises and what a test that hands in a
broken backend is entitled to see.

Every :class:`~skillweaver.contracts.Candidate` carries a ``why`` built from what
actually matched - the shared words, the cosine, the name hit - and ends with what the
skill has NO account of, because a score on its own reads as agreement.

A ranking has a winner even when nothing fits
---------------------------------------------

One measured property of the name term, because it is not obvious and it has already
cost a suite its most dangerous wrong answer. ``name_hit`` is divided by the length of
the SKILL'S name, so a short generic name is easy to match completely. On 2026-09-19
``open_order_screen`` - three stems, all of which appear in any sentence about the
Order screen - topped a composite ordering task at 0.618 with ``name_hit`` of 1.00,
while its coverage of that task was 0.364, the LOWEST of any candidate. The skills
that could actually do the errand carry long specific names (five stems for
``add_two_dishes_then_remove_one``, of which the task matched one) and scored 0.20 to
0.33 on the same term.

The ranking is left as it is: it is calibrated against the live Wikipedia suite above,
and being closest is all it claims. What that measurement argues is that CLOSEST must
not be read as GOOD ENOUGH TO RUN, which is a decision and belongs to the planner -
see :data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR` and :func:`unaddressed`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from skillweaver.contracts import Candidate, Embedder, Skill, SkillStore
from skillweaver.errors import ProviderError, SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.skills.model import signature

__all__ = ["STOPWORDS", "SkillRetriever", "accounted_for", "tokenize", "unaddressed"]

log = get_logger(__name__)

STOPWORDS = frozenset(
    """
    a an and are as at be by do for from go goes how i in into is it its me my of on onto or
    please the their then there this to up us use using want we what when where which who will
    with you your
    """.split()
)
"""Words carried by almost every task text, so they say nothing about which skill fits."""

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords and one-character noise removed, in
    order and without duplicates. ``"Search the invoices"`` gives
    ``["search", "invoices"]``."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(text.lower()):
        if len(word) > 1 and word not in STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


def _stem(token: str) -> str:
    """A deliberately blunt stem so ``invoice`` matches ``invoices`` and ``search``
    matches ``searching``. Wrong in the usual English corner cases and right often
    enough to matter for a handful of skill names."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def _stems(tokens: Iterable[str]) -> set[str]:
    return {_stem(t) for t in tokens}


def _normalized(text: str) -> str:
    """``text`` reduced to its meaningful word stems, in order, so two phrasings of
    the same sentence that differ only in punctuation, case or stopwords compare
    equal. Used to spot a task that is the sentence a skill was learned from."""
    return " ".join(_stem(t) for t in tokenize(text))


def searchable_text(skill: Skill) -> str:
    """The text a skill is matched on: its rendered signature - so the name and the
    parameter names count - then the sentence it was learned from, its summary and
    its docstring."""
    return f"{signature(skill)}\n{skill.provenance.task_text}\n{skill.summary}\n{skill.docstring}"


def unaddressed(task: str, skill: Skill, args: Mapping[str, Any] | None = None) -> list[str]:
    """The words of ``task`` that ``skill`` neither talks about nor was handed.

    Ranking answers "which of these is closest?", which always has a winner. This
    answers the other question, the one a closest-match ranking cannot: *is there
    anything in the request this skill has no account of at all?*

    A word is accounted for when it appears in the skill's own text - its name,
    summary, docstring, parameter names and the sentence it was learned from - or in
    the VALUES it is about to be called with. The second half is what keeps this from
    rejecting correct reuse: a skill learned from *order two Vegetable Rolls from
    Sakura Counter* says nothing about falafel, and ordering a falafel wrap through it
    is exactly what it is for, because "falafel" arrives as an argument.

    What is left over is the part of the errand nobody has promised to do.

    Args:
        task: The request, in the words it was asked in.
        skill: The candidate.
        args: The arguments the skill would be called with, if they are known.
            ``None`` means judge the skill's text alone, which is stricter.

    Returns:
        The unaccounted words, in the order the task used them and without
        duplicates. Empty means every word of the request is spoken for.
    """
    known = _stems(tokenize(searchable_text(skill)))
    if args:
        for value in args.values():
            known |= _stems(tokenize(str(value)))
    return [word for word in tokenize(task) if not _has_account(_stem(word), known)]


def _has_account(stem: str, known: set[str]) -> bool:
    """Whether ``stem`` is spoken for by anything in ``known``, near misses included.

    :func:`_stem` is deliberately blunt and, as a result, not symmetric: ``invoices``
    becomes ``invoic`` while ``invoice`` is left alone, so the two do not compare
    equal. Ranking survives that - a near miss only costs a skill some score - but
    here it would be the difference between "spoken for" and "nobody has promised to
    do this", and a plural is not a missing promise.

    So a stem also counts as accounted for when a known one extends it by at most two
    characters. Generosity is the safe direction for THIS question: the check exists
    to catch words with no account at all, and being wrong here means declining a
    skill that would have worked.
    """
    if stem in known:
        return True
    return any(_a_near_miss(stem, other) for other in known)


def _a_near_miss(left: str, right: str) -> bool:
    """Whether two stems are the same word bar an ending: one starts the other, the
    shorter is long enough not to be a coincidence, and at most two characters
    separate them. ``invoic``/``invoice`` yes; ``cart``/``carton`` no."""
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(short) >= 4 and len(long) - len(short) <= 2 and long.startswith(short)


def accounted_for(task: str, skill: Skill, args: Mapping[str, Any] | None = None) -> float:
    """What fraction of ``task``'s words :func:`unaddressed` finds an account of, in
    ``0.0..1.0``. ``1.0`` means nothing in the request is unexplained; an empty task
    scores ``1.0``, there being nothing left over."""
    words = tokenize(task)
    if not words:
        return 1.0
    return 1.0 - len(unaddressed(task, skill, args)) / len(words)


class SkillRetriever:
    """A :class:`~skillweaver.contracts.SkillRetriever` over any
    :class:`~skillweaver.contracts.SkillStore`.

    Args:
        store: where the skills come from. Demoted skills never leave it, because
            ``store.list()`` omits them by default.
        embedder: optional. ``None`` means pure token overlap - a fully working
            retriever with no model behind it.
        lexical_weight: how much of the score the token overlap contributes when an
            embedder is present, in ``0.0..1.0``.
        min_score: candidates at or below this are dropped, so "nothing is relevant"
            comes back as an empty list rather than a page of noise.
        unavailable_reason: why ``embedder`` is ``None``, in a few words, when
            something tried to build one and could not. It is quoted in every
            candidate's ``why``, so a keyword-ranked run says whether the model was
            turned off, never fetched, or simply not asked for.
        degrade_on_error: whether an embedder that FAILS disables itself and lets the
            search finish on token overlap. ``False`` - the default, and what the
            :class:`~skillweaver.contracts.SkillRetriever` protocol documents - lets
            the :class:`~skillweaver.errors.ProviderError` out. The orchestrator asks
            for ``True``: on a live run a broken backend should cost the ranking its
            second signal, not cost the agent its whole library.
    """

    def __init__(
        self,
        store: SkillStore,
        embedder: Embedder | None = None,
        *,
        lexical_weight: float = 0.3,
        min_score: float = 0.0,
        unavailable_reason: str = "",
        degrade_on_error: bool = False,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise SkillWeaverError(f"lexical_weight must be in 0.0..1.0, got {lexical_weight}")
        self.store = store
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.min_score = min_score
        self.degrade_on_error = degrade_on_error
        self._absent = unavailable_reason
        self._vectors: dict[str, list[float]] = {}

    @property
    def ranked_by(self) -> str:
        """``"embedder"`` or ``"keywords"``: which signal this retriever is ranking
        with RIGHT NOW, after any fall-back. What the measurement reads."""
        return "embedder" if self.embedder is not None else "keywords"

    @property
    def fallback_reason(self) -> str:
        """Why there is no embedder, or ``""`` when there is one. Set at construction
        or, after a backend fails under ``degrade_on_error``, to that failure."""
        return "" if self.embedder is not None else self._absent

    # -- scoring -----------------------------------------------------------------

    @staticmethod
    def _lexical(task_tokens: Sequence[str], skill: Skill) -> tuple[float, list[str], list[str]]:
        """``(score, shared words, name words the task also used)``.

        Coverage of the *task* drives the score - a skill that speaks to every word of
        the request beats a sprawling one that happens to contain them - with a name
        hit worth as much as the rest of the text together, because a skill called
        ``search_invoice`` really is what "search for an invoice" wants.

        The body is the skill's summary, docstring, parameter names AND the sentence
        it was learned from; see the module docstring for why the last one earns its
        place.
        """
        if not task_tokens:
            return 0.0, [], []
        wanted = _stems(task_tokens)
        name_stems = _stems(tokenize(skill.name))
        body = f"{skill.summary} {skill.docstring} {' '.join(skill.params)} "
        body_stems = _stems(tokenize(body + skill.provenance.task_text))

        shared = [t for t in task_tokens if _stem(t) in (name_stems | body_stems)]
        on_name = [t for t in task_tokens if _stem(t) in name_stems]
        coverage = len(wanted & (name_stems | body_stems)) / len(wanted)
        name_hit = len(wanted & name_stems) / len(name_stems) if name_stems else 0.0
        return min(1.0, 0.6 * coverage + 0.4 * name_hit), shared, on_name

    def _embed_all(self, task: str, skills: Sequence[Skill]) -> list[float] | None:
        """Cosine of the task against each skill, or ``None`` with no embedder.

        Vectors are cached by the exact text, so re-searching a stable library costs
        one embedding of the task.

        Raises:
            ProviderError: if the embedding backend fails.
        """
        if self.embedder is None:
            return None
        try:
            return self._cosines(task, skills)
        except ProviderError as exc:
            if not self.degrade_on_error:
                raise
            # Once, and then never again this run: a backend that failed on one batch
            # of short strings is not going to succeed on the next, and a retriever
            # that retried it would pay the timeout on every task.
            self.embedder = None
            self._absent = f"embedder failed and was dropped: {exc}"
            log.warning("skills.embedder_dropped", error=str(exc))
            return None

    def _cosines(self, task: str, skills: Sequence[Skill]) -> list[float]:
        """:meth:`_embed_all` with the embedder known to be present.

        Raises:
            ProviderError: if the embedding backend fails.
        """
        assert self.embedder is not None
        texts = [task] + [searchable_text(s) for s in skills]
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [t for t, k in zip(texts, keys, strict=True) if k not in self._vectors]
        if missing:
            try:
                fresh = self.embedder.embed(list(dict.fromkeys(missing)))
            except ProviderError:
                raise
            except Exception as exc:  # a third-party client can raise anything
                raise ProviderError(f"embedding failed: {exc}") from exc
            for text, vector in zip(dict.fromkeys(missing), fresh, strict=True):
                self._vectors[hashlib.sha256(text.encode("utf-8")).hexdigest()] = list(vector)
        task_vector = self._vectors[keys[0]]
        return [self._cosine(task_vector, self._vectors[k]) for k in keys[1:]]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        """Cosine similarity, clamped to ``0.0..1.0``.

        The contract says vectors arrive L2-normalized, so a dot product is already
        the cosine; the norms below only keep a sloppy embedder from wrecking the
        ranking. A negative cosine means "less related than unrelated", which is not a
        distinction worth ranking on, so it clamps to zero.
        """
        if len(left) != len(right):
            raise ProviderError(
                f"embedder returned vectors of different lengths: {len(left)} and {len(right)}"
            )
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        scale = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        if scale == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / scale))

    @staticmethod
    def _why(
        cosine: float | None,
        shared: Sequence[str],
        on_name: Sequence[str],
        skill: Skill,
        *,
        verbatim: bool = False,
        missing: Sequence[str] = (),
        absent: str = "",
    ) -> str:
        """The sentence a human reads next to the candidate.

        It ends with what the skill has NO account of, when there is anything, because
        a score alone reads as agreement. Measured on 2026-09-19: the trivial
        ``open_order_screen`` topped a composite ordering task at 0.618 on a perfect
        name hit - every stem of its three-word name appears in any sentence about the
        Order screen - while speaking to 4 of the task's 11 meaning-words. The score
        said "best"; the leftovers said "vegetable, rolls, sakura, counter, confirm",
        which is the sentence a reader needed.
        """
        parts: list[str] = []
        if verbatim:
            parts.append("the exact sentence this skill was learned from")
        if cosine is None:
            because = f" ({absent})" if absent else ""
            parts.append(f"no embedder{because}: keyword and token overlap only")
        else:
            parts.append(f"cosine {cosine:.2f} on the skill's searchable text")
        if shared:
            parts.append("shares " + ", ".join(f"'{w}'" for w in shared[:5]))
        else:
            parts.append("no shared keywords")
        if on_name:
            parts.append(f"name '{skill.name}' matches " + ", ".join(f"'{w}'" for w in on_name[:3]))
        if skill.stats.runs:
            parts.append(f"{skill.stats.successes}/{skill.stats.runs} runs succeeded")
        if missing:
            parts.append("no account of " + ", ".join(f"'{w}'" for w in missing[:5]))
        return "; ".join(parts)

    # -- the contract ------------------------------------------------------------

    def search(self, task: str, domain: str | None = None, k: int = 5) -> list[Candidate]:
        """At most ``k`` candidates for ``task``, best score first.

        ``domain`` restricts the search to one site or app. Demoted skills are never
        returned. Candidates scoring at or below ``min_score`` are dropped, so an
        irrelevant task gives an empty list. Ties break on ``(domain, name)``, so the
        same library and task always give the same order.

        Raises:
            ProviderError: if the embedding backend fails.
        """
        if k <= 0:
            return []
        skills = self.store.list(domain=domain)
        if not skills:
            return []

        task_tokens = tokenize(task)
        cosines = self._embed_all(task, skills)
        weight = self.lexical_weight if cosines is not None else 1.0
        asked = _normalized(task)

        scored: list[Candidate] = []
        below: list[str] = []
        for position, skill in enumerate(skills):
            lexical, shared, on_name = self._lexical(task_tokens, skill)
            cosine = None if cosines is None else cosines[position]
            score = lexical if cosine is None else (1.0 - weight) * cosine + weight * lexical
            if score <= self.min_score:
                below.append(f"{skill.name}={score:.3f}")
                continue
            verbatim = _normalized(skill.provenance.task_text) == asked
            scored.append(
                Candidate(
                    skill=skill,
                    score=round(score, 6),
                    why=self._why(
                        cosine,
                        shared,
                        on_name,
                        skill,
                        verbatim=verbatim,
                        missing=unaddressed(task, skill),
                        absent=self._absent,
                    ),
                )
            )

        scored.sort(key=lambda c: (-c.score, c.skill.domain, c.skill.name))
        top = scored[:k]
        log.debug(
            "skills.search",
            task=task,
            domain=domain or "*",
            considered=len(skills),
            returned=len(top),
            best=top[0].skill.name if top else "",
            scores=", ".join(f"{c.skill.name}={c.score:.3f}" for c in scored),
            below_min=", ".join(below),
            embedder=type(self.embedder).__name__ if self.embedder else "none",
            ranked_by=self.ranked_by,
            fallback=self.fallback_reason,
        )
        return top

    def __repr__(self) -> str:
        backend = type(self.embedder).__name__ if self.embedder is not None else "keywords"
        return f"SkillRetriever(store={self.store!r}, ranking={backend})"
