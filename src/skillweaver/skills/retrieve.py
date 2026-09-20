"""``SkillRetriever``: which stored skills are worth showing the planner.

Two signals blended (``lexical_weight``, 0.3), not switched between, because the cheap
one sanity-checks the expensive one. *Cosine* over an Embedder's vectors knows "pay a
bill" and "settle an invoice" are one errand; *token overlap* needs no model, so the
library still works with the wifi off, and with no embedder it is the whole score.

The sentence the skill was LEARNED from is in the overlap text and is measurably the
most useful part of it: a summary is written by a model, the learned sentence is what a
PERSON typed, and asking again tends to reuse the person's words. Over the four skills
the live Wikipedia suite builds, adding it left the top-ranked skill unchanged on all
six tasks and widened the margin on four.

**Which signal actually ranked is always stated** (``why``, ``ranked_by``): a silent
fall-back would make a degraded run indistinguishable from one that never had an
embedder, which is exactly the comparison every measurement of its worth depends on.
A failing embedder falls back only when asked (``degrade_on_error``).

**A ranking has a winner even when nothing fits.** ``name_hit`` divides by the length of
the SKILL'S name, so a short generic name matches completely: on 2026-09-19
``open_order_screen`` topped a composite ordering task at 0.618 with ``name_hit`` 1.00
while covering 0.364 of it, the LOWEST of any candidate. The ranking is left alone -
being closest is all it claims - but CLOSEST is not GOOD ENOUGH TO RUN, which is the
planner's decision (:data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR`,
:func:`unaddressed`).
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


def unaddressed(
    task: str,
    skill: Skill,
    args: Mapping[str, Any] | None = None,
    *,
    family: Sequence[Skill] = (),
) -> list[str]:
    """The words of ``task`` that ``skill`` neither talks about nor was handed.

    Ranking answers "which of these is closest?", which always has a winner. This answers
    the other question: is there anything in the request this skill has no account of?

    A word is accounted for when it appears in the skill's own text - name, summary,
    docstring, parameter names, the sentence it was learned from - or in the VALUES it is
    about to be called with. That second half is what keeps this from rejecting correct
    reuse: a skill learned from *order two Vegetable Rolls* says nothing about falafel,
    and ordering a falafel wrap through it is exactly what it is for.

    Two more things may speak for a word, both EARNED rather than claimed: the requests
    this skill is proven to have served (its ``precedents``), and ``family`` - skills
    performing the same workflow, usually on other sites - vouching in their own words,
    because what one shop calls *purchase* another learned as *add to cart*. The CALLER
    decides who is family and must have checked intent as well as shape; this only counts,
    and the threshold does not move for either.

    Returns:
        The unaccounted words, in the order the task used them and without duplicates.
    """
    known = _stems(tokenize(_own_words(skill)))
    for relative in family:
        known |= _stems(tokenize(_own_words(relative)))
    if args:
        for value in args.values():
            known |= _stems(tokenize(str(value)))
    return [word for word in tokenize(task) if not _has_account(_stem(word), known)]


def _own_words(skill: Skill) -> str:
    """Everything ``skill`` can say for itself: its searchable text and every request
    it is proven to have served."""
    served = "\n".join(p.task_text for p in skill.precedents)
    return f"{searchable_text(skill)}\n{served}"


def _has_account(stem: str, known: set[str]) -> bool:
    """Whether ``stem`` is spoken for by anything in ``known``, near misses included.

    :func:`_stem` is blunt and not symmetric - ``invoices`` becomes ``invoic`` while
    ``invoice`` is left alone - which ranking survives but this would not: a plural is not
    a missing promise. So a stem also counts when a known one extends it by at most two
    characters. Generosity is the safe direction here, since being wrong means declining a
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


def accounted_for(
    task: str,
    skill: Skill,
    args: Mapping[str, Any] | None = None,
    *,
    family: Sequence[Skill] = (),
) -> float:
    """What fraction of ``task``'s words :func:`unaddressed` finds an account of, in
    ``0.0..1.0``. ``1.0`` means nothing in the request is unexplained; an empty task
    scores ``1.0``, there being nothing left over."""
    words = tokenize(task)
    if not words:
        return 1.0
    return 1.0 - len(unaddressed(task, skill, args, family=family)) / len(words)


class SkillRetriever:
    """A SkillRetriever over any SkillStore.

    Args:
        embedder: ``None`` means pure token overlap - a fully working retriever with no
            model behind it.
        min_score: Candidates at or below this are dropped, so "nothing is relevant" comes
            back as an empty list rather than a page of noise.
        unavailable_reason: Why ``embedder`` is ``None``, quoted in every candidate's
            ``why``, so a keyword-ranked run says whether the model was turned off, never
            fetched, or simply not asked for.
        degrade_on_error: Whether an embedder that FAILS disables itself and lets the
            search finish on token overlap. ``False`` (the default, and what the protocol
            documents) lets the ProviderError out; the orchestrator asks for ``True``,
            because a broken backend should cost the ranking its second signal, not cost
            the agent its whole library.
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

        Coverage of the TASK drives the score - a skill that speaks to every word of the
        request beats a sprawling one that happens to contain them - with a name hit worth
        as much as the rest of the text together.

        The body is the summary, docstring, parameter names AND the sentence it was
        learned from; every later request the skill is PROVEN to have served counts the
        same way and for the same reason.
        """
        if not task_tokens:
            return 0.0, [], []
        wanted = _stems(task_tokens)
        name_stems = _stems(tokenize(skill.name))
        body = f"{skill.summary} {skill.docstring} {' '.join(skill.params)} "
        served = " ".join(p.task_text for p in skill.precedents)
        body_stems = _stems(tokenize(f"{body}{skill.provenance.task_text} {served}"))

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
            # Once, and then never again this run: a backend that failed on one batch of
            # short strings will not succeed on the next, and retrying pays the timeout
            # on every task.
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

        The contract says vectors arrive L2-normalized, so the norms below only keep a
        sloppy embedder from wrecking the ranking. A negative cosine is not a distinction
        worth ranking on, so it clamps to zero.
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

        It ends with what the skill has NO account of, because a score alone reads as
        agreement. Measured 2026-09-19: ``open_order_screen`` topped a composite ordering
        task at 0.618 on a perfect name hit while speaking to 4 of the task's 11
        meaning-words. The score said "best"; the leftovers said "vegetable, rolls,
        sakura, counter, confirm".
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

        Demoted skills are never returned, and candidates at or below ``min_score`` are
        dropped. Ties break on ``(domain, name)``, so the same library and task always
        give the same order.

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
