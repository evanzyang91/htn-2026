"""``SkillRetriever``: which stored skills are worth showing the planner.

Two signals, deliberately:

*Cosine* over an :class:`~skillweaver.contracts.Embedder`'s vectors of each skill's
summary and docstring. It understands that "pay a bill" and "settle an invoice" are
the same errand.

*Token overlap* between the task and the skill's name, summary, docstring and
parameter names. It is crude, but it needs no model, no network and no API key - so
the library is still useful on a laptop with the wifi off, and the demo still works
when a provider is down. With no embedder configured this signal is the whole score.

They are blended (``lexical_weight``, 0.3 by default) rather than switched between,
because the cheap signal is a good sanity check on the expensive one: a skill the
embedder likes but that shares no word with the task is usually a near-miss.

Every :class:`~skillweaver.contracts.Candidate` carries a ``why`` built from what
actually matched - the shared words, the cosine, the name hit - because the planner
puts it in a prompt and a human reads it over the demo's shoulder.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence

from skillweaver.contracts import Candidate, Embedder, Skill, SkillStore
from skillweaver.errors import ProviderError, SkillWeaverError
from skillweaver.logging_ import get_logger
from skillweaver.skills.model import signature

__all__ = ["STOPWORDS", "SkillRetriever", "tokenize"]

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


def searchable_text(skill: Skill) -> str:
    """The text a skill is matched on: its summary and docstring, as the contract
    says, preceded by its rendered signature so the name and parameter names count
    too."""
    return f"{signature(skill)}\n{skill.summary}\n{skill.docstring}"


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
    """

    def __init__(
        self,
        store: SkillStore,
        embedder: Embedder | None = None,
        *,
        lexical_weight: float = 0.3,
        min_score: float = 0.0,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise SkillWeaverError(f"lexical_weight must be in 0.0..1.0, got {lexical_weight}")
        self.store = store
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.min_score = min_score
        self._vectors: dict[str, list[float]] = {}

    # -- scoring -----------------------------------------------------------------

    @staticmethod
    def _lexical(task_tokens: Sequence[str], skill: Skill) -> tuple[float, list[str], list[str]]:
        """``(score, shared words, name words the task also used)``.

        Coverage of the *task* drives the score - a skill that speaks to every word of
        the request beats a sprawling one that happens to contain them - with a name
        hit worth as much as the rest of the text together, because a skill called
        ``search_invoice`` really is what "search for an invoice" wants.
        """
        if not task_tokens:
            return 0.0, [], []
        wanted = _stems(task_tokens)
        name_stems = _stems(tokenize(skill.name))
        body_stems = _stems(tokenize(f"{skill.summary} {skill.docstring} {' '.join(skill.params)}"))

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
        cosine: float | None, shared: Sequence[str], on_name: Sequence[str], skill: Skill
    ) -> str:
        """The sentence a human reads next to the candidate."""
        parts: list[str] = []
        if cosine is None:
            parts.append("no embedder: keyword and token overlap only")
        else:
            parts.append(f"cosine {cosine:.2f} on summary+docstring")
        if shared:
            parts.append("shares " + ", ".join(f"'{w}'" for w in shared[:5]))
        else:
            parts.append("no shared keywords")
        if on_name:
            parts.append(f"name '{skill.name}' matches " + ", ".join(f"'{w}'" for w in on_name[:3]))
        if skill.stats.runs:
            parts.append(f"{skill.stats.successes}/{skill.stats.runs} runs succeeded")
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

        scored: list[Candidate] = []
        for position, skill in enumerate(skills):
            lexical, shared, on_name = self._lexical(task_tokens, skill)
            cosine = None if cosines is None else cosines[position]
            score = lexical if cosine is None else (1.0 - weight) * cosine + weight * lexical
            if score <= self.min_score:
                continue
            scored.append(
                Candidate(
                    skill=skill,
                    score=round(score, 6),
                    why=self._why(cosine, shared, on_name, skill),
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
            embedder=type(self.embedder).__name__ if self.embedder else "none",
        )
        return top

    def __repr__(self) -> str:
        backend = type(self.embedder).__name__ if self.embedder is not None else "keywords"
        return f"SkillRetriever(store={self.store!r}, ranking={backend})"
