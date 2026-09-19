"""The element index and the source merge: turning raw detections into click targets.

Everything upstream of this module produces *candidates* - YOLO boxes with no text, OCR
lines with no idea what they label, a DOM dump that only exists offline. Everything
downstream asks questions in human terms: "the Submit button", "what is under the
cursor", "is the text 'Overdue' on screen". This module is the join.

Three pieces, in the order a pipeline uses them:

:func:`merge_elements`
    Fuses candidate lists into one, collapsing the same control seen twice (a YOLO
    button box and the OCR line sitting inside it) into a single element that has both
    the right kind and the right text.
:func:`stable_id`
    Gives an element an identity that survives a few pixels of layout drift, so a
    trajectory, a skill or a graph edge can refer to "that button" across observations.
:class:`ElementIndex`
    The real :class:`skillweaver.contracts.ElementIndex`: fuzzy text lookup that
    tolerates OCR noise, geometric lookup, and a cheap free-form ``best()``.

All geometry is LOGICAL pixels. Every query returns a list, ordered best-first, empty
when nothing matches, and never ``None``.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import math
import re
from typing import TYPE_CHECKING

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Point

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ElementIndex",
    "STABLE_ID_GRID",
    "build_index",
    "merge_elements",
    "normalize_text",
    "overlap_ratio",
    "reading_order",
    "stable_id",
    "with_stable_ids",
]


# --------------------------------------------------------------------------------------
# Text normalization and ordering
# --------------------------------------------------------------------------------------

_WHITESPACE = re.compile(r"[\s_|]+")

#: Characters OCR inserts around real text. Stripped from the edges of a comparison so
#: "Search:" and "[ Search ]" match a query of "Search".
_EDGE_NOISE = " \t\r\n:;,.!?*|_-–—()[]{}<>\"'`«»"


def normalize_text(text: str) -> str:
    """The canonical comparison form of a label: case-folded, whitespace collapsed,
    edge punctuation dropped.

    This is what makes ``"  SEARCH:  "`` and ``"search"`` the same string. It is also
    the exact form :func:`stable_id` hashes, so two observations of one control agree
    even when OCR adds a trailing colon.
    """
    folded = _WHITESPACE.sub(" ", text).strip().casefold()
    return folded.strip(_EDGE_NOISE)


def _words(text: str) -> list[str]:
    return [w for w in normalize_text(text).split(" ") if w]


def reading_order(elements: Iterable[Element]) -> list[Element]:
    """Elements in reading order: top-to-bottom by line, then left-to-right within a line.

    Sorting on ``(y, x)`` alone is not reading order - it scrambles a row of controls
    whose tops differ by a pixel or two, which is exactly what OCR and a detector
    produce for one toolbar. So elements are first grouped into lines by vertical
    overlap with the line's running span, then ordered by ``x`` inside each line.
    """
    remaining = sorted(elements, key=lambda e: (e.box.y, e.box.x, e.box.h))
    lines: list[tuple[float, float, list[Element]]] = []
    for element in remaining:
        box = element.box
        top, bottom = float(box.y), float(box.y + max(box.h, 1))
        centre = (top + bottom) / 2
        for i, (line_top, line_bottom, members) in enumerate(lines):
            line_centre = (line_top + line_bottom) / 2
            # Same line when each one's centre falls inside the other's vertical span:
            # tolerant of differing heights (a 14px label beside a 44px button) without
            # swallowing the row below.
            if line_top <= centre <= line_bottom and top <= line_centre <= bottom:
                members.append(element)
                lines[i] = (min(line_top, top), max(line_bottom, bottom), members)
                break
        else:
            lines.append((top, bottom, [element]))
    lines.sort(key=lambda line: (line[0], line[1]))
    ordered: list[Element] = []
    for _, _, members in lines:
        ordered.extend(sorted(members, key=lambda e: (e.box.x, e.box.y)))
    return ordered


# --------------------------------------------------------------------------------------
# Stable identity
# --------------------------------------------------------------------------------------

#: Side, in logical pixels, of the cell an element's centre is snapped to for
#: :func:`stable_id`. See that function for what the number buys and costs.
STABLE_ID_GRID = 32


def stable_id(element: Element, *, grid: int = STABLE_ID_GRID) -> str:
    """A short identity for ``element`` that survives small layout drift.

    Derived from three things: the kind, the :func:`normalize_text` form of the text,
    and the element's centre snapped to a ``grid``-pixel cell. So a banner appearing
    above the content, a font metric shifting a row by three pixels, or OCR adding a
    colon all leave the id alone, while the text changing to something else does not.

    The position is quantized by rounding, which means the guarantee is honest but
    bounded: a shift is invisible as long as the centre does not cross a cell
    boundary, so drift well under ``grid / 2`` is safe and a centre sitting right on a
    boundary is not. That is the price of a single deterministic id with no memory of
    previous observations - callers that need more should match on ``box`` and text
    themselves.

    The id is NOT unique across screens, and two genuinely identical controls in the
    same cell (two blank checkboxes) share one. It is a hint for "same thing as
    before", not a primary key.
    """
    if grid <= 0:
        raise ValueError(f"stable_id grid must be positive, got {grid}")
    centre = element.box.center
    qx = int(round(centre.x / grid))
    qy = int(round(centre.y / grid))
    key = f"{element.kind.value}\x1f{normalize_text(element.text)}\x1f{qx},{qy}"
    return hashlib.blake2s(key.encode("utf-8"), digest_size=8).hexdigest()


def with_stable_ids(elements: Iterable[Element], *, overwrite: bool = False) -> list[Element]:
    """The same elements with ``stable_id`` filled in.

    Existing ids are kept unless ``overwrite`` is set, because a producer that knows a
    truer identity (a DOM id) should win over the geometric guess.
    """
    out: list[Element] = []
    for element in elements:
        if element.stable_id is None or overwrite:
            out.append(dataclasses.replace(element, stable_id=stable_id(element)))
        else:
            out.append(element)
    return out


# --------------------------------------------------------------------------------------
# Fuzzy text matching
# --------------------------------------------------------------------------------------

# Score bands, so the ordering is a property of the design and not of tuning luck:
# exact (4.0) > prefix (3.0-3.9) > substring (2.0-2.9) > fuzzy (1.0-1.9) > no match (0).
# Every literal hit therefore outranks every fuzzy one, however good the fuzzy one is.
_EXACT = 4.0
_PREFIX_BASE = 3.0
_SUBSTRING_BASE = 2.0
_FUZZY_BASE = 1.0
_FUZZY_SPAN = 0.9

#: Minimum similarity for a fuzzy match to count at all.
_FUZZY_FLOOR = 0.62


#: How many single-character errors are forgiven outright, by query length. OCR turning
#: "Submit" into "Subrnit" must still match, and on a short query the similarity ratio
#: alone is too harsh to allow it.
def _edit_allowance(length: int) -> int:
    if length <= 3:
        return 0
    if length <= 6:
        return 1
    if length <= 12:
        return 2
    return length // 6


def _levenshtein_within(a: str, b: str, limit: int) -> bool:
    """Whether ``a`` and ``b`` are at most ``limit`` single-character edits apart."""
    if limit < 0:
        return False
    if abs(len(a) - len(b)) > limit:
        return False
    if a == b:
        return True
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > limit:  # every alignment through this row is already too expensive
            return False
        previous = current
    return previous[-1] <= limit


def _candidates(query: str, text: str) -> list[str]:
    """Substrings of ``text`` worth comparing against ``query``.

    A query is usually a fragment of a longer OCR line ("Submit" inside "Subrnit order
    now"), so comparing against the whole line understates the match badly. Word
    windows the size of the query - and its neighbours - recover it.
    """
    words = text.split(" ")
    out = [text]
    if len(words) <= 1:
        return out
    out.extend(words)
    if len(words) > 24:  # keep this linear on a text-heavy screen
        return out
    span = max(1, len(query.split(" ")))
    for size in {span, span + 1, max(1, span - 1)}:
        if size >= len(words):
            continue
        for start in range(len(words) - size + 1):
            out.append(" ".join(words[start : start + size]))
    return out


def _fuzzy_similarity(query: str, text: str) -> float:
    """Best similarity in ``0.0..1.0`` between ``query`` and any sensible piece of ``text``."""
    allowance = _edit_allowance(len(query))
    best = 0.0
    for candidate in _candidates(query, text):
        if not candidate:
            continue
        ratio = difflib.SequenceMatcher(None, query, candidate).ratio()
        if allowance and ratio < 0.9 and _levenshtein_within(query, candidate, allowance):
            # "a wrong character or two" is a match even when the ratio disagrees.
            ratio = 0.9
        if ratio > best:
            best = ratio
            if best >= 1.0:
                break
    return best


def _text_score(query: str, text: str, *, fuzzy: bool) -> float:
    """How well ``text`` answers ``query``; both already normalized. ``0.0`` for no match."""
    if not query or not text:
        return 0.0
    if text == query:
        return _EXACT
    if text.startswith(query):
        return _PREFIX_BASE + 0.9 * len(query) / len(text)
    if query in text:
        return _SUBSTRING_BASE + 0.9 * len(query) / len(text)
    if not fuzzy:
        return 0.0
    # A query LONGER than the text ("Send Reminders Now" against the "Send Reminders"
    # button) lands here too: similarity is symmetric, so it needs no special case, and
    # it correctly scores below a literal hit.
    similarity = _fuzzy_similarity(query, text)
    if similarity < _FUZZY_FLOOR:
        return 0.0
    return _FUZZY_BASE + _FUZZY_SPAN * (similarity - _FUZZY_FLOOR) / (1.0 - _FUZZY_FLOOR)


# --------------------------------------------------------------------------------------
# Kinds: words that name them, and how specific they are
# --------------------------------------------------------------------------------------

_KIND_WORDS: dict[ElementKind, frozenset[str]] = {
    ElementKind.button: frozenset({"button", "btn", "cta"}),
    ElementKind.text_field: frozenset(
        {"field", "input", "textbox", "textfield", "box", "search", "searchbox", "entry"}
    ),
    ElementKind.checkbox: frozenset({"checkbox", "check", "tickbox"}),
    ElementKind.radio: frozenset({"radio", "radiobutton"}),
    ElementKind.link: frozenset({"link", "hyperlink", "anchor"}),
    ElementKind.icon: frozenset({"icon", "glyph"}),
    ElementKind.menu: frozenset({"menu", "dropdown", "select"}),
    ElementKind.tab: frozenset({"tab"}),
    ElementKind.row: frozenset({"row", "item", "record", "line"}),
    ElementKind.image: frozenset({"image", "picture", "photo", "thumbnail"}),
    ElementKind.text: frozenset({"text", "label", "heading", "title", "caption"}),
    ElementKind.other: frozenset(),
}

_WORD_KINDS: dict[str, set[ElementKind]] = {}
for _kind, _kws in _KIND_WORDS.items():
    for _kw in _kws:
        _WORD_KINDS.setdefault(_kw, set()).add(_kind)

#: How specific a kind is. Merging prefers the higher number, because "button" says
#: more about what a click will do than "text" or "other" does.
_SPECIFICITY: dict[ElementKind, int] = {
    ElementKind.other: 0,
    ElementKind.text: 1,
    ElementKind.image: 2,
    ElementKind.row: 3,
    ElementKind.icon: 3,
    ElementKind.link: 4,
    ElementKind.menu: 4,
    ElementKind.tab: 4,
    ElementKind.button: 5,
    ElementKind.text_field: 5,
    ElementKind.checkbox: 5,
    ElementKind.radio: 5,
}

#: Kind words that are also plausible label text, so they stay in the text query as
#: well as hinting a kind: "search field" should match an input whose text is "Search".
_KIND_WORDS_ALSO_TEXT = frozenset(
    {"search", "select", "menu", "tab", "link", "item", "title", "check", "photo"}
)

#: Words that say nothing a screenshot can be matched against, so they only dilute a
#: match: filler, verbs of clicking, and position words this index cannot evaluate
#: ("the button at the top" - it has no idea which is the top one).
_STOPWORDS = frozenset(
    {
        "a",
        "above",
        "an",
        "and",
        "at",
        "below",
        "beside",
        "bottom",
        "center",
        "centre",
        "click",
        "first",
        "for",
        "in",
        "into",
        "last",
        "left",
        "lower",
        "me",
        "middle",
        "my",
        "near",
        "next",
        "of",
        "on",
        "onto",
        "please",
        "press",
        "right",
        "tap",
        "that",
        "the",
        "then",
        "this",
        "to",
        "top",
        "upper",
        "with",
    }
)


# --------------------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------------------


class ElementIndex:
    """A queryable, immutable view over the elements of one observation.

    Implements :class:`skillweaver.contracts.ElementIndex`. Construction sorts the
    elements into reading order once; every query is a scan over that, which is the
    right trade at the few-hundred-elements-per-screen scale this runs at and keeps
    ranking easy to reason about.

    Text matching is deliberately forgiving, because the text comes from OCR: case,
    surrounding whitespace, edge punctuation and one or two wrong characters all still
    match, and matches are banded so a literal hit always outranks a fuzzy one.
    """

    __slots__ = ("_elements", "_normalized", "_rank")

    def __init__(self, elements: Sequence[Element] | Iterable[Element] = ()) -> None:
        self._elements: tuple[Element, ...] = tuple(reading_order(elements))
        self._normalized: tuple[str, ...] = tuple(normalize_text(e.text) for e in self._elements)
        self._rank: dict[int, int] = {id(e): i for i, e in enumerate(self._elements)}

    # -- plumbing ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self):
        return iter(self._elements)

    def __repr__(self) -> str:
        return f"ElementIndex({len(self._elements)} elements)"

    @property
    def elements(self) -> tuple[Element, ...]:
        """The indexed elements in reading order."""
        return self._elements

    def _order(self, element: Element) -> int:
        return self._rank.get(id(element), 0)

    def _ranked(self, scored: list[tuple[float, Element]]) -> list[Element]:
        """Sort by score descending, breaking ties by reading order for determinism."""
        scored.sort(key=lambda pair: (-pair[0], self._order(pair[1])))
        return [element for _, element in scored]

    # -- contracts.ElementIndex ----------------------------------------------------

    def all(self) -> list[Element]:
        """Every element, in reading order."""
        return list(self._elements)

    def by_kind(self, kind: ElementKind) -> list[Element]:
        """Elements of exactly ``kind``, in reading order."""
        return [e for e in self._elements if e.kind == kind]

    def find_text(
        self, query: str, kind: ElementKind | None = None, fuzzy: bool = True
    ) -> list[Element]:
        """Elements whose text matches ``query``, best match first.

        Case-insensitive and whitespace-insensitive. Exact matches come first, then
        prefixes, then substrings; with ``fuzzy=True`` near matches follow, ranked
        strictly below every literal one. ``kind`` restricts the search.
        """
        q = normalize_text(query)
        if not q:
            return []
        scored: list[tuple[float, Element]] = []
        for element, text in zip(self._elements, self._normalized, strict=True):
            if kind is not None and element.kind != kind:
                continue
            score = _text_score(q, text, fuzzy=fuzzy)
            if score > 0:
                scored.append((score, element))
        return self._ranked(scored)

    def nearest(self, point: Point, kind: ElementKind | None = None) -> list[Element]:
        """Elements by distance from ``point`` to their box, nearest first.

        Distance is ``0`` for every element containing the point; those are then
        ordered smallest box first, so the innermost control wins over its container.
        """
        scored: list[tuple[float, float, Element]] = []
        for element in self._elements:
            if kind is not None and element.kind != kind:
                continue
            scored.append((_distance(element.box, point), float(element.box.area), element))
        scored.sort(key=lambda t: (t[0], t[1], self._order(t[2])))
        return [element for _, _, element in scored]

    def containing(self, point: Point) -> list[Element]:
        """Elements whose box contains ``point``, smallest (most specific) box first."""
        hits = [e for e in self._elements if e.box.contains(point)]
        hits.sort(key=lambda e: (e.box.area, self._order(e)))
        return hits

    def best(self, description: str) -> list[Element]:
        """Elements matching a free-form description, best first.

        The description is split into words that name a kind ("button", "field",
        "link") and words that should appear in the text. Naming a kind both rewards
        that kind and penalizes the others, which is what makes ``"search field"`` pick
        the input rather than the adjacent Search button. When the description names no
        text at all ("the button"), kind alone is enough to be listed.

        The first result is a guess, not an answer; callers should still check it.
        """
        tokens = [w for w in _words(description) if w]
        hinted: set[ElementKind] = set()
        residual: list[str] = []
        for token in tokens:
            kinds = _WORD_KINDS.get(token)
            if kinds:
                hinted |= kinds
                if token in _KIND_WORDS_ALSO_TEXT:
                    residual.append(token)
                continue
            if token not in _STOPWORDS:
                residual.append(token)
        query = " ".join(residual)

        scored: list[tuple[float, Element]] = []
        for element, text in zip(self._elements, self._normalized, strict=True):
            text_part = _text_score(query, text, fuzzy=True) if query else 0.0
            overlap = _word_overlap(residual, text)
            kind_match = element.kind in hinted
            if query:
                # The caller named text, so kind alone is not evidence of a match.
                if text_part <= 0 and overlap <= 0:
                    continue
            elif not kind_match:
                continue
            score = text_part + 1.2 * overlap
            if hinted:
                score += 1.6 if kind_match else -1.6
            score += 0.15 * element.confidence + 0.05 * _SPECIFICITY[element.kind]
            scored.append((score, element))
        return self._ranked(scored)

    # -- beyond the Protocol -------------------------------------------------------

    def by_id(self, stable: str) -> list[Element]:
        """Elements carrying ``stable_id == stable``, in reading order.

        Usually zero or one; more than one means two indistinguishable controls sit in
        the same :data:`STABLE_ID_GRID` cell.
        """
        return [e for e in self._elements if e.stable_id == stable]


def build_index(elements: Iterable[Element]) -> ElementIndex:
    """``ElementIndex(elements)``, as a function for use as a factory argument."""
    return ElementIndex(elements)


def _word_overlap(query_words: Sequence[str], text: str) -> float:
    """Fraction of ``query_words`` present in ``text``, allowing per-word OCR noise."""
    if not query_words:
        return 0.0
    text_words = text.split(" ") if text else []
    if not text_words:
        return 0.0
    hits = 0.0
    for word in query_words:
        if word in text_words:
            hits += 1.0
            continue
        allowance = _edit_allowance(len(word))
        if allowance and any(_levenshtein_within(word, other, allowance) for other in text_words):
            hits += 0.8
    return hits / len(query_words)


def _distance(box: Box, point: Point) -> float:
    """Euclidean distance from ``point`` to ``box``; ``0.0`` when the point is inside."""
    dx = max(box.x - point.x, 0, point.x - (box.x + max(box.w, 1) - 1))
    dy = max(box.y - point.y, 0, point.y - (box.y + max(box.h, 1) - 1))
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------------------
# Merging sources
# --------------------------------------------------------------------------------------

#: Default IoU above which two boxes are taken to be the same thing.
_DEFAULT_IOU = 0.55

#: Default fraction of the SMALLER box that must sit inside the larger for a
#: containment merge. This is the YOLO-button-plus-OCR-label case: their IoU is low
#: because the text box is much smaller, yet they are plainly one control.
_DEFAULT_CONTAINMENT = 0.75

#: How much a source is trusted to know an element's text.
_TEXT_TRUST: dict[ElementSource, int] = {
    ElementSource.dom: 3,
    ElementSource.ocr: 2,
    ElementSource.merged: 1,
    ElementSource.yolo: 0,
}

MERGED_TEXT_LIMIT = 160
"""Characters kept when a fused control's lines are joined; see :func:`_fused_text`.

Long enough for the identifying part of a list row - who it is from, what it is
about - and short enough that a screen of sixteen such rows does not fill a prompt
with body copy. Reading order puts the identifying lines first, so what a truncation
drops is the tail of the longest line.
"""


def overlap_ratio(a: Box, b: Box) -> float:
    """Intersection area over the SMALLER box's area, in ``0.0..1.0``.

    Unlike :meth:`Box.iou` this does not shrink when one box is much larger than the
    other, so it answers "is this little text box inside that button" - the question
    IoU is bad at. ``0.0`` when either box is degenerate.
    """
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    smaller = min(a.area, b.area)
    if smaller <= 0:
        return 0.0
    return (ix * iy) / smaller


def _same_thing(a: Element, b: Element, iou: float, containment: float) -> bool:
    if a.box.iou(b.box) >= iou:
        return True
    if overlap_ratio(a.box, b.box) < containment:
        return False
    # Containment only merges a vague thing into a specific one (a text line into a
    # button). Two specific controls that happen to nest - a checkbox inside a row -
    # stay separate, because both are real click targets.
    lo, hi = sorted((_SPECIFICITY[a.kind], _SPECIFICITY[b.kind]))
    return lo <= _SPECIFICITY[ElementKind.text] < hi


def merge_elements(
    *groups: Iterable[Element],
    iou: float = _DEFAULT_IOU,
    containment: float = _DEFAULT_CONTAINMENT,
) -> list[Element]:
    """Fuse element lists from different producers into one, de-duplicated list.

    Call it with one list per source - ``merge_elements(detector.detect(shot),
    reader.read(shot))`` - or with a single already-concatenated list.

    Two candidates are taken to be the same control when their boxes overlap enough:
    either ``IoU >= iou``, or ``iou``-too-low-but the smaller box sits at least
    ``containment`` inside the larger AND one of the two kinds is vague. That second
    rule is the whole point of merging: a YOLO ``button`` box and the OCR ``text`` line
    printed inside it have an IoU around 0.3 and are obviously one button.

    For each group of duplicates the survivor takes the most specific kind - and, at
    equal specificity, the highest confidence - from the group, that member's box as
    the click target, the best text available (preferring a ground-truth or OCR source,
    then the longest string), the highest confidence seen, a freshly computed
    :func:`stable_id`, and ``source=ElementSource.merged``.

    An element that matched nothing is passed through unchanged, keeping its own
    ``source`` - ``merged`` means "fused from more than one candidate", so claiming it
    for a lone OCR line would throw away where it came from. Its ``stable_id`` is
    filled in if it had none.

    The result is in reading order, ready to be an ``Observation.elements``.
    """
    pool = [element for group in groups for element in group]
    if not pool:
        return []

    # Seeding clusters with the most specific, most confident candidate first means the
    # representative is the best one and a single comparison per cluster suffices.
    seeded = sorted(
        pool,
        key=lambda e: (-_SPECIFICITY[e.kind], -e.confidence, -e.box.area, e.box.y, e.box.x),
    )
    clusters: list[list[Element]] = []
    for element in seeded:
        for cluster in clusters:
            if _same_thing(cluster[0], element, iou, containment):
                cluster.append(element)
                break
        else:
            clusters.append([element])

    merged: list[Element] = []
    for cluster in clusters:
        if len(cluster) == 1:
            only = cluster[0]
            merged.append(
                only
                if only.stable_id is not None
                else dataclasses.replace(only, stable_id=stable_id(only))
            )
            continue
        seed = cluster[0]
        fused = Element(
            box=seed.box,
            kind=seed.kind,
            text=_fused_text(cluster, seed.text),
            confidence=max(e.confidence for e in cluster),
            stable_id=None,
            source=ElementSource.merged,
        )
        merged.append(dataclasses.replace(fused, stable_id=stable_id(fused)))
    return reading_order(merged)


def _fused_text(cluster: Sequence[Element], fallback: str) -> str:
    """Everything a fused control says, in reading order, from its best source.

    A control in a real application is usually several lines: a message row prints a
    sender, a subject and a preview, a card prints a title and a caption. Each arrives
    as its own OCR line and they all fuse into the one control the user clicks, so the
    question is which of them the merged element should say.

    Keeping only the longest - which is what this did first - throws the other lines
    away, and the ones thrown away are the identifying ones: the longest line in a
    message row is the body preview, so an agent asked to open "the message from
    Billing titled 'Invoice 4471 is ready'" could search the screen and find neither
    the sender nor the subject, because only the preview survived. Every line is on
    screen and every line is something a person would name the control by, so the
    merged element says all of them, in the order they are read.

    Only the most trusted source present contributes: with DOM ground truth there is
    no reason to append OCR's guess at the same words, and mixing the two would make
    every control read its own label twice.
    """
    texted = [e for e in cluster if normalize_text(e.text)]
    if not texted:
        return fallback
    best = max(_TEXT_TRUST[e.source] for e in texted)
    lines: list[str] = []
    seen: set[str] = set()
    for element in reading_order([e for e in texted if _TEXT_TRUST[e.source] == best]):
        normalized = normalize_text(element.text)
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(element.text.strip())
    joined = " ".join(line for line in lines if line)
    if len(joined) <= MERGED_TEXT_LIMIT:
        return joined
    return joined[: MERGED_TEXT_LIMIT - 1].rstrip() + "…"
