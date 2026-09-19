"""Builders for graph tests: fingerprints with controllable similarity, and edges
with statistics set directly so routing arithmetic can be checked by hand."""

from __future__ import annotations

from datetime import UTC, datetime

from skillweaver.contracts import Click, Fingerprint, Point, Transition, UIState

DOMAIN = "shop.test"


def fp(value: str, **parts: str) -> Fingerprint:
    """A fingerprint whose ``parts`` decide how similar it is to another.

    ``Fingerprint.similarity`` is the fraction of agreeing part names over the
    union, so ``fp("a", url="u", layout="l", text="t")`` and a copy differing only
    in ``text`` score 2/3.
    """
    return Fingerprint(value=value, parts=parts)


def screen(name: str, **parts: str) -> Fingerprint:
    """A four-part fingerprint for screen ``name``: one changed part scores 0.75."""
    return fp(name, url=f"u:{name}", layout=f"l:{name}", text=f"t:{name}", chrome="c")


REAL_PARTS = 175
"""How many parts the SHIPPED fingerprinter emits for a live page, near enough.

A threshold is only meaningful against the shape of the signal it judges, so the tests
that exercise :data:`~skillweaver.graph.model.DEFAULT_MATCH_THRESHOLD` have to use a
fingerprint of roughly the real shape. A four-part ``screen`` can only express 0.0,
0.25, 0.5, 0.75 and 1.0, which says nothing about a cut of 0.26.
"""


def live_pair(agreeing: int, total: int = REAL_PARTS) -> tuple[Fingerprint, Fingerprint]:
    """Two fingerprints of realistic SHAPE that share ``agreeing`` of ``total`` parts.

    The shipped fingerprinter names each part by its CONTENT, so two screens share the
    parts they have in common and each keeps the rest; the similarity is therefore
    ``agreeing / (2 * total - agreeing)``, a Jaccard, not ``agreeing / total``. At
    ``total=175`` that puts 120 agreeing parts at 0.52 - a page pushed down by a notice -
    and 60 at 0.21 - two different pages built from one template.
    """
    shared = {f"band.shared.{i}": "1" for i in range(agreeing)}
    left = shared | {f"band.left.{i}": "1" for i in range(total - agreeing)}
    right = shared | {f"band.right.{i}": "1" for i in range(total - agreeing)}
    return fp(f"left:{agreeing}", **left), fp(f"right:{agreeing}", **right)


def state(
    fingerprint: Fingerprint, *, domain: str = DOMAIN, label: str = "", day: int = 1
) -> UIState:
    """A ``UIState`` with a deterministic ``first_seen`` so ordering is testable."""
    return UIState(
        fingerprint=fingerprint,
        domain=domain,
        label=label,
        first_seen=datetime(2026, 1, day, tzinfo=UTC),
    )


def click(x: int, y: int) -> Click:
    return Click(point=Point(x, y))


def edge(
    src: Fingerprint,
    dst: Fingerprint,
    *,
    actions: tuple = (),
    attempts: int = 1,
    successes: int = 1,
    mean_ms: float = 100.0,
    day: int | None = 1,
) -> Transition:
    """A ``Transition`` with its statistics set outright.

    Tests that care about routing arithmetic need exact ``attempts`` /
    ``successes`` / ``mean_ms``, which replaying ``observe_transition`` would only
    approximate.
    """
    return Transition(
        src=src,
        dst=dst,
        actions=actions or (click(10, 10),),
        attempts=attempts,
        successes=successes,
        mean_ms=mean_ms,
        last_verified=None if day is None or successes == 0 else datetime(2026, 1, day, tzinfo=UTC),
    )
