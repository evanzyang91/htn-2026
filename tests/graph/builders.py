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
