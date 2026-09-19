"""Shared helpers for the board sandbox smoke tests."""


def settle(page):
    """Block until every queued action has round-tripped and re-rendered.

    Every interaction posts to the server and re-renders from the response, so
    a test that asserts immediately after a click can outrun the round trip.
    """
    page.evaluate("() => window.settled()")


def ticket(state, tid):
    """The ticket with this id, from a /__state snapshot."""
    return next(t for t in state["board"]["tickets"] if t["id"] == tid)
