"""Shared helpers for the sandbox smoke tests."""


def settle(page):
    """Block until every queued action has round-tripped and re-rendered.

    Every interaction posts to the server and re-renders from the response, so
    a test that asserts immediately after a click can outrun the round trip.
    """
    page.evaluate("() => window.settled()")
