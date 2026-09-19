"""Shared helpers for the shop-site smoke tests."""


def settle(page):
    """Block until every queued action has round-tripped and re-rendered.

    Every interaction posts to the server and re-renders from the response, so
    a test that asserts immediately after a click can outrun the round trip.
    """
    page.evaluate("() => window.settled()")


def add_two_products(page):
    """Two distinct products in the cart: the shared starting point for cart flows."""
    page.click("[data-testid=add-p01]")
    settle(page)
    page.click("[data-testid=add-p13]")
    settle(page)


def fill_checkout_form(page):
    page.fill("[data-testid=form-name]", "Avery Quinn")
    settle(page)
    page.fill("[data-testid=form-email]", "avery.quinn@harbour.example")
    settle(page)
    page.fill("[data-testid=form-address]", "12 Pier Road, Port Town")
    settle(page)
