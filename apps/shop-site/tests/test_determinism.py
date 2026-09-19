"""The properties evaluation runs depend on: exact reset, and stable pixels."""

import json
import urllib.request

from helpers import add_two_products, fill_checkout_form, settle


def reset(base_url):
    return json.loads(urllib.request.urlopen(base_url + "/__reset").read())


def dirty_everything(page):
    page.fill("[data-testid=shop-search]", "lamp")
    settle(page)
    page.fill("[data-testid=shop-search]", "")
    settle(page)
    page.click("[data-testid=cat-Lighting]")
    settle(page)
    page.click("[data-testid=price-under25]")
    settle(page)
    page.select_option("[data-testid=sort-select]", "price-desc")
    settle(page)
    page.click("[data-testid=cat-All]")
    settle(page)
    page.click("[data-testid=price-any]")
    settle(page)
    add_two_products(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    page.click("[data-testid=qty-plus-p01]")
    settle(page)
    page.click("[data-testid=checkout-open]")
    settle(page)
    fill_checkout_form(page)
    page.click("[data-testid=form-save]")
    settle(page)
    page.click("[data-testid=place-order]")
    settle(page)
    page.click("[data-testid=confirm-order]")
    settle(page)


def test_reset_restores_the_exact_seed_state(page, base_url, state, seed):
    dirty_everything(page)

    s = state()
    assert s["orders"], "precondition: the app really was mutated"
    assert s["ui"]["banner"] is not None

    after = reset(base_url)["state"]

    data = {k: after[k] for k in ("meta", "catalog", "cart", "orders")}
    assert data == seed
    assert after["ui"]["screen"] == "shop"
    assert after["ui"]["search"] == ""
    assert after["ui"]["category"] == "All"
    assert after["ui"]["priceBand"] == "any"
    assert after["ui"]["sort"] == "featured"
    assert after["ui"]["cartOpen"] is False
    assert after["ui"]["checkoutOpen"] is False
    assert after["ui"]["dialogOpen"] is False
    assert after["ui"]["banner"] is None
    assert after["ui"]["form"] == {
        "name": "", "email": "", "address": "", "shipping": "standard", "save": False,
    }


def test_reset_is_idempotent(base_url):
    assert reset(base_url)["state"] == reset(base_url)["state"]


def test_state_endpoint_matches_the_seed_on_a_fresh_server(page, state, seed):
    s = state()
    assert {k: s[k] for k in ("meta", "catalog", "cart", "orders")} == seed


def test_the_same_screen_screenshots_identically_twice(page):
    """No animation, no transition, no blinking caret, no drifting clock."""
    first = page.screenshot()
    second = page.screenshot()
    assert first == second, "grid is not pixel-stable"

    page.click("[data-testid=add-p01]")
    settle(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    first = page.screenshot()
    second = page.screenshot()
    assert first == second, "cart drawer is not pixel-stable"


def test_reaching_a_screen_twice_produces_the_same_pixels(page, browser, base_url):
    """Reset, walk a flow, screenshot - then do it again and compare bytes."""

    def walk():
        urllib.request.urlopen(base_url + "/__reset").read()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(base_url + "/")
        pg.wait_for_selector("body[data-ready='1']")
        pg.click("[data-testid=cat-Kitchen]")
        settle(pg)
        pg.click("[data-testid=add-p15]")
        settle(pg)
        pg.click("[data-testid=cart-open]")
        settle(pg)
        shot = pg.screenshot()
        ctx.close()
        return shot

    assert walk() == walk()


def test_the_shop_states_do_not_look_alike(page):
    """A detector trained here needs states that fingerprint differently."""
    shots = {}
    shots["grid"] = page.screenshot()
    page.click("[data-testid=add-p01]")
    settle(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    shots["cart"] = page.screenshot()
    page.click("[data-testid=checkout-open]")
    settle(page)
    shots["checkout"] = page.screenshot()
    assert len(set(shots.values())) == 3


def test_nothing_renders_a_live_clock(page):
    """Relative times and today's date would make two runs disagree."""
    import datetime

    today = datetime.date.today()
    text = page.inner_text("body")
    assert "ago" not in text.split()
    assert str(today.year) not in text


def test_everything_is_reachable_at_1280x800(page):
    """The last card's Add button can be scrolled to and clicked with the mouse."""
    page.locator("[data-testid=add-p24]").scroll_into_view_if_needed()
    box = page.locator("[data-testid=add-p24]").bounding_box()
    assert box is not None
    assert box["y"] >= 0 and box["y"] + box["height"] <= 800
    page.click("[data-testid=add-p24]")
    settle(page)
    assert page.locator("[data-testid=cart-open]").inner_text() == "Cart (1)"
