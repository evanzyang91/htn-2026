"""The properties evaluation runs depend on: exact reset, and stable pixels."""

import json
import urllib.request

from helpers import settle


def reset(base_url):
    return json.loads(urllib.request.urlopen(base_url + "/__reset").read())


def dirty_everything(page):
    page.fill("[data-testid=mail-search]", "invoice")
    settle(page)
    page.click("[data-testid=select-all]")
    settle(page)
    page.click("[data-testid=archive-btn]")
    settle(page)
    page.click("[data-testid=compose-open]")
    settle(page)
    page.fill("[data-testid=compose-to]", "someone@example.test")
    settle(page)
    page.click("[data-testid=compose-send]")
    settle(page)
    page.click("[data-testid=nav-records]")
    settle(page)
    page.click("[data-testid=editbtn-r01]")
    settle(page)
    page.fill("[data-testid=edit-input]", "Mutated")
    settle(page)
    page.click("[data-testid=edit-save]")
    settle(page)
    page.click("[data-testid=export-btn]")
    settle(page)
    page.click("[data-testid=nav-order]")
    settle(page)
    page.click("[data-testid=cuisine-Thai]")
    settle(page)
    page.click("[data-testid=restaurant-rest-kettle]")
    settle(page)
    page.click("[data-testid=dish-d-kettle-padthai]")
    settle(page)
    page.click("[data-testid=choice-portion-large]")
    settle(page)
    page.click("[data-testid=add-to-cart]")
    settle(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=tip-600]")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)
    page.click("[data-testid=order-confirm]")
    settle(page)
    page.click("[data-testid=nav-settings]")
    settle(page)
    page.click("[data-testid=set-autoArchive]")
    settle(page)
    page.click("[data-testid=settings-save]")
    settle(page)
    page.click("[data-testid=confirm-save]")
    settle(page)


def test_reset_restores_the_exact_seed_state(page, base_url, state, seed):
    dirty_everything(page)

    s = state()
    assert s["mail"]["sent"], "precondition: the app really was mutated"
    assert s["records"]["exports"]
    assert s["order"]["orders"], "precondition: an order really was placed"
    assert s["settings"]["savedCount"] == 1

    after = reset(base_url)["state"]

    data = {k: after[k] for k in ("meta", "mail", "records", "order", "settings")}
    assert data == seed
    assert after["ui"]["screen"] == "mail"
    assert after["ui"]["mail"]["search"] == ""
    assert after["ui"]["mail"]["selected"] == []
    assert after["ui"]["mail"]["compose"]["open"] is False
    assert after["ui"]["records"]["filter"] == ""
    assert after["ui"]["order"]["view"] == "browse"
    assert after["ui"]["order"]["cuisine"] == ""
    assert after["ui"]["order"]["dialogOpen"] is False
    assert after["ui"]["settings"]["dialogOpen"] is False
    assert after["ui"]["settings"]["success"] is False


def test_reset_is_idempotent(base_url):
    assert reset(base_url)["state"] == reset(base_url)["state"]


def test_state_endpoint_matches_the_seed_on_a_fresh_server(page, state, seed):
    s = state()
    assert {k: s[k] for k in ("meta", "mail", "records", "order", "settings")} == seed


def test_the_same_screen_screenshots_identically_twice(page):
    """No animation, no transition, no blinking caret, no drifting clock."""
    for testid in ("nav-mail", "nav-records", "nav-order", "nav-settings"):
        page.click(f"[data-testid={testid}]")
        settle(page)
        first = page.screenshot()
        second = page.screenshot()
        assert first == second, f"{testid} is not pixel-stable"


def test_reaching_a_screen_twice_produces_the_same_pixels(page, browser, base_url):
    """Reset, walk a flow, screenshot - then do it again and compare bytes."""

    def walk():
        urllib.request.urlopen(base_url + "/__reset").read()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(base_url + "/")
        pg.wait_for_selector("body[data-ready='1']")
        pg.fill("[data-testid=mail-search]", "plan")
        settle(pg)
        pg.click("[data-testid=select-all]")
        settle(pg)
        shot = pg.screenshot()
        ctx.close()
        return shot

    assert walk() == walk()


def test_the_ordering_flow_reaches_the_same_pixels_after_a_reset(page, browser, base_url):
    """The longest state-changing flow in the app, twice, byte for byte.

    This is the property the admission gate leans on: a skill is only stored
    after being RE-RUN from its recorded starting screen, so an ordering task -
    which places an order and cannot otherwise be undone - is only learnable
    because /__reset makes the second run see exactly the first run's world.
    """

    def walk():
        urllib.request.urlopen(base_url + "/__reset").read()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(base_url + "/")
        pg.wait_for_selector("body[data-ready='1']")
        for testid in (
            "nav-order",
            "cuisine-Thai",
            "restaurant-rest-kettle",
            "dish-d-kettle-padthai",
            "choice-portion-large",
            "add-to-cart",
            "ord-tab-cart",
            "checkout-btn",
            "order-confirm",
        ):
            pg.click(f"[data-testid={testid}]")
            settle(pg)
        shot = pg.screenshot()
        ctx.close()
        return shot

    assert walk() == walk()


def test_the_four_screens_do_not_look_alike(page):
    """A detector trained here needs screens that fingerprint differently."""
    shots = {}
    for name in ("mail", "records", "order", "settings"):
        page.click(f"[data-testid=nav-{name}]")
        settle(page)
        shots[name] = page.screenshot()
    assert len(set(shots.values())) == 4


def test_nothing_renders_a_live_clock(page):
    """Relative times and today's date would make two runs disagree."""
    import datetime

    today = datetime.date.today()
    for name in ("mail", "records", "order", "settings"):
        page.click(f"[data-testid=nav-{name}]")
        settle(page)
        text = page.inner_text("body")
        assert "ago" not in text
        assert str(today.year) not in text
