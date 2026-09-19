"""The properties evaluation runs depend on: exact reset, and stable pixels."""

import json
import urllib.request

from helpers import settle


def reset(base_url):
    return json.loads(urllib.request.urlopen(base_url + "/__reset").read())


def dirty_everything(page):
    page.fill("[data-testid=board-search]", "gauge")
    settle(page)
    page.fill("[data-testid=board-search]", "")
    settle(page)
    page.click("[data-testid=move-btn-t01]")
    settle(page)
    page.click("[data-testid=move-to-t01-inprogress]")
    settle(page)
    page.click("[data-testid=card-title-t02]")
    settle(page)
    page.fill("[data-testid=detail-title-input]", "Mutated title")
    settle(page)
    page.click("[data-testid=detail-save-title]")
    settle(page)
    page.fill("[data-testid=detail-comment-input]", "Mutating comment")
    settle(page)
    page.click("[data-testid=detail-add-comment]")
    settle(page)
    page.select_option("[data-testid=detail-assign]", "Ada Lindgren")
    settle(page)
    page.click("[data-testid=detail-close]")
    settle(page)
    page.click("[data-testid=new-ticket-btn]")
    settle(page)
    page.fill("[data-testid=compose-title]", "A brand new ticket")
    settle(page)
    page.click("[data-testid=compose-create]")
    settle(page)
    page.click("[data-testid=archive-done-btn]")
    settle(page)
    page.click("[data-testid=archive-confirm]")
    settle(page)
    page.select_option("[data-testid=filter-priority]", "High")
    settle(page)


def test_reset_restores_the_exact_seed_state(page, base_url, state, seed):
    dirty_everything(page)

    s = state()
    assert s["archived"], "precondition: the app really was mutated"
    assert s["board"]["counters"]["ticket"] == 15
    assert any(t["key"] == "LAN-15" for t in s["board"]["tickets"])

    after = reset(base_url)["state"]

    data = {k: after[k] for k in ("meta", "board", "archived")}
    assert data == seed
    assert after["ui"]["search"] == ""
    assert after["ui"]["assignee"] == "all"
    assert after["ui"]["priority"] == "all"
    assert after["ui"]["openId"] is None
    assert after["ui"]["moveMenuFor"] is None
    assert after["ui"]["compose"]["open"] is False
    assert after["ui"]["dialogOpen"] is False
    assert after["ui"]["banner"] is None
    assert after["ui"]["draft"] == {"title": "", "comment": ""}


def test_reset_is_idempotent(base_url):
    assert reset(base_url)["state"] == reset(base_url)["state"]


def test_state_endpoint_matches_the_seed_on_a_fresh_server(page, state, seed):
    s = state()
    assert {k: s[k] for k in ("meta", "board", "archived")} == seed


def test_the_same_screen_screenshots_identically_twice(page):
    """No animation, no transition, no blinking caret, no drifting clock."""
    # the plain board
    first = page.screenshot()
    assert first == page.screenshot(), "board is not pixel-stable"

    # the detail panel
    page.click("[data-testid=card-title-t02]")
    settle(page)
    first = page.screenshot()
    assert first == page.screenshot(), "detail panel is not pixel-stable"

    # the new-ticket modal
    page.click("[data-testid=new-ticket-btn]")
    settle(page)
    first = page.screenshot()
    assert first == page.screenshot(), "new-ticket modal is not pixel-stable"


def test_reaching_a_screen_twice_produces_the_same_pixels(page, browser, base_url):
    """Reset, walk a flow, screenshot - then do it again and compare bytes."""

    def walk():
        urllib.request.urlopen(base_url + "/__reset").read()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        pg = ctx.new_page()
        pg.goto(base_url + "/")
        pg.wait_for_selector("body[data-ready='1']")
        pg.fill("[data-testid=board-search]", "schema")
        settle(pg)
        pg.click("[data-testid=card-title-t11]")
        settle(pg)
        shot = pg.screenshot()
        ctx.close()
        return shot

    assert walk() == walk()


def test_nothing_renders_a_live_clock(page):
    """Relative times and today's date would make two runs disagree."""
    import datetime

    today = datetime.date.today()
    page.click("[data-testid=card-title-t02]")
    settle(page)
    text = page.inner_text("body")
    assert "ago" not in text.split()
    assert str(today.year) not in text
