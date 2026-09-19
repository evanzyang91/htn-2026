"""The two modal flows: creating a ticket, and archiving the Done column."""

from helpers import settle, ticket


def test_create_a_new_ticket(page, state):
    page.click("[data-testid=new-ticket-btn]")
    settle(page)
    assert page.locator("[data-testid=compose-dialog]").is_visible()
    assert page.is_disabled("[data-testid=compose-create]")

    page.fill("[data-testid=compose-title]", "Audit the pairing logs")
    settle(page)
    page.fill("[data-testid=compose-description]", "Look for repeated failures per device.")
    settle(page)
    page.select_option("[data-testid=compose-assignee]", "June Nakagawa")
    settle(page)
    page.select_option("[data-testid=compose-priority]", "Low")
    settle(page)
    assert page.is_enabled("[data-testid=compose-create]")

    page.click("[data-testid=compose-create]")
    settle(page)

    s = state()
    assert s["ui"]["compose"]["open"] is False
    assert s["board"]["counters"]["ticket"] == 15
    new = ticket(s, "t15")
    assert new == {
        "id": "t15",
        "key": "LAN-15",
        "title": "Audit the pairing logs",
        "description": "Look for repeated failures per device.",
        "assignee": "June Nakagawa",
        "priority": "Low",
        "column": "backlog",
        "comments": [],
    }
    assert s["ui"]["banner"] == "Created LAN-15 in Backlog"
    assert page.locator("[data-testid=colcount-backlog]").inner_text() == "6"
    assert page.locator("[data-testid=card-title-t15]").inner_text() == "Audit the pairing logs"


def test_cancel_creates_nothing_and_clears_the_form(page, state):
    page.click("[data-testid=new-ticket-btn]")
    settle(page)
    page.fill("[data-testid=compose-title]", "Should not exist")
    settle(page)
    page.click("[data-testid=compose-cancel]")
    settle(page)

    s = state()
    assert s["ui"]["compose"]["open"] is False
    assert s["ui"]["compose"]["title"] == ""
    assert len(s["board"]["tickets"]) == 14

    # reopening starts from a blank form
    page.click("[data-testid=new-ticket-btn]")
    settle(page)
    assert page.input_value("[data-testid=compose-title]") == ""


def test_archive_done_asks_first_and_archives_on_confirm(page, state):
    page.click("[data-testid=archive-done-btn]")
    settle(page)
    assert page.locator("[data-testid=archive-dialog]").is_visible()
    assert page.locator("[data-testid=archive-dialog] h2").inner_text() == "Archive 3 tickets?"
    assert page.locator("[data-testid=archive-list] li").count() == 3

    page.click("[data-testid=archive-confirm]")
    settle(page)

    s = state()
    assert s["ui"]["dialogOpen"] is False
    assert sorted(t["id"] for t in s["archived"]) == ["t06", "t12", "t14"]
    assert not any(t["column"] == "done" for t in s["board"]["tickets"])
    assert s["ui"]["banner"] == "Archived 3 tickets"
    assert page.locator("[data-testid=colcount-done]").inner_text() == "0"
    assert page.is_disabled("[data-testid=archive-done-btn]")


def test_keep_them_cancels_the_archive(page, state):
    page.click("[data-testid=archive-done-btn]")
    settle(page)
    page.click("[data-testid=archive-cancel]")
    settle(page)

    s = state()
    assert s["ui"]["dialogOpen"] is False
    assert s["archived"] == []
    assert page.locator("[data-testid=colcount-done]").inner_text() == "3"


def test_archiving_a_ticket_open_in_the_detail_panel_closes_it(page, state):
    page.click("[data-testid=card-title-t06]")  # t06 sits in Done
    settle(page)
    page.click("[data-testid=archive-done-btn]")
    settle(page)
    page.click("[data-testid=archive-confirm]")
    settle(page)

    s = state()
    assert s["ui"]["openId"] is None
    assert page.locator("[data-testid=detail]").count() == 0
