"""Mail flows: search, multi-select, archive, label, reader, compose and send."""

from helpers import settle


def test_search_then_select_all_then_archive(page, state):
    page.fill("[data-testid=mail-search]", "plan")
    settle(page)
    assert page.locator("[data-testid^=msg-]").count() == 2

    page.click("[data-testid=select-all]")
    settle(page)
    assert page.locator("[data-testid=sel-count]").inner_text() == "2 selected"
    assert page.is_enabled("[data-testid=archive-btn]")

    page.click("[data-testid=archive-btn]")
    settle(page)

    s = state()
    archived = sorted(m["id"] for m in s["mail"]["messages"] if m["archived"])
    assert archived == ["m01", "m13"]
    assert s["ui"]["mail"]["banner"] == "Archived 2 conversations"
    assert s["ui"]["mail"]["selected"] == []
    assert page.locator("[data-testid=folder-archived] .count").inner_text() == "2"


def test_archive_button_is_disabled_until_something_is_selected(page):
    assert page.is_disabled("[data-testid=archive-btn]")
    assert page.is_disabled("[data-testid=label-btn]")
    page.click("[data-testid=msgcheck-m02]")
    settle(page)
    assert page.is_enabled("[data-testid=archive-btn]")
    assert page.is_enabled("[data-testid=label-btn]")


def test_opening_a_message_shows_the_reader_and_marks_it_read(page, state):
    assert page.locator("[data-testid=reader]").count() == 0
    assert [m for m in state()["mail"]["messages"] if m["id"] == "m05"][0]["unread"] is True

    page.click("[data-testid=msgopen-m05]")
    settle(page)

    assert page.locator("[data-testid=reader-subject]").inner_text() == "Lunch on Thursday?"
    assert "transit hub" in page.locator("[data-testid=reader-body]").inner_text()
    assert "split" in page.locator(".mail-body").get_attribute("class")

    s = state()
    assert s["ui"]["mail"]["openId"] == "m05"
    assert [m for m in s["mail"]["messages"] if m["id"] == "m05"][0]["unread"] is False

    page.click("[data-testid=reader-close]")
    settle(page)
    assert page.locator("[data-testid=reader]").count() == 0


def test_label_action_applies_to_the_selection(page, state):
    page.click("[data-testid=msgcheck-m06]")
    settle(page)
    page.click("[data-testid=label-btn]")
    settle(page)
    assert page.locator("[data-testid=label-menu]").is_visible()

    page.click("[data-testid=label-option-Work]")
    settle(page)

    s = state()
    m06 = [m for m in s["mail"]["messages"] if m["id"] == "m06"][0]
    assert m06["labels"] == ["Travel", "Work"]
    assert s["ui"]["mail"]["banner"] == "Labelled 1 conversation as Work"
    assert s["ui"]["mail"]["labelMenuOpen"] is False


def test_compose_and_send(page, state):
    page.click("[data-testid=compose-open]")
    settle(page)
    assert page.locator("[data-testid=compose-window]").is_visible()
    assert page.is_disabled("[data-testid=compose-send]")

    page.fill("[data-testid=compose-to]", "dana.whitfield@northwind.example")
    settle(page)
    page.fill("[data-testid=compose-subject]", "Capacity plan approved")
    settle(page)
    page.fill("[data-testid=compose-body]", "Signed off on the west row move.")
    settle(page)
    assert page.is_enabled("[data-testid=compose-send]")

    page.click("[data-testid=compose-send]")
    settle(page)

    s = state()
    assert s["ui"]["mail"]["compose"]["open"] is False
    assert s["mail"]["sent"] == [
        {
            "id": "s01",
            "to": "dana.whitfield@northwind.example",
            "subject": "Capacity plan approved",
            "body": "Signed off on the west row move.",
            "date": "Mar 12",
        }
    ]

    page.click("[data-testid=folder-sent]")
    settle(page)
    assert page.locator("[data-testid=sent-s01]").count() == 1


def test_typed_text_survives_the_render_round_trip(page, state):
    """Every keystroke posts to the server; the queue must keep them in order."""
    subject = "A fairly long subject line typed in one go"
    page.click("[data-testid=compose-open]")
    settle(page)
    page.fill("[data-testid=compose-subject]", subject)
    settle(page)
    assert page.input_value("[data-testid=compose-subject]") == subject
    assert state()["ui"]["mail"]["compose"]["subject"] == subject
