"""Board flows: columns, the move menu, filters, and the detail panel."""

from helpers import settle, ticket


def col_count(page, col):
    return page.locator(f"[data-testid=colcount-{col}]").inner_text()


def test_the_four_columns_show_the_seed_spread(page, seed):
    for col, expected in (("backlog", "5"), ("inprogress", "4"), ("review", "2"), ("done", "3")):
        assert col_count(page, col) == expected
    assert page.locator("[data-testid^=card-title-]").count() == len(seed["board"]["tickets"]) == 14


def test_move_a_ticket_between_columns_via_the_move_menu(page, state):
    assert page.locator("[data-testid=move-menu-t02]").count() == 0

    page.click("[data-testid=move-btn-t02]")
    settle(page)
    assert page.locator("[data-testid=move-menu-t02]").is_visible()
    # the menu lists exactly the other three columns, by visible name
    labels = page.locator("[data-testid=move-menu-t02] .menuitem").all_inner_texts()
    assert labels == ["Backlog", "Review", "Done"]

    page.click("[data-testid=move-to-t02-review]")
    settle(page)

    s = state()
    assert ticket(s, "t02")["column"] == "review"
    assert s["ui"]["moveMenuFor"] is None
    assert s["ui"]["banner"] == "Moved LAN-2 to Review"
    assert col_count(page, "inprogress") == "3"
    assert col_count(page, "review") == "3"
    assert page.locator("[data-testid=board-banner]").inner_text().startswith("Moved LAN-2")


def test_the_move_menu_toggles_closed_without_moving(page, state):
    page.click("[data-testid=move-btn-t02]")
    settle(page)
    page.click("[data-testid=move-btn-t02]")
    settle(page)
    assert page.locator("[data-testid=move-menu-t02]").count() == 0
    assert ticket(state(), "t02")["column"] == "inprogress"


def test_search_narrows_every_column(page, state):
    page.fill("[data-testid=board-search]", "gauge")
    settle(page)
    assert page.locator("[data-testid^=card-title-]").count() == 1
    assert col_count(page, "inprogress") == "1"
    assert col_count(page, "backlog") == "0"
    assert state()["ui"]["search"] == "gauge"
    assert page.locator("[data-testid=colstack-backlog] .col-empty").inner_text() == "No tickets"


def test_filter_by_assignee(page, state):
    page.select_option("[data-testid=filter-assignee]", "Ada Lindgren")
    settle(page)
    s = state()
    assert s["ui"]["assignee"] == "Ada Lindgren"
    assert page.locator("[data-testid^=card-title-]").count() == 2  # t05, t10
    assert page.locator("[data-testid=card-title-t05]").count() == 1
    assert page.locator("[data-testid=card-title-t10]").count() == 1


def test_filter_by_priority(page, state):
    page.select_option("[data-testid=filter-priority]", "High")
    settle(page)
    assert state()["ui"]["priority"] == "High"
    assert page.locator("[data-testid^=card-title-]").count() == 4  # t02 t05 t09 t11


def test_clicking_a_card_title_opens_the_detail_panel(page, state):
    assert page.locator("[data-testid=detail]").count() == 0
    page.click("[data-testid=card-title-t09]")
    settle(page)

    assert page.locator("[data-testid=detail-key]").inner_text() == "LAN-9"
    assert page.locator("[data-testid=detail-status]").inner_text() == "Review"
    assert page.locator("[data-testid=detail-priority]").inner_text() == "High"
    assert "clock drifts" in page.locator("[data-testid=detail-desc]").inner_text()
    assert page.locator("[data-testid=comment-c05]").count() == 1
    assert state()["ui"]["openId"] == "t09"

    page.click("[data-testid=detail-close]")
    settle(page)
    assert page.locator("[data-testid=detail]").count() == 0
    assert state()["ui"]["openId"] is None


def test_rename_a_ticket_from_the_detail_panel(page, state):
    page.click("[data-testid=card-title-t03]")
    settle(page)
    assert page.input_value("[data-testid=detail-title-input]") == \
        "Rewrite onboarding copy for the pairing step"

    page.fill("[data-testid=detail-title-input]", "Pairing step copy, second pass")
    settle(page)
    page.click("[data-testid=detail-save-title]")
    settle(page)

    s = state()
    assert ticket(s, "t03")["title"] == "Pairing step copy, second pass"
    assert s["ui"]["banner"] == "Renamed LAN-3 to Pairing step copy, second pass"
    assert page.locator("[data-testid=card-title-t03]").inner_text() == "Pairing step copy, second pass"


def test_save_title_is_disabled_when_the_draft_is_empty(page):
    page.click("[data-testid=card-title-t03]")
    settle(page)
    page.fill("[data-testid=detail-title-input]", "")
    settle(page)
    assert page.is_disabled("[data-testid=detail-save-title]")


def test_add_a_comment_from_the_detail_panel(page, state):
    page.click("[data-testid=card-title-t05]")
    settle(page)
    assert page.is_disabled("[data-testid=detail-add-comment]")

    page.fill("[data-testid=detail-comment-input]", "Plan tier is now part of the key.")
    settle(page)
    page.click("[data-testid=detail-add-comment]")
    settle(page)

    s = state()
    last = ticket(s, "t05")["comments"][-1]
    assert last == {
        "id": "c07",
        "author": "Rowan Ellery",
        "text": "Plan tier is now part of the key.",
        "date": "Mar 12",
    }
    assert s["ui"]["draft"]["comment"] == ""
    assert s["ui"]["banner"] == "Comment added to LAN-5"
    assert page.locator("[data-testid=comment-c07]").count() == 1


def test_reassign_from_the_detail_panel(page, state):
    page.click("[data-testid=card-title-t07]")
    settle(page)
    page.select_option("[data-testid=detail-assign]", "Theo Barros")
    settle(page)

    s = state()
    assert ticket(s, "t07")["assignee"] == "Theo Barros"
    assert s["ui"]["banner"] == "Assigned LAN-7 to Theo Barros"
    assert page.locator("[data-testid=card-assignee-t07]").inner_text() == "Theo Barros"


def test_typed_text_survives_the_render_round_trip(page, state):
    """Every keystroke posts to the server; the queue must keep them in order."""
    text = "A fairly long comment typed in one go"
    page.click("[data-testid=card-title-t02]")
    settle(page)
    page.fill("[data-testid=detail-comment-input]", text)
    settle(page)
    assert page.input_value("[data-testid=detail-comment-input]") == text
    assert state()["ui"]["draft"]["comment"] == text


def test_the_bottom_card_is_reachable_at_1280x800(page, state):
    """With the detail panel open the columns narrow, titles wrap, and Backlog
    overflows the fold; its stack must scroll (overflow-y: auto), never clip,
    so the bottom-most card stays reachable by mouse wheel and click."""
    page.click("[data-testid=card-title-t01]")
    settle(page)
    assert page.evaluate(
        "() => { const s = document.querySelector('[data-testid=colstack-backlog]');"
        " return s.scrollHeight > s.clientHeight; }"
    )
    page.click("[data-testid=move-btn-t13]")  # bottom-most Backlog card
    settle(page)
    page.click("[data-testid=move-to-t13-review]")
    settle(page)
    assert ticket(state(), "t13")["column"] == "review"
