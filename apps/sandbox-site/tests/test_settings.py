"""Settings: the multi-step save with its confirm-or-cancel branch."""

from helpers import settle


def goto_settings(page):
    page.click("[data-testid=nav-settings]")
    settle(page)


def make_two_changes(page):
    page.click("[data-testid=set-notifyDesktop]")
    settle(page)
    page.select_option("[data-testid=set-timezone]", "America/Toronto")
    settle(page)


def test_save_is_disabled_until_something_changes(page):
    goto_settings(page)
    assert page.is_disabled("[data-testid=settings-save]")
    assert page.is_disabled("[data-testid=settings-discard]")
    assert page.locator("[data-testid=settings-dirty]").inner_text() == "No unsaved changes"


def test_editing_fields_does_not_save_anything_yet(page, state, seed):
    goto_settings(page)
    make_two_changes(page)
    assert page.locator("[data-testid=settings-dirty]").inner_text() == "2 unsaved changes"
    assert page.is_enabled("[data-testid=settings-save]")

    s = state()
    assert s["settings"]["timezone"] == seed["settings"]["timezone"]
    assert s["settings"]["notifyDesktop"] == seed["settings"]["notifyDesktop"]
    assert s["settings"]["savedCount"] == 0
    assert s["ui"]["settings"]["draft"]["timezone"] == "America/Toronto"


def test_save_opens_a_confirmation_dialog_listing_the_changes(page, state):
    goto_settings(page)
    make_two_changes(page)
    page.click("[data-testid=settings-save]")
    settle(page)

    assert page.locator("[data-testid=confirm-dialog]").is_visible()
    assert page.locator("[data-testid=scrim]").is_visible()
    assert state()["ui"]["settings"]["dialogOpen"] is True
    items = [li.inner_text() for li in page.locator("[data-testid=confirm-list] li").all()]
    assert items == ["Time zone: UTC → America/Toronto", "Desktop notifications: Off → On"]


def test_cancel_branch_closes_the_dialog_and_saves_nothing(page, state, seed):
    goto_settings(page)
    make_two_changes(page)
    page.click("[data-testid=settings-save]")
    settle(page)
    page.click("[data-testid=confirm-cancel]")
    settle(page)

    s = state()
    assert page.locator("[data-testid=confirm-dialog]").count() == 0
    assert s["ui"]["settings"]["dialogOpen"] is False
    assert s["settings"]["timezone"] == seed["settings"]["timezone"]
    assert s["settings"]["savedCount"] == 0
    # the draft survives a cancel, so the flow can be resumed
    assert page.locator("[data-testid=settings-dirty]").inner_text() == "2 unsaved changes"
    assert page.locator("[data-testid=settings-success]").count() == 0


def test_confirm_branch_applies_the_changes_and_shows_success(page, state):
    goto_settings(page)
    make_two_changes(page)
    page.click("[data-testid=settings-save]")
    settle(page)
    page.click("[data-testid=confirm-save]")
    settle(page)

    s = state()
    assert page.locator("[data-testid=confirm-dialog]").count() == 0
    assert s["settings"]["timezone"] == "America/Toronto"
    assert s["settings"]["notifyDesktop"] is True
    assert s["settings"]["savedCount"] == 1
    assert s["ui"]["settings"]["success"] is True
    assert "Settings saved" in page.locator("[data-testid=settings-success]").inner_text()
    assert page.locator("[data-testid=settings-dirty]").inner_text() == "No unsaved changes"
    assert page.is_disabled("[data-testid=settings-save]")


def test_discard_reverts_the_draft(page, state, seed):
    goto_settings(page)
    make_two_changes(page)
    page.click("[data-testid=settings-discard]")
    settle(page)
    assert page.locator("[data-testid=settings-dirty]").inner_text() == "No unsaved changes"
    assert state()["ui"]["settings"]["draft"]["timezone"] == seed["settings"]["timezone"]
