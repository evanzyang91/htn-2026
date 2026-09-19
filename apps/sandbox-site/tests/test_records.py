"""Records flows: filter, sort, inline edit, bulk action, export."""

from helpers import settle


def rows(page):
    return page.locator("[data-testid=rec-table] tbody tr")


def goto_records(page):
    page.click("[data-testid=nav-records]")
    settle(page)


def test_table_starts_with_the_full_seed(page, seed):
    goto_records(page)
    assert rows(page).count() == len(seed["records"]["rows"]) == 40
    assert page.locator("[data-testid=rec-footer-count]").inner_text() == "Showing 40 of 40 records"


def test_filter_narrows_the_table(page, state):
    goto_records(page)
    page.fill("[data-testid=rec-filter]", "Ingest")
    settle(page)
    assert rows(page).count() == 8
    assert page.locator("[data-testid=rec-footer-count]").inner_text() == "Showing 8 of 40 records"
    assert state()["ui"]["records"]["filter"] == "Ingest"


def test_sorting_a_column_toggles_direction(page, state):
    goto_records(page)
    page.click("[data-testid=sort-records]")
    settle(page)
    assert state()["ui"]["records"]["sortDir"] == "asc"
    asc = [c.inner_text() for c in rows(page).locator("td.num").all()]

    page.click("[data-testid=sort-records]")
    settle(page)
    assert state()["ui"]["records"]["sortDir"] == "desc"
    desc = [c.inner_text() for c in rows(page).locator("td.num").all()]

    assert asc == list(reversed(desc))
    assert asc == sorted(asc, key=lambda v: int(v.replace(",", "")))


def test_inline_edit_of_one_cell(page, state):
    goto_records(page)
    page.click("[data-testid=editbtn-r03]")
    settle(page)
    assert page.input_value("[data-testid=edit-input]") == "Cinder Gateway"

    page.fill("[data-testid=edit-input]", "Cinder Gateway v2")
    settle(page)
    page.click("[data-testid=edit-save]")
    settle(page)

    s = state()
    assert [r for r in s["records"]["rows"] if r["id"] == "r03"][0]["name"] == "Cinder Gateway v2"
    assert s["ui"]["records"]["editingId"] is None
    assert s["ui"]["records"]["banner"] == "Renamed to Cinder Gateway v2"
    assert page.locator("[data-testid=edit-input]").count() == 0


def test_inline_edit_can_be_cancelled(page, state):
    goto_records(page)
    page.click("[data-testid=editbtn-r03]")
    settle(page)
    page.fill("[data-testid=edit-input]", "Should Not Stick")
    settle(page)
    page.click("[data-testid=edit-cancel]")
    settle(page)
    assert [r for r in state()["records"]["rows"] if r["id"] == "r03"][0]["name"] == "Cinder Gateway"


def test_bulk_status_across_the_selection(page, state):
    goto_records(page)
    page.fill("[data-testid=rec-filter]", "Ingest")
    settle(page)
    assert page.is_disabled("[data-testid=bulk-btn]")

    page.click("[data-testid=rec-select-all]")
    settle(page)
    assert page.locator("[data-testid=rec-selcount]").inner_text() == "8 selected"
    assert page.is_enabled("[data-testid=bulk-btn]")

    page.click("[data-testid=bulk-btn]")
    settle(page)
    page.click("[data-testid=bulk-option-Paused]")
    settle(page)

    s = state()
    ingest = [r for r in s["records"]["rows"] if r["category"] == "Ingest"]
    assert len(ingest) == 8
    assert {r["status"] for r in ingest} == {"Paused"}
    assert s["ui"]["records"]["selected"] == []
    # rows outside the filter are untouched
    assert {r["status"] for r in s["records"]["rows"] if r["category"] != "Ingest"} != {"Paused"}


def test_export_records_the_visible_rows(page, state):
    goto_records(page)
    page.fill("[data-testid=rec-filter]", "Ingest")
    settle(page)
    page.click("[data-testid=export-btn]")
    settle(page)

    exports = state()["records"]["exports"]
    assert len(exports) == 1
    assert exports[0]["count"] == 8
    assert exports[0]["filter"] == "Ingest"
    assert len(exports[0]["rowIds"]) == 8
    assert page.locator("[data-testid=rec-exports]").inner_text() == "Exports this session: 1"


def test_export_prefers_the_selection_when_there_is_one(page, state):
    goto_records(page)
    page.click("[data-testid=reccheck-r01]")
    settle(page)
    page.click("[data-testid=reccheck-r02]")
    settle(page)
    page.click("[data-testid=export-btn]")
    settle(page)

    exports = state()["records"]["exports"]
    assert exports[0]["count"] == 2
    assert sorted(exports[0]["rowIds"]) == ["r01", "r02"]
