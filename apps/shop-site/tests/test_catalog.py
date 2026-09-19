"""Catalog flows: search, category filter, price band, sort."""

from helpers import settle


def names(page):
    return [el.inner_text() for el in page.locator("[data-testid^=name-]").all()]


def test_the_grid_shows_every_product_on_load(page, seed):
    assert page.locator("[data-testid^=card-]").count() == len(seed["catalog"]["products"])
    assert page.locator("[data-testid=result-count]").inner_text() == "24 of 24 products"


def test_category_filter_narrows_the_grid(page, state):
    page.click("[data-testid=cat-Kitchen]")
    settle(page)
    assert state()["ui"]["category"] == "Kitchen"
    assert page.locator("[data-testid^=card-]").count() == 6
    assert page.locator("[data-testid=result-count]").inner_text() == "6 of 24 products"
    assert page.get_attribute("[data-testid=cat-Kitchen]", "aria-current") == "true"

    page.click("[data-testid=cat-All]")
    settle(page)
    assert page.locator("[data-testid^=card-]").count() == 24


def test_price_band_radios_narrow_the_grid(page, state, seed):
    page.click("[data-testid=price-over75]")
    settle(page)
    assert state()["ui"]["priceBand"] == "over75"
    expected = [p for p in seed["catalog"]["products"] if p["price"] > 7500]
    assert page.locator("[data-testid^=card-]").count() == len(expected)
    assert page.is_checked("[data-testid=price-over75]")
    assert not page.is_checked("[data-testid=price-any]")


def test_search_filters_by_name(page, state):
    page.fill("[data-testid=shop-search]", "lamp")
    settle(page)
    assert state()["ui"]["search"] == "lamp"
    assert sorted(names(page)) == ["Sextant Desk Lamp", "Vane Headlamp", "Wharf Floor Lamp"]

    page.fill("[data-testid=shop-search]", "zzz")
    settle(page)
    assert page.locator("[data-testid^=card-]").count() == 0
    assert "No products match" in page.inner_text("[data-testid=gridwrap]")


def test_filters_compose(page):
    page.click("[data-testid=cat-Tools]")
    settle(page)
    page.click("[data-testid=price-under25]")
    settle(page)
    assert sorted(names(page)) == [
        "Anchor Claw Hammer", "Dockside Socket Wrench", "Ebb Tide Hand Saw",
    ]


def test_sort_orders_the_grid(page, state, seed):
    page.select_option("[data-testid=sort-select]", "price-asc")
    settle(page)
    assert state()["ui"]["sort"] == "price-asc"
    by_price = sorted(seed["catalog"]["products"], key=lambda p: (p["price"], p["id"]))
    assert names(page) == [p["name"] for p in by_price]

    page.select_option("[data-testid=sort-select]", "rating-desc")
    settle(page)
    assert names(page)[0] == "Mainsail Chef Knife"  # rating 4.9

    page.select_option("[data-testid=sort-select]", "name-asc")
    settle(page)
    assert names(page) == sorted(names(page))


def test_search_survives_the_render_round_trip(page, state):
    """Every keystroke posts to the server; the queue must keep them in order."""
    q = "a fairly long query typed in one go"
    page.fill("[data-testid=shop-search]", q)
    settle(page)
    assert page.input_value("[data-testid=shop-search]") == q
    assert state()["ui"]["search"] == q
