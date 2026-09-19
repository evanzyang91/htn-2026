"""Checkout: the form, the confirm-or-back branch, and the placed order."""

from helpers import add_two_products, fill_checkout_form, settle


def open_checkout(page):
    add_two_products(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    page.click("[data-testid=checkout-open]")
    settle(page)


def test_checkout_opens_from_the_cart(page, state):
    open_checkout(page)
    s = state()
    assert s["ui"]["cartOpen"] is True and s["ui"]["checkoutOpen"] is True
    assert page.locator("[data-testid=checkout-panel]").is_visible()
    assert (
        page.locator("[data-testid=checkout-summary]").inner_text() == "2 items · subtotal $93.98"
    )


def test_back_to_cart_returns_to_the_lines_view(page, state):
    open_checkout(page)
    page.click("[data-testid=checkout-back]")
    settle(page)
    assert state()["ui"]["checkoutOpen"] is False
    assert page.locator("[data-testid=cart-lines]").is_visible()
    assert page.locator("[data-testid=checkout-panel]").count() == 0


def test_place_order_is_disabled_until_the_form_is_complete(page, state):
    open_checkout(page)
    assert page.is_disabled("[data-testid=place-order]")
    fill_checkout_form(page)
    assert page.is_enabled("[data-testid=place-order]")

    s = state()
    assert s["ui"]["form"]["name"] == "Avery Quinn"
    assert s["orders"] == []  # nothing placed yet


def test_shipping_speed_changes_the_total(page, state):
    open_checkout(page)
    assert page.locator("[data-testid=checkout-total]").inner_text() == "$93.98"
    page.select_option("[data-testid=form-shipping]", "overnight")
    settle(page)
    assert state()["ui"]["form"]["shipping"] == "overnight"
    assert page.locator("[data-testid=checkout-shipping-fee]").inner_text() == "$24.00"
    assert page.locator("[data-testid=checkout-total]").inner_text() == "$117.98"


def test_place_order_opens_a_confirm_dialog(page, state):
    open_checkout(page)
    fill_checkout_form(page)
    page.click("[data-testid=place-order]")
    settle(page)
    assert state()["ui"]["dialogOpen"] is True
    assert page.locator("[data-testid=confirm-dialog]").is_visible()
    assert page.locator("[data-testid=scrim]").is_visible()
    items = [li.inner_text() for li in page.locator("[data-testid=confirm-list] li").all()]
    assert items == [
        "Items: 2",
        "Total: $93.98",
        "Ship to: Avery Quinn, 12 Pier Road, Port Town",
    ]


def test_back_branch_closes_the_dialog_and_places_nothing(page, state):
    open_checkout(page)
    fill_checkout_form(page)
    page.click("[data-testid=place-order]")
    settle(page)
    page.click("[data-testid=confirm-back]")
    settle(page)

    s = state()
    assert page.locator("[data-testid=confirm-dialog]").count() == 0
    assert s["ui"]["dialogOpen"] is False
    assert s["orders"] == []
    # the form and the cart survive, so the flow can be resumed
    assert s["ui"]["form"]["name"] == "Avery Quinn"
    assert s["cart"]["lines"] != []


def test_confirm_branch_places_the_order(page, state):
    open_checkout(page)
    fill_checkout_form(page)
    page.select_option("[data-testid=form-shipping]", "express")
    settle(page)
    page.click("[data-testid=place-order]")
    settle(page)
    page.click("[data-testid=confirm-order]")
    settle(page)

    s = state()
    assert len(s["orders"]) == 1
    order = s["orders"][0]
    assert order["id"] == "o01"
    assert order["number"] == 1001
    assert order["items"] == [
        {"productId": "p01", "name": "Anchor Claw Hammer", "price": 1899, "qty": 1},
        {"productId": "p13", "name": "Mainsail Chef Knife", "price": 7499, "qty": 1},
    ]
    assert order["subtotal"] == 9398
    assert order["shippingFee"] == 900
    assert order["total"] == 10298
    assert order["shipping"] == "express"
    assert order["name"] == "Avery Quinn"
    assert order["email"] == "avery.quinn@harbour.example"
    assert order["address"] == "12 Pier Road, Port Town"
    assert order["saveDetails"] is False
    assert order["placed"] == "Mar 12"

    # the cart empties, every panel closes, and the banner announces the order
    assert s["cart"]["lines"] == []
    assert s["ui"]["cartOpen"] is False
    assert s["ui"]["checkoutOpen"] is False
    assert s["ui"]["dialogOpen"] is False
    assert s["ui"]["banner"] == "Order #1001 placed - thank you!"
    assert page.locator("[data-testid=cart-drawer]").count() == 0
    assert "Order #1001 placed" in page.inner_text("[data-testid=shop-banner]")
    assert page.locator("[data-testid=cart-open]").inner_text() == "Cart (0)"

    # save was left unchecked, so the form resets to blank
    assert s["ui"]["form"] == {
        "name": "",
        "email": "",
        "address": "",
        "shipping": "standard",
        "save": False,
    }

    page.click("[data-testid=shop-banner-dismiss]")
    settle(page)
    assert state()["ui"]["banner"] is None
    assert page.locator("[data-testid=shop-banner]").count() == 0


def test_save_details_keeps_the_form_for_the_next_order(page, state):
    open_checkout(page)
    fill_checkout_form(page)
    page.click("[data-testid=form-save]")
    settle(page)
    assert state()["ui"]["form"]["save"] is True
    page.click("[data-testid=place-order]")
    settle(page)
    page.click("[data-testid=confirm-order]")
    settle(page)

    s = state()
    assert s["orders"][0]["saveDetails"] is True
    assert s["ui"]["form"]["name"] == "Avery Quinn"
    assert s["ui"]["form"]["address"] == "12 Pier Road, Port Town"

    # a second order gets the next counter id
    page.click("[data-testid=add-p20]")
    settle(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    page.click("[data-testid=checkout-open]")
    settle(page)
    assert page.is_enabled("[data-testid=place-order]")  # saved details still filled
    page.click("[data-testid=place-order]")
    settle(page)
    page.click("[data-testid=confirm-order]")
    settle(page)
    s = state()
    assert [o["id"] for o in s["orders"]] == ["o01", "o02"]
    assert s["orders"][1]["number"] == 1002
    assert s["ui"]["banner"] == "Order #1002 placed - thank you!"
