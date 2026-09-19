"""Cart flows: add, drawer, quantity steppers, remove, subtotal."""

from helpers import add_two_products, settle


def test_add_to_cart_updates_the_header_count(page, state):
    assert page.locator("[data-testid=cart-open]").inner_text() == "Cart (0)"
    add_two_products(page)
    assert page.locator("[data-testid=cart-open]").inner_text() == "Cart (2)"
    assert state()["cart"]["lines"] == [
        {"productId": "p01", "qty": 1},
        {"productId": "p13", "qty": 1},
    ]


def test_adding_the_same_product_twice_grows_one_line(page, state):
    page.click("[data-testid=add-p05]")
    settle(page)
    page.click("[data-testid=add-p05]")
    settle(page)
    assert state()["cart"]["lines"] == [{"productId": "p05", "qty": 2}]
    assert page.locator("[data-testid=add-p05]").inner_text() == "Add to cart (2)"


def test_the_drawer_opens_and_closes(page, state):
    assert page.locator("[data-testid=cart-drawer]").count() == 0
    page.click("[data-testid=cart-open]")
    settle(page)
    assert state()["ui"]["cartOpen"] is True
    assert page.locator("[data-testid=cart-drawer]").is_visible()
    assert "Your cart is empty" in page.inner_text("[data-testid=cart-lines]")
    assert page.is_disabled("[data-testid=checkout-open]")

    page.click("[data-testid=cart-close]")
    settle(page)
    assert state()["ui"]["cartOpen"] is False
    assert page.locator("[data-testid=cart-drawer]").count() == 0


def test_quantity_steppers(page, state):
    add_two_products(page)
    page.click("[data-testid=cart-open]")
    settle(page)

    # minus is disabled at quantity 1; plus always works
    assert page.is_disabled("[data-testid=qty-minus-p01]")
    page.click("[data-testid=qty-plus-p01]")
    settle(page)
    page.click("[data-testid=qty-plus-p01]")
    settle(page)
    assert page.locator("[data-testid=qty-p01]").inner_text() == "3"
    assert state()["cart"]["lines"][0] == {"productId": "p01", "qty": 3}

    page.click("[data-testid=qty-minus-p01]")
    settle(page)
    assert state()["cart"]["lines"][0] == {"productId": "p01", "qty": 2}

    # at quantity 1 the minus button is disabled, so the floor is unreachable
    assert page.is_disabled("[data-testid=qty-minus-p13]")
    assert state()["cart"]["lines"][1] == {"productId": "p13", "qty": 1}


def test_subtotal_and_line_totals(page):
    add_two_products(page)  # p01 $18.99, p13 $74.99
    page.click("[data-testid=cart-open]")
    settle(page)
    page.click("[data-testid=qty-plus-p01]")
    settle(page)
    assert page.locator("[data-testid=linetotal-p01]").inner_text() == "$37.98"
    assert page.locator("[data-testid=linetotal-p13]").inner_text() == "$74.99"
    assert page.locator("[data-testid=cart-subtotal]").inner_text() == "$112.97"


def test_remove_deletes_the_line(page, state):
    add_two_products(page)
    page.click("[data-testid=cart-open]")
    settle(page)
    page.click("[data-testid=remove-p01]")
    settle(page)
    assert state()["cart"]["lines"] == [{"productId": "p13", "qty": 1}]
    assert page.locator("[data-testid=line-p01]").count() == 0

    page.click("[data-testid=remove-p13]")
    settle(page)
    assert state()["cart"]["lines"] == []
    assert "Your cart is empty" in page.inner_text("[data-testid=cart-lines]")
    assert page.locator("[data-testid=cart-open]").inner_text() == "Cart (0)"
