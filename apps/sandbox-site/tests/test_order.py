"""Pantry Lane flows: catalogue search, detail navigation, required choices,
the cart, and the multi-step commitment that moves a cart into order history.

The last of those is what this surface exists for. It is the only place in the
app where a task changes state through several dependent steps, and every test
here also has to be true again after ``GET /__reset`` - see test_determinism.
"""

import json
import urllib.request

from helpers import settle


def goto_order(page):
    page.click("[data-testid=nav-order]")
    settle(page)


def cards(page):
    return page.locator("[data-testid=restaurant-grid] .restcard")


def add_pad_thai(page, portion="large"):
    """The canonical add: back to the catalogue, into a restaurant, a dish, choose, add.

    Adding leaves the shopper on the restaurant page, which is what a real one
    does, so this starts by going back to Browse and clearing any cuisine chip.
    """
    page.click("[data-testid=ord-tab-browse]")
    settle(page)
    page.click("[data-testid=cuisine-all]")
    settle(page)
    page.click("[data-testid=restaurant-rest-kettle]")
    settle(page)
    page.click("[data-testid=dish-d-kettle-padthai]")
    settle(page)
    page.click(f"[data-testid=choice-portion-{portion}]")
    settle(page)
    page.click("[data-testid=add-to-cart]")
    settle(page)


# -- the catalogue ---------------------------------------------------------------


def test_browse_lists_every_seeded_restaurant(page, seed):
    goto_order(page)
    assert cards(page).count() == len(seed["order"]["restaurants"]) == 7
    assert page.locator("[data-testid=ord-result-count]").inner_text() == (
        "Showing 7 of 7 restaurants"
    )


def test_search_matches_a_dish_not_only_a_restaurant_name(page, state):
    goto_order(page)
    page.fill("[data-testid=ord-search]", "churros")
    settle(page)
    assert cards(page).count() == 1
    assert page.locator("[data-testid=restaurant-rest-ember]").count() == 1
    assert state()["ui"]["order"]["search"] == "churros"


def test_cuisine_filter_narrows_and_toggles_off(page, state):
    goto_order(page)
    page.click("[data-testid=cuisine-Thai]")
    settle(page)
    assert cards(page).count() == 2
    assert state()["ui"]["order"]["cuisine"] == "Thai"

    page.click("[data-testid=cuisine-Thai]")
    settle(page)
    assert state()["ui"]["order"]["cuisine"] == ""
    assert cards(page).count() == 7


def test_opening_a_restaurant_navigates_into_its_menu(page, state):
    goto_order(page)
    page.click("[data-testid=restaurant-rest-sakura]")
    settle(page)
    s = state()
    assert s["ui"]["order"]["view"] == "restaurant"
    assert s["ui"]["order"]["restaurantId"] == "rest-sakura"
    assert page.locator("[data-testid=rest-name]").inner_text() == "Sakura Counter"
    assert page.locator("[data-testid=dish-list] .dishrow").count() == 4

    page.click("[data-testid=ord-back]")
    settle(page)
    assert state()["ui"]["order"]["view"] == "browse"


# -- required choices ------------------------------------------------------------


def test_a_dish_with_options_cannot_be_added_until_they_are_chosen(page, state):
    goto_order(page)
    page.click("[data-testid=restaurant-rest-kettle]")
    settle(page)
    page.click("[data-testid=dish-d-kettle-padthai]")
    settle(page)

    assert page.is_disabled("[data-testid=add-to-cart]")
    assert page.locator("[data-testid=optreq-portion]").inner_text().lower() == "required"

    page.click("[data-testid=choice-portion-large]")
    settle(page)
    assert page.is_enabled("[data-testid=add-to-cart]")
    assert page.locator("[data-testid=optreq-portion]").inner_text().lower() == "chosen"
    assert state()["ui"]["order"]["picks"] == {"portion": "large"}


def test_every_option_group_is_required_not_just_the_first(page, state):
    goto_order(page)
    page.click("[data-testid=restaurant-rest-forno]")
    settle(page)
    page.click("[data-testid=dish-d-forno-margherita]")
    settle(page)
    page.click("[data-testid=choice-size-fourteen]")
    settle(page)
    assert page.is_disabled("[data-testid=add-to-cart]")

    page.click("[data-testid=choice-crust-thin]")
    settle(page)
    page.click("[data-testid=add-to-cart]")
    settle(page)

    line = state()["order"]["cart"][0]
    assert line["name"] == "Margherita Pizza"
    assert line["choiceText"] == "Fourteen inch, Thin"


def test_an_option_price_delta_reaches_the_cart(page, state):
    goto_order(page)
    add_pad_thai(page, portion="large")
    line = state()["order"]["cart"][0]
    assert line["unitPriceCents"] == 1450 + 300
    assert line["choices"] == [{"group": "Portion", "choice": "Large"}]


def test_a_dish_with_no_options_adds_straight_away(page, state):
    goto_order(page)
    page.click("[data-testid=restaurant-rest-kettle]")
    settle(page)
    page.click("[data-testid=dish-d-kettle-mangorice]")
    settle(page)
    assert page.is_enabled("[data-testid=add-to-cart]")
    page.click("[data-testid=add-to-cart]")
    settle(page)
    assert state()["order"]["cart"][0]["name"] == "Mango Sticky Rice"


# -- the cart --------------------------------------------------------------------


def test_the_quantity_stepper_carries_into_the_cart(page, state):
    goto_order(page)
    page.click("[data-testid=restaurant-rest-kettle]")
    settle(page)
    page.click("[data-testid=dish-d-kettle-springrolls]")
    settle(page)
    page.click("[data-testid=choice-count-eight]")
    settle(page)
    page.click("[data-testid=qty-inc]")
    settle(page)
    page.click("[data-testid=qty-inc]")
    settle(page)
    assert page.locator("[data-testid=qty-value]").inner_text() == "3"
    page.click("[data-testid=add-to-cart]")
    settle(page)
    assert state()["order"]["cart"][0]["qty"] == 3


def test_adding_the_same_dish_and_choices_twice_merges_the_line(page, state):
    goto_order(page)
    add_pad_thai(page)
    add_pad_thai(page)
    cart = state()["order"]["cart"]
    assert len(cart) == 1
    assert cart[0]["qty"] == 2


def test_the_same_dish_with_different_choices_is_a_separate_line(page, state):
    goto_order(page)
    add_pad_thai(page, portion="large")
    add_pad_thai(page, portion="regular")
    cart = state()["order"]["cart"]
    assert len(cart) == 2
    assert [line["choiceText"] for line in cart] == ["Large", "Regular"]


def test_changing_a_quantity_in_the_cart(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=cartinc-c01]")
    settle(page)
    page.click("[data-testid=cartinc-c01]")
    settle(page)
    assert page.locator("[data-testid=cartqty-c01]").inner_text() == "3"
    assert state()["order"]["cart"][0]["qty"] == 3

    page.click("[data-testid=cartdec-c01]")
    settle(page)
    assert state()["order"]["cart"][0]["qty"] == 2
    # a line never steps below one - removing is its own action
    assert page.locator("[data-testid=cartsum-c01]").inner_text() == "$35.00"


def test_removing_a_line_leaves_the_rest_of_the_cart(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-back]")
    settle(page)
    page.click("[data-testid=restaurant-rest-ember]")
    settle(page)
    page.click("[data-testid=dish-d-ember-elote]")
    settle(page)
    page.click("[data-testid=add-to-cart]")
    settle(page)

    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=cartremove-c01]")
    settle(page)

    cart = state()["order"]["cart"]
    assert [line["name"] for line in cart] == ["Elote"]
    assert state()["ui"]["order"]["banner"] == "Removed Pad Thai from the cart"


def test_a_removed_line_id_is_never_handed_out_again(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=cartremove-c01]")
    settle(page)
    page.click("[data-testid=ord-tab-browse]")
    settle(page)
    add_pad_thai(page)
    assert state()["order"]["cart"][0]["lineId"] == "c02"


def test_the_running_total_adds_delivery_and_tip(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    assert page.locator("[data-testid=total-subtotal]").inner_text() == "$17.50"
    assert page.locator("[data-testid=total-delivery]").inner_text() == "$3.99"
    assert page.locator("[data-testid=total-tip]").inner_text() == "$4.00"
    assert page.locator("[data-testid=total-grand]").inner_text() == "$25.49"

    page.click("[data-testid=tip-600]")
    settle(page)
    assert page.locator("[data-testid=total-grand]").inner_text() == "$27.49"
    assert state()["ui"]["order"]["tipCents"] == 600


def test_an_empty_cart_offers_nothing_to_check_out(page):
    goto_order(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    assert page.locator("[data-testid=checkout-btn]").count() == 0
    assert page.locator("[data-testid=cart-view] .empty").count() == 1


# -- the commitment --------------------------------------------------------------


def test_placing_an_order_needs_the_confirmation(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)

    assert page.locator("[data-testid=order-dialog]").count() == 1
    assert state()["ui"]["order"]["dialogOpen"] is True
    assert state()["order"]["orders"] == []

    page.click("[data-testid=order-cancel]")
    settle(page)
    s = state()
    assert s["ui"]["order"]["dialogOpen"] is False
    assert s["order"]["orders"] == []
    # cancelling keeps the cart intact
    assert len(s["order"]["cart"]) == 1


def test_a_placed_order_empties_the_cart_and_enters_the_history(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)
    page.click("[data-testid=order-confirm]")
    settle(page)

    s = state()
    assert s["order"]["cart"] == []
    assert len(s["order"]["orders"]) == 1
    placed = s["order"]["orders"][0]
    assert placed["id"] == "o01"
    assert placed["restaurants"] == ["Copper Kettle"]
    assert placed["itemCount"] == 1
    assert placed["subtotalCents"] == 1750
    assert placed["totalCents"] == 1750 + 399 + 400
    assert placed["address"] == "18 Alder Street, Apt 4"
    assert placed["placedOn"] == s["meta"]["today"]
    assert placed["status"] == "Confirmed"

    assert s["ui"]["order"]["view"] == "orders"
    assert page.locator("[data-testid=order-o01]").count() == 1
    assert page.locator("[data-testid=ordertotal-o01]").inner_text() == "$25.49"


def test_the_screen_total_and_the_placed_order_agree(page, state):
    """Cents on the server, cents in the browser - the two must not drift."""
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=cartinc-c01]")
    settle(page)
    page.click("[data-testid=tip-200]")
    settle(page)
    shown = page.locator("[data-testid=total-grand]").inner_text()

    page.click("[data-testid=checkout-btn]")
    settle(page)
    assert page.locator("[data-testid=dialog-total]").inner_text().startswith("Total " + shown)
    page.click("[data-testid=order-confirm]")
    settle(page)
    assert page.locator("[data-testid=ordertotal-o01]").inner_text() == shown


def test_an_order_can_span_two_restaurants(page, state):
    goto_order(page)
    page.click("[data-testid=cuisine-Thai]")
    settle(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-browse]")
    settle(page)
    page.click("[data-testid=cuisine-all]")
    settle(page)
    page.click("[data-testid=restaurant-rest-forno]")
    settle(page)
    page.click("[data-testid=dish-d-forno-tiramisu]")
    settle(page)
    page.click("[data-testid=add-to-cart]")
    settle(page)

    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)
    page.click("[data-testid=order-confirm]")
    settle(page)

    placed = state()["order"]["orders"][0]
    assert placed["restaurants"] == ["Copper Kettle", "Forno Nove"]
    assert sorted(line["name"] for line in placed["items"]) == ["Pad Thai", "Tiramisu"]


def test_the_delivery_address_is_editable_and_is_recorded(page, state):
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.fill("[data-testid=ord-address]", "9 Beech Lane")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)
    page.click("[data-testid=order-confirm]")
    settle(page)
    assert state()["order"]["orders"][0]["address"] == "9 Beech Lane"


# -- reset, the property the admission gate depends on ---------------------------


def test_reset_undoes_a_placed_order(page, base_url, state, seed):
    """A learned skill is only stored after being re-run, so every state change
    the ordering flow makes has to be undoable. This is that guarantee."""
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=ord-tab-cart]")
    settle(page)
    page.click("[data-testid=checkout-btn]")
    settle(page)
    page.click("[data-testid=order-confirm]")
    settle(page)
    assert state()["order"]["orders"]

    after = json.loads(urllib.request.urlopen(base_url + "/__reset").read())["state"]
    assert after["order"] == seed["order"]
    assert after["order"]["cart"] == []
    assert after["order"]["orders"] == []
    assert after["ui"]["order"]["view"] == "browse"
    assert after["ui"]["order"]["restaurantId"] is None
    assert after["ui"]["order"]["tipCents"] == seed["order"]["defaultTipCents"]
    assert after["ui"]["order"]["address"] == seed["order"]["defaultAddress"]
    assert after["ui"]["order"]["lineSeq"] == 0


def test_the_header_cart_chip_opens_the_cart(page, state):
    """It looks like the cart button every ordering site has, so it is one."""
    goto_order(page)
    add_pad_thai(page)
    page.click("[data-testid=cart-total]")
    settle(page)
    assert state()["ui"]["order"]["view"] == "cart"
    assert page.locator("[data-testid=cart-view]").count() == 1
