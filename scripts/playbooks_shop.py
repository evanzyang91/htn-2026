"""Playbooks for the ten tasks of ``eval/shop.yaml`` on ``apps/shop-site``.

The storefront, whose idiom is a grid of product cards: every card repeats the same
controls, so naming a control never identifies one and where it sits always does.
See ``scripts/playbooks_common.py`` for the shared machinery and
``scripts/scripted_operator.py`` for what a playbook stands in for.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any

from playbooks_common import (  # type: ignore[import-not-found]
    bind,
    choose,
    fill,
    owned_by,
    press,
    sees,
    skill,
)
from scripted_operator import (  # type: ignore[import-not-found]
    Element,
    Playbook,
    click,
    find,
)

__all__ = ["PLAYBOOKS", "playbooks"]

SIDEBAR = (0, 60, 210, 800)
"""The filter rail. Category names also appear on every card as its chip, so a click
meant for the filter has to say which of them it means."""

DRAWER = (890, 50, 1280, 800)
"""The cart, which is drawn over the right-hand columns of the grid."""

HEADER = (700, 0, 1280, 52)


def add_to_cart(product: str, *, times: int = 1, done: bool = False):
    """Press the Add to cart button belonging to one named product."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        title = find(elements, product)
        if title is None:
            return None
        button = owned_by(elements, "Add to cart", title)
        if button is None:
            return None
        clicks = "".join(f'ctx.ctl.click(el["{button.id}"])\n' for _ in range(times))
        return {"code": clicks, "done": done}

    return step


def add_first_card(done: bool = False):
    """Press the Add to cart button of the top-left card, whatever it is now."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        buttons = [e for e in elements if "addtocart" in e.text.replace(" ", "").casefold()]
        grid = [e for e in buttons if e.cx < 880]
        if not grid:
            return None
        first = min(grid, key=lambda e: (e.cy, e.cx))
        return click(first, done=done)

    return step


def checkout_form(name: str, email: str, address: str, shipping: str):
    """Fill the checkout's fields and choose a shipping speed, in one move.

    Looked up through ``ctx.see`` rather than by the ids the prompt listed, because
    those were captured before the block began: typing into the first field re-renders
    the form, and the second field's box was measured on a page that no longer exists.
    Filling in a form IS one decision - nobody means to type a name and then stop -
    and every move is a move the agent must later remember making.
    """

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        return {
            "code": (
                "def under(cap):\n"
                "    labels = ctx.see.find_text(cap)\n"
                '    ctx.expect(bool(labels), "no caption reading " + cap)\n'
                "    label = labels[0]\n"
                "    best = None\n"
                "    for element in ctx.see.all():\n"
                "        drop = element.box.y - label.box.y\n"
                "        if drop < 8 or drop > 60:\n"
                "            continue\n"
                "        # A field is left-aligned with its caption and is field-sized.\n"
                "        # Without both, the nearest thing under a caption can be a\n"
                "        # fragment of whatever the panel is drawn on top of.\n"
                "        if element.box.x - label.box.x < -8:\n"
                "            continue\n"
                "        if element.box.x - label.box.x > 90 or element.box.w < 80:\n"
                "            continue\n"
                "        if best is None or element.box.y < best.box.y:\n"
                "            best = element\n"
                '    ctx.expect(best is not None, "nothing under " + cap)\n'
                "    return best\n"
                "\n"
                "def put(cap, value):\n"
                "    ctx.ctl.click(under(cap))\n"
                '    ctx.ctl.press("Control", "a")\n'
                "    ctx.ctl.type_text(value)\n"
                "\n"
                f'put("Full name", {name!r})\n'
                f'put("Email", {email!r})\n'
                f'put("Address", {address!r})\n'
                "    \n".strip()
                + "\n"
                f'ctx.ctl.click(under("Shipping speed"))\n'
                f"ctx.ctl.type_text({shipping!r})\n"
                'ctx.ctl.press("Enter")\n'
            )
        }

    return (step,)


def place_the_order(done: bool = False):
    """Press Place order, scrolling the drawer to it when the cart is tall enough to
    have pushed it past the bottom of the window."""

    def step(elements: Sequence[Element], failures: int) -> dict[str, Any] | None:
        return {
            "code": (
                'hits = [h for h in ctx.see.find_text("Place order") '
                'if h.kind.value == "button"]\n'
                "if not hits:\n"
                "    tall = None\n"
                "    for element in ctx.see.all():\n"
                "        if element.box.x + element.box.w // 2 < 890:\n"
                "            continue\n"
                "        if tall is None or element.box.area > tall.box.area:\n"
                "            tall = element\n"
                '    ctx.expect(tall is not None, "the checkout drawer is not on screen")\n'
                "    ctx.ctl.scroll(tall, 0, 300)\n"
                '    hits = [h for h in ctx.see.find_text("Place order") '
                'if h.kind.value == "button"]\n'
                'ctx.expect(bool(hits), "no Place order button in the drawer")\n'
                "ctx.ctl.click(hits[0])\n"
            ),
            "done": done,
        }

    return step


def _under(elements: Sequence[Element], caption: str) -> Element | None:
    """The input beneath a form caption, inside the drawer."""
    label = find(elements, caption, within=DRAWER)
    if label is None:
        return None
    best: Element | None = None
    for element in elements:
        drop = element.cy - label.cy
        if not 8 <= drop <= 60 or abs(element.x - label.x) > 90:
            continue
        if best is None or element.cy < best.cy:
            best = element
    return best


# --------------------------------------------------------------------------------------
# The ten
# --------------------------------------------------------------------------------------

_ALL: tuple[Playbook, ...] = (
    # -- single-step -------------------------------------------------------------------
    Playbook(
        task="Show only the products in the Tools category.",
        steps=(press("Tools", within=SIDEBAR, done=True),),
        skill=skill(
            "filter_by_category",
            "Filter the product grid to one category.",
            "the shop, showing only that category.",
            ("category",),
            {"category": "Tools"},
            "    # The rail on the left, not the same word printed on every card.\n"
            "    hits = ctx.see.find_text(category)\n"
            '    ctx.ctl.click(region(ctx, hits, 0, 60, 210, 800, category + " in the rail"))\n'
            "    return True\n",
            sees("products"),
        ),
        bind=bind("filter_by_category", {"category": "Tools"}, "the sentence names the category"),
    ),
    Playbook(
        task='Search the shop for "lantern".',
        steps=(fill("Search products", "lantern", done=True),),
        skill=skill(
            "search_the_shop",
            "Search the shop for a string.",
            "the shop, filtered to the search.",
            ("query",),
            {"query": "lantern"},
            '    box = only(ctx, ctx.see.find_text("Search products"), "the search box")\n'
            "    fill_in(ctx, box, query)\n"
            "    return True\n",
            sees("products"),
        ),
        bind=bind("search_the_shop", {"query": "lantern"}, "the sentence quotes the search"),
    ),
    Playbook(
        task="Sort the products by price, cheapest first.",
        steps=(choose("Featured", "Price: low", done=True),),
        skill=skill(
            "sort_the_products",
            "Sort the product grid by one of its orderings.",
            "the shop, sorted.",
            ("sort",),
            {"sort": "Price: low to high"},
            '    control = only(ctx, ctx.see.find_text("Sort by"), "the sort control")\n'
            "    # The control shows the CURRENT ordering beside its caption; the list of\n"
            "    # orderings is drawn by the operating system and is in no screenshot.\n"
            "    beside = None\n"
            "    for element in ctx.see.all():\n"
            "        if abs(element.box.y - control.box.y) > 14:\n"
            "            continue\n"
            "        if element.box.x <= control.box.x:\n"
            "            continue\n"
            "        if beside is None or element.box.x < beside.box.x:\n"
            "            beside = element\n"
            '    ctx.expect(beside is not None, "nothing beside the sort caption")\n'
            "    pick(ctx, beside, sort)\n"
            "    return True\n",
            sees("products"),
        ),
        bind=bind(
            "sort_the_products",
            {"sort": "Price: low to high"},
            "the sentence names the ordering",
        ),
    ),
    Playbook(
        task="Show only the products under $25.",
        steps=(press("Under $25", within=SIDEBAR, done=True),),
        skill=skill(
            "filter_by_price",
            "Filter the product grid to one price band.",
            "the shop, showing only that band.",
            ("band",),
            {"band": "Under $25"},
            "    hits = ctx.see.find_text(band)\n"
            '    ctx.ctl.click(region(ctx, hits, 0, 60, 210, 800, band + " in the rail"))\n'
            "    return True\n",
            sees("products"),
        ),
        bind=bind("filter_by_price", {"band": "Under $25"}, "the sentence names the band"),
    ),
    # -- multi-step --------------------------------------------------------------------
    Playbook(
        task="Add the Anchor Claw Hammer to the cart.",
        steps=(add_to_cart("Anchor Claw Hammer", done=True),),
        skill=skill(
            "add_product_to_cart",
            "Add one named product to the cart.",
            "the shop, with that product in the cart.",
            ("product",),
            {"product": "Anchor Claw Hammer"},
            "    title = only(ctx, ctx.see.find_text(product), product)\n"
            '    ctx.ctl.click(owned_by(ctx, "Add to cart", title, 20, 130, 240))\n'
            "    return True\n",
            sees("Cart"),
        ),
        bind=bind(
            "add_product_to_cart",
            {"product": "Anchor Claw Hammer"},
            "the sentence names the product",
        ),
    ),
    Playbook(
        task="Put two of the Brass Level 24 in in the cart.",
        steps=(add_to_cart("Brass Level", times=2, done=True),),
        skill=skill(
            "add_several_to_cart",
            "Add a named product to the cart a number of times.",
            "the shop, with that many of the product in the cart.",
            ("product", "quantity"),
            {"product": "Brass Level 24 in", "quantity": "2"},
            "    title = only(ctx, ctx.see.find_text(product), product)\n"
            '    button = owned_by(ctx, "Add to cart", title, 20, 130, 240)\n'
            "    # Pressing it again is what the card is for; the cart's own stepper has\n"
            "    # no readable label at all.\n"
            "    for _step in range(int(quantity)):\n"
            "        ctx.ctl.click(button)\n"
            "    return True\n",
            sees("Cart"),
        ),
        bind=bind(
            "add_several_to_cart",
            {"product": "Brass Level 24 in", "quantity": "2"},
            "the sentence names the product and how many",
        ),
    ),
    Playbook(
        task="Open the shopping cart.",
        steps=(press("Cart", within=HEADER, done=True),),
        skill=skill(
            "open_the_cart",
            "Open the cart drawer.",
            "the shop, with the cart drawer open.",
            (),
            {},
            '    hits = ctx.see.find_text("Cart")\n'
            '    ctx.ctl.click(region(ctx, hits, 700, 0, 1280, 52, "the cart button"))\n'
            "    return True\n",
            sees("Subtotal"),
        ),
        bind=bind("open_the_cart", {}, "the library already knows how to open the cart"),
    ),
    Playbook(
        task="Add the Cobalt Drill Bit Set to the cart, then take it back out again.",
        steps=(
            add_to_cart("Cobalt Drill Bit Set"),
            press("Cart", within=HEADER),
            press("Remove", within=DRAWER, done=True),
        ),
        skill=skill(
            "add_then_remove",
            "Add a product to the cart and then remove it again.",
            "the shop, with the cart open and empty.",
            ("product",),
            {"product": "Cobalt Drill Bit Set"},
            "    title = only(ctx, ctx.see.find_text(product), product)\n"
            '    ctx.ctl.click(owned_by(ctx, "Add to cart", title, 20, 130, 240))\n'
            '    hits = ctx.see.find_text("Cart")\n'
            '    ctx.ctl.click(region(ctx, hits, 700, 0, 1280, 52, "the cart button"))\n'
            '    gone = ctx.see.find_text("Remove")\n'
            '    ctx.ctl.click(region(ctx, gone, 890, 50, 1280, 800, "the remove button"))\n'
            "    return True\n",
            sees("Subtotal"),
        ),
        bind=bind(
            "add_then_remove",
            {"product": "Cobalt Drill Bit Set"},
            "the sentence names the product to add and take out",
        ),
    ),
    # -- composite ---------------------------------------------------------------------
    Playbook(
        task=(
            "Show only the Lighting products, sort them by price with the cheapest first, "
            "and add the first one to the cart."
        ),
        steps=(
            press("Lighting", within=SIDEBAR),
            choose("Featured", "Price: low"),
            add_first_card(done=True),
        ),
        skill=skill(
            "cheapest_in_category",
            "Filter to a category, sort by price and add the cheapest to the cart.",
            "the shop, filtered and sorted, with the cheapest product in the cart.",
            ("category", "sort"),
            {"category": "Lighting", "sort": "Price: low to high"},
            "    hits = ctx.see.find_text(category)\n"
            '    ctx.ctl.click(region(ctx, hits, 0, 60, 210, 800, category + " in the rail"))\n'
            '    control = only(ctx, ctx.see.find_text("Sort by"), "the sort control")\n'
            "    beside = None\n"
            "    for element in ctx.see.all():\n"
            "        if abs(element.box.y - control.box.y) > 14:\n"
            "            continue\n"
            "        if element.box.x <= control.box.x:\n"
            "            continue\n"
            "        if beside is None or element.box.x < beside.box.x:\n"
            "            beside = element\n"
            '    ctx.expect(beside is not None, "nothing beside the sort caption")\n'
            "    pick(ctx, beside, sort)\n"
            "    # Whatever is now top-left is the cheapest, which is the point of\n"
            "    # sorting first rather than naming a product.\n"
            "    first = None\n"
            '    for element in ctx.see.find_text("Add to cart"):\n'
            "        if element.box.x > 880:\n"
            "            continue\n"
            "        if first is None or (element.box.y, element.box.x) < (\n"
            "            first.box.y,\n"
            "            first.box.x,\n"
            "        ):\n"
            "            first = element\n"
            '    ctx.expect(first is not None, "no product cards on screen")\n'
            "    ctx.ctl.click(first)\n"
            "    return True\n",
            sees("Cart"),
        ),
        bind=bind(
            "cheapest_in_category",
            {"category": "Lighting", "sort": "Price: low to high"},
            "the sentence names the category and the ordering",
        ),
    ),
    Playbook(
        task=(
            "Add the Cobalt Drill Bit Set to the cart and place the order for Dana Reed, "
            "dana.reed@harbour.example, 14 Harbour Road, with express shipping."
        ),
        steps=(
            add_to_cart("Cobalt Drill Bit Set"),
            press("Cart", within=HEADER),
            press("Checkout", within=DRAWER),
            *checkout_form("Dana Reed", "dana.reed@harbour.example", "14 Harbour Road", "Express"),
            place_the_order(),
            press("Confirm order", done=True),
        ),
        skill=skill(
            "buy_one_product",
            "Add a product to the cart and place the order for someone.",
            "the shop, with the order placed and the cart empty.",
            ("product", "name", "email", "address", "shipping"),
            {
                "product": "Cobalt Drill Bit Set",
                "name": "Dana Reed",
                "email": "dana.reed@harbour.example",
                "address": "14 Harbour Road",
                "shipping": "Express",
            },
            "    title = only(ctx, ctx.see.find_text(product), product)\n"
            '    ctx.ctl.click(owned_by(ctx, "Add to cart", title, 20, 130, 240))\n'
            '    hits = ctx.see.find_text("Cart")\n'
            '    ctx.ctl.click(region(ctx, hits, 700, 0, 1280, 52, "the cart button"))\n'
            '    ctx.ctl.click(only(ctx, controls(ctx, "Checkout"), "the checkout button"))\n'
            '    fill_in(ctx, under(ctx, "Full name", 60, 90), name)\n'
            '    fill_in(ctx, under(ctx, "Email", 60, 90), email)\n'
            '    fill_in(ctx, under(ctx, "Address", 60, 90), address)\n'
            '    pick(ctx, under(ctx, "Shipping speed", 60, 90), shipping)\n'
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Place order"), "place order"))\n'
            '    ctx.ctl.click(only(ctx, ctx.see.find_text("Confirm order"), "the confirm"))\n'
            "    return True\n",
            sees("placed"),
        ),
        bind=bind(
            "buy_one_product",
            {
                "product": "Cobalt Drill Bit Set",
                "name": "Dana Reed",
                "email": "dana.reed@harbour.example",
                "address": "14 Harbour Road",
                "shipping": "Express",
            },
            "the sentence names the product and every detail of the order",
        ),
    ),
)


def playbooks() -> dict[str, Playbook]:
    """Every task of the shop suite, keyed by the sentence the agent is given."""
    return {playbook.task: playbook for playbook in _ALL}


PLAYBOOKS = playbooks()


if __name__ == "__main__":
    for book in _ALL:
        if book.skill is not None:
            ast.parse(book.skill["code"])
            ast.parse(book.skill["verifier_code"])
    print(json.dumps({"playbooks": len(PLAYBOOKS)}, indent=2))
