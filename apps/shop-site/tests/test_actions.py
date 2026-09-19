"""Every action in the vocabulary round-trips through POST /api/act.

These tests speak HTTP directly - no browser - so they pin the server
contract itself: request shape, seq echo, complete-state responses, and
a 400 with an explanatory error for anything outside the vocabulary.
"""

import json
import urllib.error
import urllib.request

import pytest


def post(base_url, action, payload=None, seq=7):
    req = urllib.request.Request(
        base_url + "/api/act",
        data=json.dumps({"action": action, "payload": payload or {}, "seq": seq}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


@pytest.fixture(autouse=True)
def _fresh(base_url):
    urllib.request.urlopen(base_url + "/__reset").read()


def test_response_echoes_seq_and_returns_the_complete_state(base_url, seed):
    data = post(base_url, "shop.search", {"q": "lamp"}, seq=42)
    assert data["seq"] == 42
    assert set(data["state"]) == {"meta", "catalog", "cart", "orders", "ui"}
    assert data["state"]["catalog"] == seed["catalog"]
    assert data["state"]["ui"]["search"] == "lamp"


def test_every_catalog_action_round_trips(base_url):
    assert post(base_url, "shop.search", {"q": "rope"})["state"]["ui"]["search"] == "rope"
    assert (
        post(base_url, "shop.category", {"category": "Outdoor"})["state"]["ui"]["category"]
        == "Outdoor"
    )
    assert (
        post(base_url, "shop.priceBand", {"band": "25to75"})["state"]["ui"]["priceBand"] == "25to75"
    )
    assert (
        post(base_url, "shop.sort", {"sort": "rating-desc"})["state"]["ui"]["sort"] == "rating-desc"
    )


def test_invalid_filter_values_are_ignored_not_errors(base_url):
    assert post(base_url, "shop.category", {"category": "Nope"})["state"]["ui"]["category"] == "All"
    assert post(base_url, "shop.priceBand", {"band": "free"})["state"]["ui"]["priceBand"] == "any"
    assert post(base_url, "shop.sort", {"sort": "chaos"})["state"]["ui"]["sort"] == "featured"


def test_every_cart_action_round_trips(base_url):
    s = post(base_url, "cart.add", {"productId": "p02"})["state"]
    assert s["cart"]["lines"] == [{"productId": "p02", "qty": 1}]
    s = post(base_url, "cart.increment", {"productId": "p02"})["state"]
    assert s["cart"]["lines"] == [{"productId": "p02", "qty": 2}]
    s = post(base_url, "cart.decrement", {"productId": "p02"})["state"]
    assert s["cart"]["lines"] == [{"productId": "p02", "qty": 1}]
    s = post(base_url, "cart.decrement", {"productId": "p02"})["state"]
    assert s["cart"]["lines"] == [{"productId": "p02", "qty": 1}]  # floor at 1
    assert post(base_url, "cart.open")["state"]["ui"]["cartOpen"] is True
    assert post(base_url, "cart.close")["state"]["ui"]["cartOpen"] is False
    s = post(base_url, "cart.remove", {"productId": "p02"})["state"]
    assert s["cart"]["lines"] == []
    # unknown product ids are ignored, not errors
    assert post(base_url, "cart.add", {"productId": "p99"})["state"]["cart"]["lines"] == []


def test_every_checkout_action_round_trips(base_url):
    post(base_url, "cart.add", {"productId": "p01"})
    s = post(base_url, "checkout.open")["state"]
    assert s["ui"]["cartOpen"] is True and s["ui"]["checkoutOpen"] is True

    for field, value in (
        ("name", "Avery Quinn"),
        ("email", "avery.quinn@harbour.example"),
        ("address", "12 Pier Road, Port Town"),
        ("shipping", "overnight"),
        ("save", True),
    ):
        s = post(base_url, "checkout.field", {"field": field, "value": value})["state"]
        assert s["ui"]["form"][field] == value

    s = post(base_url, "checkout.submit")["state"]
    assert s["ui"]["dialogOpen"] is True
    s = post(base_url, "checkout.cancel")["state"]
    assert s["ui"]["dialogOpen"] is False
    post(base_url, "checkout.submit")
    s = post(base_url, "checkout.confirm")["state"]
    assert len(s["orders"]) == 1
    assert s["orders"][0]["total"] == 1899 + 2400
    assert s["cart"]["lines"] == []
    assert s["ui"]["banner"] == "Order #1001 placed - thank you!"
    s = post(base_url, "shop.dismissBanner")["state"]
    assert s["ui"]["banner"] is None

    s = post(base_url, "checkout.back")["state"]
    assert s["ui"]["checkoutOpen"] is False


def test_checkout_guards(base_url):
    # empty cart: checkout will not open, submit will not arm the dialog
    assert post(base_url, "checkout.open")["state"]["ui"]["checkoutOpen"] is False
    assert post(base_url, "checkout.submit")["state"]["ui"]["dialogOpen"] is False
    # incomplete form: submit will not arm the dialog either
    post(base_url, "cart.add", {"productId": "p01"})
    post(base_url, "checkout.open")
    assert post(base_url, "checkout.submit")["state"]["ui"]["dialogOpen"] is False
    # confirm without an armed dialog places nothing
    assert post(base_url, "checkout.confirm")["state"]["orders"] == []


def test_unknown_action_is_a_400_with_an_error_body(base_url):
    req = urllib.request.Request(
        base_url + "/api/act",
        data=json.dumps({"action": "warp.speed", "payload": {}, "seq": 1}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400
    assert json.loads(exc.value.read()) == {"error": "unknown action: warp.speed"}


def test_bad_json_is_a_400(base_url):
    req = urllib.request.Request(
        base_url + "/api/act",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400
