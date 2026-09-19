"""Tests for :mod:`skillweaver.perception.fingerprint`.

The pair tests read the labelled PNGs committed under ``tests/fixtures/shots/pairs``, so
nothing here needs a browser, a model or a network. Regenerate them with
``uv run python tests/fixtures/shots/pairs/generate.py``.

Each of the three signals also has its own unit tests, so a regression points at one
signal rather than at "the fingerprint changed".
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skillweaver.contracts import Box, Element, ElementKind, ElementSource, Screenshot
from skillweaver.errors import PerceptionError
from skillweaver.perception.fingerprint import (
    SAME_STATE_THRESHOLD,
    StateFingerprinter,
    normalize_url,
    perceptual_hash,
    structural_hash,
)

PAIRS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shots" / "pairs"

CAPTURED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Loading the committed pairs
# --------------------------------------------------------------------------------------


def _load_side(pair_dir: Path, side: dict) -> tuple[Screenshot, tuple[Element, ...], str | None]:
    """Rebuild one half of a pair as the arguments ``Fingerprinter.fingerprint`` takes."""
    screenshot = Screenshot(
        png=(pair_dir / side["png"]).read_bytes(),
        width=side["width"],
        height=side["height"],
        scale=side["scale"],
        captured_at=CAPTURED_AT,
    )
    elements = tuple(
        Element(
            box=Box(e["x"], e["y"], e["w"], e["h"]),
            kind=ElementKind(e["kind"]),
            text=e["text"],
            confidence=1.0,
            stable_id=None,
            source=ElementSource.dom,
        )
        for e in side["elements"]
    )
    return screenshot, elements, side["url"]


def _all_pairs() -> list[dict]:
    metas = [
        json.loads(path.read_text()) | {"dir": path.parent}
        for path in sorted(PAIRS_DIR.glob("*/meta.json"))
    ]
    assert metas, f"no fixture pairs under {PAIRS_DIR}; run generate.py"
    return metas


PAIRS = _all_pairs()


def _score(meta: dict) -> float:
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["a"]))
    b = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["b"]))
    assert a.similarity(b) == b.similarity(a), "similarity must be symmetric"
    return a.similarity(b)


def _shot(array: np.ndarray, *, scale: float = 1.0) -> Screenshot:
    """A ``Screenshot`` carrying ``array`` (``H x W x 3`` uint8) as its PNG."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    height, width = array.shape[:2]
    return Screenshot(
        png=buffer.getvalue(),
        width=round(width / scale),
        height=round(height / scale),
        scale=scale,
        captured_at=CAPTURED_AT,
    )


def _gradient(width: int = 800, height: int = 600, *, seed: int = 0) -> np.ndarray:
    """A deterministic image with real structure at every scale (not a flat field)."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    return np.repeat(np.repeat(coarse, height // 12, axis=0), width // 16, axis=1)


# --------------------------------------------------------------------------------------
# The threshold, over the labelled pairs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("meta", PAIRS, ids=lambda m: m["name"])
def test_labelled_pair_falls_on_the_right_side_of_the_threshold(meta: dict) -> None:
    score = _score(meta)
    if meta["label"] == "same":
        assert score > SAME_STATE_THRESHOLD, f"{meta['name']}: {meta['why']} (scored {score:.3f})"
    else:
        assert score < SAME_STATE_THRESHOLD, f"{meta['name']}: {meta['why']} (scored {score:.3f})"


def test_the_threshold_sits_in_a_wide_gap() -> None:
    """The point of the constant: there is real daylight either side of it, not a hair."""
    same = [_score(m) for m in PAIRS if m["label"] == "same"]
    different = [_score(m) for m in PAIRS if m["label"] == "different"]
    assert min(same) - max(different) >= 0.2, f"same={sorted(same)} different={sorted(different)}"
    assert min(same) - SAME_STATE_THRESHOLD > 0.1
    assert SAME_STATE_THRESHOLD - max(different) > 0.1


def test_same_screen_twice_is_exactly_one() -> None:
    """Two captures of an unchanged screen must collapse to ONE graph node, not two."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["a"]))
    b = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["b"]))
    assert a == b and hash(a) == hash(b)
    assert a.similarity(b) == 1.0
    assert len({a, b}) == 1


def test_shared_layout_does_not_carry_a_pair_over_the_threshold() -> None:
    """The trap: same URL, same structure, different content. Layout must not outvote pixels."""
    meta = next(m for m in PAIRS if m["name"] == "same_layout_different_content")
    fingerprinter = StateFingerprinter()
    a_shot, a_elements, a_url = _load_side(meta["dir"], meta["a"])
    b_shot, b_elements, b_url = _load_side(meta["dir"], meta["b"])
    a = fingerprinter.fingerprint(a_shot, a_elements, a_url)
    b = fingerprinter.fingerprint(b_shot, b_elements, b_url)

    layout_parts = [name for name in a.parts if name.startswith("layout.")]
    assert layout_parts, "the pair should have a structural signal at all"
    assert all(a.parts[name] == b.parts[name] for name in layout_parts)
    assert a.parts["url"] == b.parts["url"]
    assert a.similarity(b) < SAME_STATE_THRESHOLD


def test_parts_name_all_three_signals() -> None:
    """``parts`` is the explanation of a match, so every signal has to be visible in it."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    fingerprint = StateFingerprinter().fingerprint(*_load_side(meta["dir"], meta["a"]))
    names = set(fingerprint.parts)
    assert "url" in names
    assert any(name.startswith("layout.") for name in names)
    assert any(name.startswith("phash.") for name in names)


# --------------------------------------------------------------------------------------
# Silence is not agreement
# --------------------------------------------------------------------------------------


def _sparse(*bands: tuple[int, int, int, int]) -> np.ndarray:
    """A white 800x600 frame carrying only the given dark ``(y0, y1, x0, x1)`` strips."""
    frame = np.full((600, 800, 3), 255, dtype=np.uint8)
    for y0, y1, x0, x1 in bands or ((40, 80, 100, 300),):
        frame[y0:y1, x0:x1] = 30
    return frame


def test_an_empty_quadrant_emits_no_layout_part() -> None:
    """A quadrant with nothing in it is the absence of evidence, not a piece of it."""
    top_left_only = [_element(20, 40), _element(120, 90)]
    parts = StateFingerprinter().fingerprint(_shot(_gradient()), top_left_only, None).parts
    assert "layout.q0" in parts
    assert not {"layout.q1", "layout.q2", "layout.q3"} & set(parts)


def test_a_featureless_hash_row_emits_no_phash_part() -> None:
    """A band of flat background has no horizontal structure to report."""
    parts = StateFingerprinter().fingerprint(_shot(_sparse()), [], None).parts
    phash_parts = [name for name in parts if name.startswith("phash.")]
    bands = len(perceptual_hash(_shot(_sparse())).split("-"))
    assert phash_parts, "the one strip of content must still be reported"
    assert len(phash_parts) < bands, "the blank bands must not all report in"


def test_two_sparse_screens_get_no_credit_for_being_blank_in_the_same_places() -> None:
    """The regression this rule exists for: on mostly-empty screens, agreement used to be
    dominated by matching emptiness, and two different states scored 0.714."""
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(_shot(_sparse((40, 80, 100, 300))), [_element(120, 50)], None)
    b = fingerprinter.fingerprint(
        _shot(_sparse((200, 240, 430, 700))), [_element(520, 210, ElementKind.menu)], None
    )
    assert a != b
    assert a.similarity(b) < SAME_STATE_THRESHOLD


def test_suppressed_chunks_still_count_towards_identity() -> None:
    """``parts`` hides uninformative chunks, but ``value`` must not: two frames differing
    only where nothing is reported are still two different frames."""
    quiet = _sparse()
    noisy = quiet.copy()
    noisy[500:540, 100:300] = 0  # a strip in a band that is blank in `quiet`
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(_shot(quiet), [], None)
    b = fingerprinter.fingerprint(_shot(noisy), [], None)
    assert a != b, "identity must reflect the whole signal, not just the reported parts"


# --------------------------------------------------------------------------------------
# Signal 1: the perceptual hash, alone
# --------------------------------------------------------------------------------------


def test_perceptual_hash_is_deterministic() -> None:
    shot = _shot(_gradient())
    assert perceptual_hash(shot) == perceptual_hash(shot)


def test_perceptual_hash_ignores_the_device_scale_factor() -> None:
    """A 2x Retina capture of one screen must hash like the 1x capture: only antialiasing."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_retina")
    one_x, _, _ = _load_side(meta["dir"], meta["a"])
    two_x, _, _ = _load_side(meta["dir"], meta["b"])
    assert two_x.scale == 2.0 and one_x.scale == 1.0
    assert perceptual_hash(one_x) == perceptual_hash(two_x)


def test_perceptual_hash_survives_pixel_noise() -> None:
    """Capture noise moves every pixel a little; the hash averages it away."""
    base = _gradient()
    rng = np.random.default_rng(11)
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    assert perceptual_hash(_shot(base)) == perceptual_hash(_shot(noisy))


@pytest.mark.parametrize(
    ("pair", "allowed_losses"),
    [("same_screen_scrolled", 2), ("dense_text_scrolled_slightly", 6)],
)
def test_perceptual_hash_survives_a_few_pixels_of_scroll(pair: str, allowed_losses: int) -> None:
    """A real scroll of a real page must leave most row chunks intact.

    ``dense_text_scrolled_slightly`` is the hard one and gets a looser budget: a page of
    solid body text moves every line of type when it scrolls, so some bands do change.
    """
    meta = next(m for m in PAIRS if m["name"] == pair)
    still, _, _ = _load_side(meta["dir"], meta["a"])
    scrolled, _, _ = _load_side(meta["dir"], meta["b"])
    rows_still = perceptual_hash(still).split("-")
    rows_scrolled = perceptual_hash(scrolled).split("-")
    agreeing = sum(x == y for x, y in zip(rows_still, rows_scrolled, strict=True))
    assert agreeing >= len(rows_still) - allowed_losses, (
        f"{agreeing}/{len(rows_still)} row chunks survived"
    )


def test_perceptual_hash_separates_two_pages_of_different_prose() -> None:
    """The counterweight to the scroll tolerance above: smoothing must not blur two
    different articles in one template into the same hash."""
    meta = next(m for m in PAIRS if m["name"] == "dense_text_different_article")
    one, _, _ = _load_side(meta["dir"], meta["a"])
    other, _, _ = _load_side(meta["dir"], meta["b"])
    rows_one = perceptual_hash(one).split("-")
    rows_other = perceptual_hash(other).split("-")
    agreeing = sum(x == y for x, y in zip(rows_one, rows_other, strict=True))
    assert agreeing < len(rows_one) // 2, f"{agreeing}/{len(rows_one)} row chunks agreed"


def test_perceptual_hash_separates_genuinely_different_screens() -> None:
    meta = next(m for m in PAIRS if m["name"] == "different_screens")
    invoices, _, _ = _load_side(meta["dir"], meta["a"])
    settings, _, _ = _load_side(meta["dir"], meta["b"])
    assert perceptual_hash(invoices) != perceptual_hash(settings)


def test_perceptual_hash_is_stable_on_a_flat_field() -> None:
    """Flat regions have no true left-right difference; without a deadzone the bits between
    them would be decided by rounding noise, so near-identical flats must still agree."""
    flat = np.full((600, 800, 3), 240, dtype=np.uint8)
    rng = np.random.default_rng(3)
    jittered = np.clip(flat.astype(np.int16) + rng.normal(0, 1.5, flat.shape), 0, 255).astype(
        np.uint8
    )
    assert perceptual_hash(_shot(flat)) == perceptual_hash(_shot(jittered))


def test_perceptual_hash_handles_a_frame_smaller_than_its_grid() -> None:
    """A 4x4 icon has fewer rows than the hash has bands: degrade, do not raise."""
    rng = np.random.default_rng(5)
    tiny = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    expected_bands = len(perceptual_hash(_shot(_gradient())).split("-"))
    assert len(perceptual_hash(_shot(tiny)).split("-")) == expected_bands


def test_perceptual_hash_raises_perception_error_on_undecodable_png() -> None:
    broken = Screenshot(png=b"not a png", width=800, height=600, scale=1.0, captured_at=CAPTURED_AT)
    with pytest.raises(PerceptionError):
        perceptual_hash(broken)


# --------------------------------------------------------------------------------------
# Signal 2: the structural hash, alone
# --------------------------------------------------------------------------------------


def _element(x: int, y: int, kind: ElementKind = ElementKind.button, text: str = "") -> Element:
    return Element(Box(x, y, 80, 32), kind, text, 1.0, None, ElementSource.merged)


def test_structural_hash_ignores_text() -> None:
    """Text is the volatile part: a row's label changing is not a change of screen."""
    a = [_element(20, 100, text="Globex"), _element(20, 200, text="Initech")]
    b = [_element(20, 100, text="Hooli"), _element(20, 200, text="Pied Piper")]
    assert structural_hash(a, 800, 600) == structural_hash(b, 800, 600)


def test_structural_hash_ignores_element_order() -> None:
    elements = [_element(20, 100), _element(600, 400, ElementKind.link)]
    assert structural_hash(elements, 800, 600) == structural_hash(elements[::-1], 800, 600)


def test_structural_hash_ignores_a_few_pixels_of_drift() -> None:
    """Positions are quantized to a coarse grid, so a focus ring nudging a box is invisible."""
    a = [_element(20, 100), _element(600, 400, ElementKind.link)]
    b = [_element(23, 104), _element(603, 397, ElementKind.link)]
    assert structural_hash(a, 800, 600) == structural_hash(b, 800, 600)


def test_structural_hash_notices_a_new_element() -> None:
    a = [_element(20, 100)]
    b = [_element(20, 100), _element(300, 300, ElementKind.menu)]
    assert structural_hash(a, 800, 600) != structural_hash(b, 800, 600)


def test_structural_hash_notices_a_changed_kind() -> None:
    a = [_element(20, 100, ElementKind.button)]
    b = [_element(20, 100, ElementKind.text_field)]
    assert structural_hash(a, 800, 600) != structural_hash(b, 800, 600)


def test_structural_hash_notices_a_move_across_the_screen() -> None:
    a = [_element(20, 100)]
    b = [_element(600, 480)]
    assert structural_hash(a, 800, 600) != structural_hash(b, 800, 600)


def test_structural_hash_is_quadrant_local() -> None:
    """One quadrant changing must leave the other three intact, so a diff is attributable."""
    a = [_element(20, 40), _element(600, 40), _element(20, 480), _element(600, 480)]
    b = [*a, _element(700, 500, ElementKind.menu)]
    quadrants_a = structural_hash(a, 800, 600).split("-")
    quadrants_b = structural_hash(b, 800, 600).split("-")
    differing = [i for i, (x, y) in enumerate(zip(quadrants_a, quadrants_b, strict=True)) if x != y]
    assert differing == [3], f"expected only the bottom-right quadrant to move, got {differing}"


def test_structural_hash_of_an_empty_element_list_degrades() -> None:
    assert structural_hash([], 800, 600) == structural_hash([], 800, 600)
    assert structural_hash([], 800, 600) != structural_hash([_element(20, 100)], 800, 600)


def test_structural_hash_survives_a_degenerate_viewport() -> None:
    """A zero-size viewport is nonsense, but it must not divide by zero."""
    assert structural_hash([_element(20, 100)], 0, 0)


# --------------------------------------------------------------------------------------
# Signal 3: the normalized URL, alone
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://shop.test/cart", "https://shop.test/cart"),
        ("https://shop.test/cart?ref=email#top", "https://shop.test/cart"),
        ("HTTPS://SHOP.TEST/Cart", "https://shop.test/Cart"),
        ("https://shop.test:443/cart", "https://shop.test/cart"),
        ("https://shop.test:8443/cart", "https://shop.test:8443/cart"),
        ("https://shop.test/cart/", "https://shop.test/cart"),
        ("https://shop.test/", "https://shop.test/"),
        ("https://shop.test/orders/12345", "https://shop.test/orders/{id}"),
        ("https://shop.test/orders/12345/items", "https://shop.test/orders/{id}/items"),
        (
            "https://shop.test/s/0f8b3c1d-1111-2222-3333-444455556666",
            "https://shop.test/s/{id}",
        ),
        ("https://shop.test/t/deadbeefcafe1234", "https://shop.test/t/{id}"),
        ("https://shop.test/t/abc123", "https://shop.test/t/abc123"),
        ("file:///Users/x/pages/invoices.html", "file:///Users/x/pages/invoices.html"),
    ],
)
def test_normalize_url_reduces_a_url_to_a_state_pattern(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


@pytest.mark.parametrize("url", [None, "", "   "])
def test_normalize_url_returns_none_when_there_is_no_url(url: str | None) -> None:
    assert normalize_url(url) is None


def test_normalize_url_does_not_raise_on_nonsense() -> None:
    """A desktop controller can hand over anything; never make the caller catch."""
    for nonsense in (
        "not a url",
        "http://[::1",
        "://",
        "%%%%",
        "https://shop.test:notaport/cart",  # urlsplit defers this until .port is read
        "https://shop.test:99999999/cart",
    ):
        normalize_url(nonsense)


def test_two_urls_differing_only_in_query_share_a_pattern() -> None:
    assert normalize_url("https://shop.test/search?q=cats") == normalize_url(
        "https://shop.test/search?q=dogs"
    )


# --------------------------------------------------------------------------------------
# Degradation: the fingerprinter as a whole
# --------------------------------------------------------------------------------------


def test_fingerprint_without_a_url_degrades_rather_than_raising() -> None:
    """A desktop controller has no URL at all; the other two signals must carry it."""
    shot = _shot(_gradient())
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(shot, [_element(20, 100)], None)
    b = fingerprinter.fingerprint(shot, [_element(20, 100)], None)
    assert "url" not in a.parts
    assert a == b and a.similarity(b) == 1.0


def test_fingerprint_with_an_empty_element_list_degrades_rather_than_raising() -> None:
    """A detector finding nothing must not invent a structural match between two screens."""
    fingerprinter = StateFingerprinter()
    url = "https://shop.test/cart"
    a = fingerprinter.fingerprint(_shot(_gradient(seed=1)), [], url)
    b = fingerprinter.fingerprint(_shot(_gradient(seed=2)), [], url)
    assert not any(name.startswith("layout.") for name in a.parts)
    assert a != b
    assert a.similarity(b) < SAME_STATE_THRESHOLD


def test_fingerprint_with_neither_url_nor_elements_still_identifies_the_pixels() -> None:
    fingerprinter = StateFingerprinter()
    shot = _shot(_gradient())
    assert fingerprinter.fingerprint(shot, [], None) == fingerprinter.fingerprint(shot, [], None)
    assert fingerprinter.fingerprint(shot, [], None) != fingerprinter.fingerprint(
        _shot(_gradient(seed=9)), [], None
    )


def test_fingerprint_raises_perception_error_on_an_undecodable_screenshot() -> None:
    broken = Screenshot(png=b"", width=800, height=600, scale=1.0, captured_at=CAPTURED_AT)
    with pytest.raises(PerceptionError):
        StateFingerprinter().fingerprint(broken, [], None)


def test_fingerprint_is_deterministic_across_instances() -> None:
    """Different runs, different processes: the same screen must land on the same node id."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    args = _load_side(meta["dir"], meta["a"])
    assert StateFingerprinter().fingerprint(*args).value == (
        StateFingerprinter().fingerprint(*args).value
    )


def test_state_fingerprinter_satisfies_the_fingerprinter_protocol() -> None:
    from skillweaver.contracts import Fingerprinter

    assert isinstance(StateFingerprinter(), Fingerprinter)
