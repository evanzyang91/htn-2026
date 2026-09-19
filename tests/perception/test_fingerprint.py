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
    band_parts,
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


def _column_page(height: int = 800, width: int = 1280) -> np.ndarray:
    """A deterministic stand-in for a page: header, left rail, a column of body lines.

    Used ONLY to compare a frame with ITSELF under some transformation - a taller
    viewport, a shift. It is deliberately not used to argue that two DIFFERENT pages stay
    apart: a hand-drawn frame shares far more with another hand-drawn frame than two real
    pages share, and measuring against one would flatter the fingerprint. Two of these
    "articles" score 0.325 to 0.496 against each other where two real Wikipedia articles
    score 0.034, so the different-page claims are made against real captures instead.
    """
    frame = np.full((height, width, 3), 250, dtype=np.uint8)
    frame[:52] = 240  # header bar
    frame[20:36, 24:180] = 60  # wordmark
    frame[20:36, 400:900] = 210  # search box
    frame[64:, :210] = 246  # left rail
    for y in range(80, height - 20, 34):  # rail links
        frame[y : y + 12, 24:170] = 90
    for y in range(80, height - 20, 26):  # body lines
        frame[y : y + 13, 260 : 1040 - (y * 37 % 260)] = 40
    return frame


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


def test_the_threshold_sits_in_a_gap_on_this_corpus() -> None:
    """The point of the constant: there is daylight either side of it, not a hair.

    These are the CONTRIVED pairs, which separate widely - they were built to. The live
    captures behind ``SAME_STATE_THRESHOLD`` are what makes the gap narrow, and the
    binding pair is cross-corpus: a live page whose advertisement rotated (0.305)
    against ``same_layout_different_content`` here (0.213). Both halves are asserted,
    here and in :func:`test_a_page_pushed_down_by_a_notice_is_still_the_same_screen`.
    """
    same = [_score(m) for m in PAIRS if m["label"] == "same"]
    different = [_score(m) for m in PAIRS if m["label"] == "different"]
    assert min(same) - max(different) >= 0.4, f"same={sorted(same)} different={sorted(different)}"
    assert min(same) - SAME_STATE_THRESHOLD > 0.3
    assert SAME_STATE_THRESHOLD - max(different) > 0.04


def test_same_screen_twice_is_exactly_one() -> None:
    """Two captures of an unchanged screen must collapse to ONE graph node, not two."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    fingerprinter = StateFingerprinter()
    a = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["a"]))
    b = fingerprinter.fingerprint(*_load_side(meta["dir"], meta["b"]))
    assert a == b and hash(a) == hash(b)
    assert a.similarity(b) == 1.0
    assert len({a, b}) == 1


def test_shared_layout_and_url_do_not_carry_a_pair_over_the_threshold() -> None:
    """The trap this cut exists for: same URL, same chrome, same element layout, every
    row's text different - one list for two different accounts.

    It is the hardest DIFFERENT pair anywhere in the calibration, and it is what stops
    the threshold being lowered further. Admitting it would let a skill run against the
    wrong account's data.
    """
    meta = next(m for m in PAIRS if m["name"] == "same_layout_different_content")
    fingerprinter = StateFingerprinter()
    a_shot, a_elements, a_url = _load_side(meta["dir"], meta["a"])
    b_shot, b_elements, b_url = _load_side(meta["dir"], meta["b"])
    a = fingerprinter.fingerprint(a_shot, a_elements, a_url)
    b = fingerprinter.fingerprint(b_shot, b_elements, b_url)

    assert structural_hash(a_elements, a_shot.width, a_shot.height) == structural_hash(
        b_elements, b_shot.width, b_shot.height
    ), "the pair is only interesting while its layouts really are identical"
    assert a.parts["url"] == b.parts["url"]
    assert a.similarity(b) < SAME_STATE_THRESHOLD


def test_the_element_index_contributes_no_part() -> None:
    """Measured, not assumed: quadrant parts cost at BOTH ends on the live corpus, and a
    detector that reads 47 elements on one capture and 31 on the next reports that as
    disagreement. The elements still reach ``value`` - see the next test."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    shot, elements, url = _load_side(meta["dir"], meta["a"])
    with_elements = StateFingerprinter().fingerprint(shot, elements, url)
    without = StateFingerprinter().fingerprint(shot, [], url)
    assert with_elements.parts == without.parts
    assert with_elements != without, "but they must still be different FRAMES"


def test_parts_name_both_signals() -> None:
    """``parts`` is the explanation of a match, so every signal has to be visible in it."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    fingerprint = StateFingerprinter().fingerprint(*_load_side(meta["dir"], meta["a"]))
    names = set(fingerprint.parts)
    assert "url" in names
    assert any(name.startswith("band.") for name in names)
    assert not any(name.startswith(("layout.", "phash.")) for name in names)


# --------------------------------------------------------------------------------------
# Silence is not agreement
# --------------------------------------------------------------------------------------


def _sparse(*bands: tuple[int, int, int, int]) -> np.ndarray:
    """A white 800x600 frame carrying only the given dark ``(y0, y1, x0, x1)`` strips."""
    frame = np.full((600, 800, 3), 255, dtype=np.uint8)
    for y0, y1, x0, x1 in bands or ((40, 80, 100, 300),):
        frame[y0:y1, x0:x1] = 30
    return frame


def test_a_featureless_band_emits_no_part() -> None:
    """A band of flat background has no horizontal structure to report."""
    shot = _shot(_sparse())
    parts = band_parts(shot)
    assert parts, "the one strip of content must still be reported"
    assert len(parts) < len(perceptual_hash(shot).split("-")), "blank bands must not report in"


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
# The defect: a page that moved is still the page
# --------------------------------------------------------------------------------------


def _page_shifted_down(shot: Screenshot, pixels: int, fill: int = 235) -> Screenshot:
    """``shot`` with a notice ``pixels`` tall pushed onto the top, exactly as a
    fundraising appeal, a cookie bar or a logged-out prompt does it."""
    frame = np.asarray(Image.open(io.BytesIO(shot.png)).convert("RGB"))
    moved = np.empty_like(frame)
    moved[:pixels] = fill
    moved[pixels:] = frame[: frame.shape[0] - pixels]
    return _shot(moved, scale=shot.scale)


@pytest.mark.parametrize(("pixels", "floor"), [(20, 0.95), (60, 0.85), (120, 0.70), (200, 0.55)])
def test_a_page_pushed_down_by_a_notice_is_still_the_same_screen(pixels: int, floor: float) -> None:
    """THE regression. A notice arriving at the top displaces everything below it, and
    under a fingerprint whose parts were named by grid ROW that scored a live page
    against itself at 0.040 - the same as two unrelated websites - so the graph grew a
    fresh node for a screen it already knew, every single visit.

    The floors fall with the size of the notice, because a notice really does hide that
    much of the screen; what matters is that they stay far above the cut.
    """
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    shot, _, url = _load_side(meta["dir"], meta["a"])
    fingerprinter = StateFingerprinter()
    before = fingerprinter.fingerprint(shot, [], url)
    after = fingerprinter.fingerprint(_page_shifted_down(shot, pixels), [], url)

    score = before.similarity(after)
    assert score >= floor, f"a {pixels}px notice scored the page against itself at {score:.3f}"
    assert score > SAME_STATE_THRESHOLD
    assert before != after, "it is still a different FRAME, and value must say so"


def test_the_same_screen_that_moved_keeps_its_parts_but_not_its_value() -> None:
    """The two jobs of a fingerprint, pulled apart: ``value`` is a node id and must
    change when anything does, ``parts`` is the recognition and must not."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    shot, _, url = _load_side(meta["dir"], meta["a"])
    moved = _page_shifted_down(shot, 60)
    assert perceptual_hash(shot) != perceptual_hash(moved), "the POSITIONAL hash does move"
    kept = set(band_parts(shot)) & set(band_parts(moved))
    assert len(kept) > 0.7 * len(band_parts(shot))


def test_a_band_part_is_named_by_its_content_and_never_by_its_position() -> None:
    """The mechanism, stated directly: the name of a part carries no ``y``."""
    shot = _shot(_sparse((100, 140, 100, 300)))
    moved = _shot(_sparse((300, 340, 100, 300)))
    assert set(band_parts(shot)) == set(band_parts(moved))
    for name, value in band_parts(shot).items():
        assert name == f"band.{value}#{name.rsplit('#', 1)[1]}"


def test_occupancy_still_counts_but_only_in_powers_of_two() -> None:
    """Position is thrown away; how MUCH of the screen a pattern covers is not. Without
    it, two lists that share a layout and differ only in how far each block runs would
    collapse into one screen."""
    short = band_parts(_shot(_sparse((100, 140, 100, 300))))
    tall = band_parts(_shot(_sparse((100, 400, 100, 300))))
    assert set(short) < set(tall), "a taller run of one pattern must add parts, not replace them"
    nudged = band_parts(_shot(_sparse((100, 152, 100, 300))))
    assert set(short) == set(nudged), "but 12px more of it must cost nothing"


@pytest.mark.parametrize(
    ("pair", "what"),
    [
        ("different_screens", "an index page against a settings page: nothing in common"),
        ("article_vs_search_results", "an article against the results page for a search"),
        ("dense_text_different_article", "one template, entirely different prose"),
        ("modal_open_vs_closed", "a dialog and its scrim over the page beneath"),
        ("same_layout_different_content", "one list for two accounts: same URL, chrome, layout"),
    ],
)
def test_differently_laid_out_pages_stay_different_screens(pair: str, what: str) -> None:
    """Tolerance must not become "everything is one node".

    These are REAL browser captures, which is the point - the claim is about pages, and
    a hand-drawn stand-in is not a page. Live captures agree: an article against a page
    of search results scores 0.036, two different Wikipedia articles 0.034, and two
    search-result pages for different queries 0.189 - that last being the highest any
    genuinely different pair reached anywhere in the calibration.
    """
    meta = next(m for m in PAIRS if m["name"] == pair)
    assert meta["label"] == "different", f"{pair} is not a different-state pair"
    assert _score(meta) < SAME_STATE_THRESHOLD, what


def test_an_article_and_a_page_of_search_results_stay_different_screens() -> None:
    """Named on its own because it is the pair a tolerant identity is likeliest to
    collapse, and the one the graph would be worst served by collapsing.

    The two pages share a site, a header, a toolbar and a stylesheet; only the body
    differs - prose against a column of hits. They score 0.054 here and 0.036 on live
    Wikipedia. Merging them would give the router an edge out of "the article" that
    actually leaves from the results page, and a skill written for one would run on the
    other.
    """
    meta = next(m for m in PAIRS if m["name"] == "article_vs_search_results")
    article = StateFingerprinter().fingerprint(*_load_side(meta["dir"], meta["a"]))
    results = StateFingerprinter().fingerprint(*_load_side(meta["dir"], meta["b"]))
    assert article.similarity(results) < SAME_STATE_THRESHOLD


def test_the_results_page_pushed_down_is_still_the_results_page() -> None:
    """And the counterweight, on the very same page: a notice at the top moves every hit
    down 120px and it is still the screen a skill was written for.

    Together these two are the whole claim of this module. One real page moved scores
    0.659; two real pages that merely share a template score 0.054. A cut between them
    exists - which, before parts were named by content, it did not.
    """
    moved = next(m for m in PAIRS if m["name"] == "search_results_pushed_down")
    apart = next(m for m in PAIRS if m["name"] == "article_vs_search_results")
    assert _score(moved) > SAME_STATE_THRESHOLD
    assert _score(moved) > _score(apart) + 0.4


def test_a_notice_does_not_turn_a_page_into_a_different_page() -> None:
    """The two properties together, which is the whole claim: the SAME page that moved
    stays closer to itself than two DIFFERENT pages ever get to each other."""
    fingerprinter = StateFingerprinter()
    meta = next(m for m in PAIRS if m["name"] == "dense_text_different_article")
    here, _, url = _load_side(meta["dir"], meta["a"])
    other, _, other_url = _load_side(meta["dir"], meta["b"])
    a = fingerprinter.fingerprint(here, [], url)
    moved = fingerprinter.fingerprint(_page_shifted_down(here, 120), [], url)
    elsewhere = fingerprinter.fingerprint(other, [], other_url)
    assert a.similarity(moved) > a.similarity(elsewhere) + 0.4


# --------------------------------------------------------------------------------------
# Signal 1: the band hash, alone
# --------------------------------------------------------------------------------------


def test_perceptual_hash_is_deterministic() -> None:
    shot = _shot(_gradient())
    assert perceptual_hash(shot) == perceptual_hash(shot)


def test_a_retina_capture_is_the_same_screen_as_the_one_x_capture() -> None:
    """A 2x capture of one screen differs from the 1x capture only by resampling, so it
    must resolve to the same node. It is no longer bit-identical - 32 columns across the
    viewport is fine enough to see the resampling - so this is a similarity claim."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_retina")
    one_x, _, url = _load_side(meta["dir"], meta["a"])
    two_x, _, _ = _load_side(meta["dir"], meta["b"])
    assert two_x.scale == 2.0 and one_x.scale == 1.0
    fingerprinter = StateFingerprinter()
    score = fingerprinter.fingerprint(one_x, [], url).similarity(
        fingerprinter.fingerprint(two_x, [], url)
    )
    assert score > SAME_STATE_THRESHOLD + 0.3, f"scored {score:.3f}"


@pytest.mark.parametrize(("sigma", "floor"), [(1.5, 0.90), (6, 0.55), (24, 0.40)])
def test_pixel_noise_does_not_lose_the_screen(sigma: float, floor: float) -> None:
    """A lossy or noisy capture path moves every pixel a little. The deadzone and the
    band averaging absorb it; at 32 columns they no longer absorb it EXACTLY, so what is
    asserted is that the screen is still recognisable, not that the hash is unchanged."""
    meta = next(m for m in PAIRS if m["name"] == "same_screen_twice")
    shot, _, url = _load_side(meta["dir"], meta["a"])
    frame = np.asarray(Image.open(io.BytesIO(shot.png)).convert("RGB"))
    rng = np.random.default_rng(11)
    noisy = np.clip(frame.astype(np.int16) + rng.normal(0, sigma, frame.shape), 0, 255).astype(
        np.uint8
    )
    fingerprinter = StateFingerprinter()
    score = fingerprinter.fingerprint(shot, [], url).similarity(
        fingerprinter.fingerprint(_shot(noisy), [], url)
    )
    assert score >= floor, f"sigma {sigma} scored {score:.3f}"
    assert score > SAME_STATE_THRESHOLD


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


def test_perceptual_hash_handles_a_frame_shorter_than_one_band() -> None:
    """A 4x4 icon is shorter than a single band is tall: degrade, do not raise.

    Bands are a fixed number of LOGICAL pixels apart rather than a fixed count, which is
    what lets a 900px viewport line up with an 800px one; the price is that a frame can
    be too short to fill even one, and numpy's ``same`` convolution silently returns the
    wrong length there.
    """
    rng = np.random.default_rng(5)
    tiny = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    assert len(perceptual_hash(_shot(tiny)).split("-")) == 1
    assert len(perceptual_hash(_shot(_gradient())).split("-")) > 1


def test_a_taller_viewport_lines_up_with_a_shorter_one() -> None:
    """Bands are pitched in logical pixels, not cut into a fixed number per frame, so a
    window resized taller shows the same screen with more of it rather than a rescaled
    and unrecognisable one."""
    short = _column_page(height=800)
    tall = np.full((900, 1280, 3), 250, dtype=np.uint8)
    tall[:800] = short
    fingerprinter = StateFingerprinter()
    score = fingerprinter.fingerprint(_shot(short), [], None).similarity(
        fingerprinter.fingerprint(_shot(tall), [], None)
    )
    assert score > SAME_STATE_THRESHOLD + 0.3, f"scored {score:.3f}"


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
    assert a.similarity(b) < SAME_STATE_THRESHOLD, "two different screens, no borrowed match"


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
