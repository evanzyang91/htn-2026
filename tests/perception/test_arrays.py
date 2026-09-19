"""The shared decode: one frame, decoded once, handed to every reader that wants it.

An observation shows the same screenshot to the detector, the OCR pass and the
fingerprinter. Each used to decode the PNG itself, so a plain observation paid three
decodes of identical bytes - and on a Retina frame the logical one also paid a resize.
These tests pin the two things that makes safe: the cache is keyed by CONTENT, so it
can never answer with another frame's pixels, and what it hands back is identical to
what the screenshot itself would have produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from skillweaver.contracts import Screenshot, utcnow
from skillweaver.errors import PerceptionError
from skillweaver.perception import arrays
from skillweaver.perception.screenshot import encode_png, from_array


@pytest.fixture(autouse=True)
def _empty_cache():
    """Every test starts with nothing remembered, and leaves nothing behind."""
    arrays.clear_cache()
    yield
    arrays.clear_cache()


def _frame(value: int, *, width: int = 24, height: int = 16, scale: float = 1.0) -> Screenshot:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    pixels[0, 0] = (value, 255 - value, (value * 7) % 256)  # make each frame distinguishable
    return from_array(pixels, scale=scale)


def test_the_same_bytes_decode_once_however_often_they_are_asked_for() -> None:
    shot = _frame(40)
    first = arrays.array_of(shot, logical=False)
    second = arrays.array_of(shot, logical=False)

    assert second is first, "the second reader gets the array the first one decoded"
    assert arrays.cache_info() == {"hits": 1, "misses": 1, "entries": 1}


def test_a_screenshot_rebuilt_from_the_same_bytes_still_hits() -> None:
    """Keyed by content, not by object identity: a screenshot that was serialised and
    read back is the same frame and must not be decoded twice."""
    shot = _frame(55)
    twin = Screenshot(
        png=shot.png,
        width=shot.width,
        height=shot.height,
        scale=shot.scale,
        captured_at=shot.captured_at,
    )

    assert arrays.array_of(twin) is arrays.array_of(shot)
    assert arrays.cache_info()["misses"] == 1


def test_different_frames_never_share_an_entry() -> None:
    a, b = _frame(10), _frame(200)
    assert not np.array_equal(arrays.array_of(a), arrays.array_of(b))
    assert arrays.cache_info()["misses"] == 2


def test_the_logical_and_physical_views_are_cached_apart() -> None:
    """A Retina frame has two different right answers, and one must never stand in for
    the other: a logical array served where physical was asked for would halve every
    coordinate the detector reports."""
    shot = _frame(90, width=48, height=32, scale=2.0)

    physical = arrays.array_of(shot, logical=False)
    logical = arrays.array_of(shot, logical=True)

    assert physical.shape[:2] == (32, 48)
    assert logical.shape[:2] == (16, 24)
    assert arrays.cache_info()["entries"] == 2


def test_what_comes_back_is_what_the_screenshot_itself_would_have_produced() -> None:
    for scale, logical in ((1.0, True), (1.0, False), (2.0, True), (2.0, False)):
        shot = _frame(120, width=40, height=20, scale=scale)
        assert np.array_equal(
            arrays.array_of(shot, logical=logical), shot.to_array(logical=logical)
        ), f"scale={scale} logical={logical}"


def test_the_cached_array_is_read_only_so_one_reader_cannot_corrupt_another() -> None:
    shot = _frame(77)
    array = arrays.array_of(shot)
    with pytest.raises(ValueError):
        array[0, 0] = 0


def test_the_cache_is_bounded_and_forgets_the_oldest_frame() -> None:
    frames = [_frame(value) for value in range(arrays.CACHE_ENTRIES + 2)]
    for shot in frames:
        arrays.array_of(shot)

    assert arrays.cache_info()["entries"] == arrays.CACHE_ENTRIES
    before = arrays.cache_info()["misses"]
    arrays.array_of(frames[0])  # the oldest was evicted, so this is a fresh decode
    assert arrays.cache_info()["misses"] == before + 1
    arrays.array_of(frames[-1])  # the newest is still there
    assert arrays.cache_info()["misses"] == before + 1


def test_undecodable_bytes_still_raise_rather_than_being_remembered() -> None:
    """A failure must not be cached as a success, and must keep the perception error
    the rest of the stack is written against."""
    broken = Screenshot(
        png=b"not a png at all", width=10, height=10, scale=1.0, captured_at=utcnow()
    )
    with pytest.raises(PerceptionError):
        arrays.array_of(broken)
    with pytest.raises(PerceptionError):
        arrays.array_of(broken)
    assert arrays.cache_info()["entries"] == 0


def test_a_frame_whose_bytes_differ_only_at_one_pixel_is_a_different_frame() -> None:
    """The digest is over the PNG bytes, so this is really a test that nothing
    truncates or samples them."""
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    first = Screenshot(png=encode_png(pixels), width=8, height=8, scale=1.0, captured_at=utcnow())
    pixels[7, 7] = (1, 2, 3)
    second = Screenshot(png=encode_png(pixels), width=8, height=8, scale=1.0, captured_at=utcnow())

    assert arrays.array_of(first) is not arrays.array_of(second)
    assert arrays.cache_info()["misses"] == 2
