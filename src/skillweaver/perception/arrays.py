"""One decode of a screenshot, shared by everything that looks at it.

A :class:`~skillweaver.contracts.Screenshot` carries PNG bytes and decodes them on
every :meth:`~skillweaver.contracts.Screenshot.to_array` call, which is the right
default for a value type: it owns no cache and can be passed anywhere. But one
observation hands the SAME frame to three readers in a row - the detector, the OCR
pass and the fingerprinter - so a plain observation pays for three decodes of
identical bytes, and on a Retina frame the logical one also pays a LANCZOS resize.

:func:`array_of` is that decode, memoized on the PNG bytes themselves. Keyed by
content rather than by object identity, so a screenshot rebuilt from the same bytes
hits the cache and a screenshot whose bytes differ can never collide.

The cache is small on purpose. Perception looks at one frame at a time and the
before/after pair of a critic call is the widest window that matters, so a handful of
entries catches everything worth catching while a large cache of decoded frames would
hold tens of megabytes of pixels alive for no gain.

Returned arrays are treated as READ-ONLY by every caller here and are marked
non-writeable, because handing the same array to two readers and letting one of them
modify it in place is the kind of bug that shows up as a detection that only fails
when OCR ran first.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from skillweaver.contracts import Screenshot

__all__ = ["CACHE_ENTRIES", "array_of", "cache_info", "clear_cache"]

CACHE_ENTRIES = 6
"""Decoded frames kept. Two observations' worth of physical and logical arrays, plus
room for a crop; beyond that a decode is cheaper than the memory."""

_lock = threading.Lock()
_cache: OrderedDict[tuple[bool, float, int, int, int, bytes], Any] = OrderedDict()
_hits = 0
_misses = 0


def array_of(screenshot: Screenshot, *, logical: bool = True) -> Any:
    """The screenshot as a numpy array, decoded once per distinct frame.

    Same contract as :meth:`~skillweaver.contracts.Screenshot.to_array`, including
    which errors it raises, with the decode memoized on the PNG bytes.

    Args:
        screenshot: The frame to decode.
        logical: ``True`` for the logical-pixel frame the rest of the system measures
            in, ``False`` for the physical pixels a model wants to run on.

    Returns:
        A read-only array. Callers that need to modify it must copy it first.

    Raises:
        PerceptionError: if the bytes are not a decodable image.
    """
    # The bytes alone are NOT the frame. Asked for the logical view, ``to_array``
    # resizes by the screenshot's own scale and logical size, so two screenshots
    # carrying identical PNG bytes at different scales have different right answers
    # and keying on the bytes alone would serve one where the other was asked for -
    # halving every coordinate derived from it. The length leads so that two frames
    # of different sizes are separated before their bytes are ever compared.
    key = (
        logical,
        screenshot.scale,
        screenshot.width,
        screenshot.height,
        len(screenshot.png),
        screenshot.png,
    )
    global _hits, _misses
    with _lock:
        found = _cache.get(key)
        if found is not None:
            _cache.move_to_end(key)
            _hits += 1
            return found

    # Decoded outside the lock: it is the expensive part and a second thread doing it
    # concurrently wastes one decode, where holding the lock would serialise every
    # reader in the process behind whichever one arrived first.
    array = screenshot.to_array(logical=logical)
    try:
        array.flags.writeable = False
    except (AttributeError, ValueError):  # a non-numpy array, or one that owns no buffer
        pass

    with _lock:
        _misses += 1
        _cache[key] = array
        _cache.move_to_end(key)
        while len(_cache) > CACHE_ENTRIES:
            _cache.popitem(last=False)
    return array


def cache_info() -> dict[str, int]:
    """Hits, misses and current size - for a benchmark or a test, not for logic."""
    with _lock:
        return {"hits": _hits, "misses": _misses, "entries": len(_cache)}


def clear_cache() -> None:
    """Forget every decoded frame. Only useful to a test measuring cold decodes."""
    global _hits, _misses
    with _lock:
        _cache.clear()
        _hits = _misses = 0
