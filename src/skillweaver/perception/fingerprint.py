"""UI state fingerprinting: the answer to "which screen am I on?".

:class:`StateFingerprinter` turns a frame into a :class:`~skillweaver.contracts.Fingerprint`
whose ``value`` is a graph node id and whose ``parts`` explain *why* two screens matched.

Three independent signals
-------------------------

**Perceptual hash** (``phash.r0``..``phash.r19``) - a difference hash computed directly on
numpy: luma, area-averaged down to a 20x17 grid, then each cell compared with its
right-hand neighbour through a deadzone. Two things make it survive a recapture of the
same screen. Averaging each cell over thousands of pixels, with a vertical window several
band-heights wide so band edges do not snap, absorbs antialiasing and a few pixels of
scroll; the deadzone stops a flat region - where the true difference between neighbours
is zero - from deciding its bit on rounding noise. One part per grid ROW, so a change
confined to part of the screen only spoils the rows it touches.

**Structural hash** (``layout.q0``..``layout.q3``) - over the element index: each element
becomes a ``kind@cell+size`` token with its centre quantized to a coarse 4x4 grid and its
size bucketed to 16 logical pixels. Text is deliberately ignored, because text is the
volatile part. Tokens are sorted per screen quadrant and one part is emitted per quadrant,
so a dialog appearing in the middle does not erase the evidence from the corners.

**Normalized URL** (``url``) - scheme and host lowercased, default port and query and
fragment dropped, volatile path segments (digits, hex ids, UUIDs) replaced by ``{id}``.
Omitted entirely when there is no URL, rather than contributing a fake match.

Why parts are chunked
---------------------

:meth:`Fingerprint.similarity` is fixed by ``contracts``: it is the fraction of part NAMES
whose sub-hashes agree. Three whole-signal parts would therefore only ever score 0, 1/3,
2/3 or 1 - too coarse to separate "scrolled a little" from "same layout, other content".
Splitting each signal into locality-preserving chunks turns that fraction into a real
gradient, and the chunk counts (up to 20 visual, 4 structural, 1 URL) are the weighting:
pixels outvote layout, because two screens sharing a layout with different content are two
different states.

Note what this costs. Chunk agreement is exact, never graded, so a perturbation that
nudges every chunk a little scores no better than one that rewrites the screen. That is
why the pixel signal is smoothed as hard as it is: it has to put a perturbed chunk back on
exactly the value it had, not merely close to it.

Silence is not agreement
------------------------

A part is only emitted for a chunk that carries evidence: a screen quadrant with no
elements in it, and a hash row with no horizontal structure in it, emit nothing. Because
``similarity`` divides by the UNION of both fingerprints' part names, two sparse screens
that are blank in the same places no longer collect credit for it - measured on the fake
invoicing app in ``tests/fakes/scenario.py``, whose states are mostly empty, this is the
difference between scoring two different states 0.714 and scoring them 0.143.

``value``, unlike ``parts``, always hashes the whole signal including the silent chunks,
so two frames are equal only when everything about them matches.

Threshold
---------

:data:`SAME_STATE_THRESHOLD` is the recommended "same state" cut. See
``tests/perception/test_fingerprint.py`` for the measured separation it is based on.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from urllib.parse import urlsplit

import numpy as np

from skillweaver.contracts import Element, Fingerprint, Screenshot
from skillweaver.errors import PerceptionError

SAME_STATE_THRESHOLD = 0.62
"""Recommended cut for "these two frames are the same UI state".

Measured over the labelled pairs in ``tests/fixtures/shots/pairs`` (800x600 browser
captures, around 24 parts each):

===============================  =====  =========================================
pair                             score  what differs
===============================  =====  =========================================
same_screen_twice                1.000  nothing
same_screen_caret_blink          1.000  the text caret's blink phase
same_screen_clock_tick           1.000  a clock digit, and a ``?t=`` query
same_screen_retina               1.000  captured at 2x instead of 1x
same_screen_no_url               1.000  no URL at all, plus a caret blink
same_screen_scrolled             0.917  scrolled 6px, pulling a new row into view
dense_text_scrolled_slightly     0.750  solid body text scrolled 8px
-------------------------------  -----  -----------------------------------------
same_layout_different_content    0.500  every row's text; layout and URL identical
modal_open_vs_closed             0.240  a dialog and its scrim
dense_text_different_article     0.167  one template, entirely different prose
different_screens                0.000  another page entirely
===============================  =====  =========================================

Same-state bottoms out at 0.750 and different-state tops out at 0.500, so 0.62 sits in the
middle of a 0.25-wide gap with about 0.13 of margin either side. Both ends are the hard
cases on purpose: the same-state floor is a page of solid text, the worst thing you can
scroll past a pixel hash, and the different-state ceiling is two screens contrived to
share their URL, their chrome and their element layout exactly.

It holds outside that corpus too. On live third-party pages the fingerprinter was never
tuned against - English Wikipedia and the Python documentation - a recapture scores 1.000,
a 2x Retina capture 0.960 and a 3-8px scroll 0.760 to 0.800, while two sister articles
with the same skin score 0.000 and unrelated pages 0.000 to 0.040. Additive pixel noise on
those pages, standing in for a lossy or noisy capture path, still scores 0.760 at sigma
40. Driving the sparse fake invoicing app of ``tests/fakes/scenario.py``, whose line-art
frames are mostly blank, separates its four reachable states at 0.333 or below while every
revisit of a state recovers its node exactly.

The known limit is scrolling, and it is worth stating precisely. On a dense page of body
text the pixel signal holds to about 8 logical pixels (0.800) and then falls off a cliff:
16px already scores 0.440 and 40px scores 0.160, both read as a different state. Ordinary
application UI, which has far more whitespace, holds further. This is a deliberate trade -
a hash blurry enough to shrug off a 40px scroll is also too blurry to notice a dialog
opening - but a caller that scrolls ON PURPOSE should track the state it scrolled from
rather than expect this threshold to tie the two captures together.
"""

# --- perceptual hash ------------------------------------------------------------------

_PHASH_ROWS = 20
"""Grid rows, and therefore the number of ``phash.rN`` parts."""

_PHASH_BITS = 16
"""Bits per row; the grid is ``_PHASH_ROWS`` x ``_PHASH_BITS + 1`` cells."""

_PHASH_OVERSAMPLE = 4
"""Fine rows computed per output row, setting the resolution of the smoothing window.

Straight area averaging would cut the frame into hard horizontal bands, and scrolling a
few pixels swaps that fraction of a band's content across a boundary - enough to flip
bits. Computing this many times more rows than are needed lets the window in
:func:`_band_weights` taper smoothly instead of snapping; :data:`_PHASH_SMOOTH` sets how
wide it tapers.
"""

_PHASH_SMOOTH = 3.0
"""Width of the vertical smoothing window, in band heights.

This is the knob that makes the hash survive scrolling a page of solid text. The window
has to be several times taller than a line of body text, so that a band's value is an
average over many lines and shifting the text by a few pixels barely moves it. At 1.0 -
smoothing just wide enough to soften the band edges - an 8px scroll of a dense
documentation page scored 0.381; at 3.0 the same pair scores 0.760, while two genuinely
different screens are unaffected because they differ at every scale.
"""

_PHASH_TOLERANCE = 2.0
"""Deadzone, in luma units (0..255), for the neighbour comparison.

A cell is only called brighter than its neighbour when it beats it by this much, so the
huge flat regions of a UI - page background, a card's fill - resolve to a stable ``0``
instead of a coin flip on rounding noise. Quantizing the cell values instead would only
move the coin flip to the quantization boundaries: measured on the fixture pages, a
step-4 quantizer was already losing a third of its rows to additive noise at sigma 1,
where this deadzone still matches exactly and holds past sigma 24.
"""

# --- structural hash ------------------------------------------------------------------

_LAYOUT_GRID = 4
"""Position quantization: element centres are binned to a ``_LAYOUT_GRID`` square grid.

Deliberately coarse - 200x150 logical pixels per cell on a 800x600 viewport. The division
of labour is that the structural signal answers "what kinds of thing, roughly where" and
the pixel signal answers "and exactly how does it look", so precision here buys nothing
and costs robustness: a finer grid mostly reports that scrolling moved a box across a cell
boundary. Measured over the fixtures and the live pages, widening the cells from 50px to
200px lifts the worst same-state pair from 0.667 to 0.750 and moves no different-state
pair at all.
"""

_LAYOUT_BANDS = 2
"""Quadrant split per axis, so ``_LAYOUT_BANDS ** 2`` ``layout.qN`` parts."""

_SIZE_BUCKET = 16
"""Element width/height are rounded to this many logical pixels before hashing."""

# --- URL ------------------------------------------------------------------------------

_DEFAULT_PORTS = {"http": "80", "https": "443", "ftp": "21", "ws": "80", "wss": "443"}

_VOLATILE_SEGMENT = re.compile(
    r"""^(
        \d+                                                     # 42
        | [0-9a-fA-F]{8,}                                        # deadbeefcafe
        | [0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}     # a UUID
    )$""",
    re.VERBOSE,
)

_ID_PLACEHOLDER = "{id}"


def _digest(*chunks: str) -> str:
    """A short, stable digest of ``chunks``. ``\\x1f`` keeps the joins unambiguous."""
    return hashlib.sha256("\x1f".join(chunks).encode()).hexdigest()[:16]


def _area_downsample(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Average ``gray`` down to ``out_h`` x ``out_w`` by box (area) averaging.

    Averaging rather than sampling is what makes the hash ignore antialiasing and a few
    pixels of scroll: every output cell is the mean of thousands of input pixels, so
    moving the content by a handful of rows barely moves the mean.
    """
    height, width = gray.shape
    if height < out_h or width < out_w:
        # Tiny frame: repeat pixels up to at least the grid size, then average down.
        gray = np.repeat(
            np.repeat(gray, -(-out_h // max(height, 1)), axis=0), -(-out_w // max(width, 1)), axis=1
        )
        height, width = gray.shape
    y_edges = np.linspace(0, height, out_h + 1).astype(int)
    x_edges = np.linspace(0, width, out_w + 1).astype(int)
    rows = np.add.reduceat(gray, y_edges[:-1], axis=0) / np.diff(y_edges)[:, None]
    cells = np.add.reduceat(rows, x_edges[:-1], axis=1) / np.diff(x_edges)[None, :]
    return cells


def _band_weights(rows: int, oversample: int, smooth: float = 1.0) -> np.ndarray:
    """A ``rows x (rows * oversample)`` triangular smoothing matrix, each row summing to 1.

    Output band ``r`` is centred on the fine rows it would have owned outright, and its
    weight decays linearly to zero ``smooth`` band-heights either side.
    """
    fine = rows * oversample
    centres = np.arange(rows) * oversample + (oversample - 1) / 2.0
    distance = np.abs(np.arange(fine)[None, :] - centres[:, None])
    weights = np.clip(1.0 - distance / (oversample * smooth), 0.0, None)
    return weights / weights.sum(axis=1, keepdims=True)


def _phash_rows(screenshot: Screenshot) -> list[str]:
    """One hex string per grid row of the difference hash, top row first.

    Raises:
        PerceptionError: if the screenshot cannot be decoded or hashed.
    """
    try:
        image = screenshot.to_array()
        if image.size == 0:
            return ["0" * ((_PHASH_BITS + 3) // 4)] * _PHASH_ROWS
        gray = image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        fine = _area_downsample(gray, _PHASH_ROWS * _PHASH_OVERSAMPLE, _PHASH_BITS + 1)
        cells = _band_weights(_PHASH_ROWS, _PHASH_OVERSAMPLE, _PHASH_SMOOTH) @ fine
        bits = (cells[:, 1:] - cells[:, :-1]) > _PHASH_TOLERANCE
    except PerceptionError:
        raise
    except Exception as exc:  # numpy/Pillow surprises stay inside the perception boundary
        raise PerceptionError(f"perceptual hash failed: {exc}") from exc
    width = (_PHASH_BITS + 3) // 4
    return [format(int("".join("1" if b else "0" for b in row), 2), f"0{width}x") for row in bits]


def perceptual_hash(screenshot: Screenshot) -> str:
    """The whole difference hash of ``screenshot`` as one hex string, rows joined by ``-``.

    Deterministic, and tolerant of antialiasing and of a few pixels of scroll.

    It encodes STRUCTURE, not brightness or colour: two frames whose edges fall in the
    same places hash alike even when their palettes differ, so a recoloured banner reads
    as the same screen. That is usually what you want from a state identity; when it is
    not, the structural and URL signals are what separate the two.

    Raises:
        PerceptionError: if the screenshot cannot be decoded.
    """
    return "-".join(_phash_rows(screenshot))


def _layout_buckets(elements: Sequence[Element], width: int, height: int) -> list[list[str]]:
    """The quantized, text-free element tokens of each screen quadrant.

    An empty bucket means "no elements in this quadrant", which is the absence of
    evidence rather than a piece of it - see :meth:`StateFingerprinter.fingerprint`.
    """
    span_x = max(float(width), 1.0)
    span_y = max(float(height), 1.0)
    cell_w = span_x / _LAYOUT_GRID
    cell_h = span_y / _LAYOUT_GRID
    buckets: list[list[str]] = [[] for _ in range(_LAYOUT_BANDS * _LAYOUT_BANDS)]
    for element in elements:
        box = element.box
        gx = min(max(int((box.x + box.w / 2) // cell_w), 0), _LAYOUT_GRID - 1)
        gy = min(max(int((box.y + box.h / 2) // cell_h), 0), _LAYOUT_GRID - 1)
        w_bucket = round(max(box.w, 0) / _SIZE_BUCKET)
        h_bucket = round(max(box.h, 0) / _SIZE_BUCKET)
        band = _LAYOUT_GRID // _LAYOUT_BANDS
        quadrant = min(gy // band, _LAYOUT_BANDS - 1) * _LAYOUT_BANDS + min(
            gx // band, _LAYOUT_BANDS - 1
        )
        buckets[quadrant].append(f"{element.kind.value}@{gx},{gy}+{w_bucket}x{h_bucket}")
    return buckets


def structural_hash(elements: Sequence[Element], width: int, height: int) -> str:
    """Digest of the element layout: sorted kinds with quantized positions, text ignored.

    Returns the digest of the empty layout for an empty ``elements``; never raises.
    """
    return "-".join(_digest(*sorted(bucket)) for bucket in _layout_buckets(elements, width, height))


def normalize_url(url: str | None) -> str | None:
    """``url`` reduced to a state pattern, or ``None`` when there is nothing usable.

    Query and fragment are dropped, scheme and host are lowercased, a default port is
    removed, a trailing slash is dropped, and volatile path segments (digits, long hex
    strings, UUIDs) become ``{id}``. Never raises: an unparseable URL returns ``None``.
    """
    if not url or not url.strip():
        return None
    try:
        split = urlsplit(url.strip())
        scheme = split.scheme.lower()
        # ``hostname`` and ``port`` parse lazily, so both can raise here, not at urlsplit.
        host = (split.hostname or "").lower()
        port = split.port if split.netloc else None
    except ValueError:
        return None
    if port is not None and _DEFAULT_PORTS.get(scheme) == str(port):
        port = None
    authority = f"{host}:{port}" if port is not None else host
    segments = [
        _ID_PLACEHOLDER if _VOLATILE_SEGMENT.match(segment) else segment
        for segment in split.path.split("/")
    ]
    path = "/".join(segments)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    pattern = f"{scheme}://{authority}{path}" if scheme or authority else path
    return pattern or None


class StateFingerprinter:
    """A :class:`~skillweaver.contracts.Fingerprinter` combining pixels, layout and URL.

    Stateless and deterministic: the same screenshot, elements and URL always produce an
    equal fingerprint, and nothing is cached between calls.
    """

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        """Identify the screen. See the module docstring for the three signals.

        Raises:
            PerceptionError: if the screenshot cannot be decoded or hashed.
        """
        pattern = normalize_url(url)
        buckets = _layout_buckets(elements, screenshot.width, screenshot.height)
        quadrants = [_digest(*sorted(bucket)) for bucket in buckets]
        rows = _phash_rows(screenshot)

        parts: dict[str, str] = {}
        if pattern is not None:
            parts["url"] = pattern
        for i, bucket in enumerate(buckets):
            if bucket:
                parts[f"layout.q{i}"] = quadrants[i]
        for i, row in enumerate(rows):
            if row.strip("0"):
                parts[f"phash.r{i}"] = row

        # ``value`` hashes the WHOLE signal, including the chunks left out of ``parts``,
        # so two frames are equal only when everything about them matches.
        value = _digest(pattern or "", *quadrants, *rows)
        return Fingerprint(value, parts)
