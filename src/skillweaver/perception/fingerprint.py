"""UI state fingerprinting: the answer to "which screen am I on?".

:class:`StateFingerprinter` turns a frame into a :class:`~skillweaver.contracts.Fingerprint`
whose ``value`` is a graph node id and whose ``parts`` explain *why* two screens matched.

The problem this shape solves
-----------------------------

A real page moves. A fundraising notice, a cookie bar, a logged-out prompt or an A/B
strip arrives at the top and pushes everything below it DOWN, and a screen the agent
knows perfectly well stops being recognisable. Measured live on 2026-09-19 with the
previous design - a difference hash on a grid pegged to the viewport, one part per grid
ROW - a 200px notice scored a page against itself at **0.040**, the same score as two
unrelated websites. Since every graph node is keyed on a fingerprint, that does not
merely fail one run: the per-site graph grows a fresh node on every visit and never
accumulates.

Anchoring a part to a fixed grid row is what caused it. Shift the page by 200px and
every row's content moves to a different row, so every part name still exists and every
one of them disagrees. The fix is to stop naming a part by WHERE it is.

Two signals
-----------

**Content-addressed pixel bands** (``band.<pattern>#<k>``) - the frame is reduced to
horizontal bands and each band becomes a short bit pattern; the part is then named by
that PATTERN, never by its position. Two screens therefore agree on a band the moment
they both contain it, wherever it sits, which is exactly the invariance a page that
scrolled or was pushed down needs. ``#<k>`` carries how much of the screen the pattern
covers, in powers of two (see :data:`_BAND_PITCH`), so occupancy still counts without a
few pixels of it mattering.

**Normalized URL** (``url``) - scheme and host lowercased, default port and query and
fragment dropped, volatile path segments (digits, hex ids, UUIDs) replaced by ``{id}``.
Omitted entirely when there is no URL, rather than contributing a fake match.

What is NOT in ``parts``, and why
---------------------------------

The element index is hashed into ``value`` (see :func:`structural_hash`) but deliberately
contributes NO part. It was measured, not assumed: over the 88 live captures behind
:data:`SAME_STATE_THRESHOLD`, adding the four quadrant parts of the previous design moved
the same-state floor from 0.305 to 0.301 and the different-state ceiling from 0.213 to
0.225 - it costs at both ends. Detection and OCR do not reproduce across loads (47
elements on one capture of a page, 31 on the next when a notice pushed content off the
bottom), so an element signal reports that noise as disagreement. A sketch of the
on-screen TEXT was measured too and is worse still: it drops the same-state floor to
0.151 while leaving the different-state ceiling where it was.

Bands are full width on purpose
-------------------------------

A band spans the whole viewport, so a narrow thing changing - a rotating advertisement in
a right-hand rail - spoils every band it touches. Tiling the frame into columns fixes
that and was measured: it lifts a page with a rotating ad from 0.335 to 0.692. It also
lifts two DIFFERENT Wikipedia articles from 0.034 to **0.505**, because half-width bands
carry too few bits to tell one body of text from another. That is the trade refused here:
the conjunction of all :data:`_BAND_BITS` bits across the full width is both what makes a
band fragile and what makes two different screens score near zero, and the second is
worth more than the first.

Silence is not agreement
------------------------

A band with no horizontal structure in it - flat background - emits nothing. Because
``similarity`` divides by the UNION of both fingerprints' part names, two sparse screens
that are blank in the same places collect no credit for it.

``value``, unlike ``parts``, hashes the WHOLE signal - the URL, the element layout and
every band including the silent ones - so two frames are equal only when everything about
them matches.

Threshold
---------

:data:`SAME_STATE_THRESHOLD` is the recommended "same state" cut, and
:mod:`skillweaver.graph.route` applies it to routing. See that constant for the two
corpora it was calibrated on.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from urllib.parse import urlsplit

import numpy as np

from skillweaver.contracts import Element, Fingerprint, Screenshot
from skillweaver.errors import PerceptionError

SAME_STATE_THRESHOLD = 0.26
"""Recommended cut for "these two frames are the same UI state".

Calibrated on 2026-09-19 against TWO corpora at once, because either one alone is
misleading: the committed pairs are contrived and reproduce exactly, and live pages
never reproduce at all.

**Live pages** - 88 captures through the shipped pipeline (``YoloDetector`` +
``RapidOcrReader`` + this fingerprinter), Chromium headless at 1280x800, across eight
pages of four sites (Wikipedia Main Page, two articles, two search-result pages, two
docs.python.org pages, one MDN page). Each page was captured in two fresh browsers, in
one browser twice, with a notice of 60, 120 and 200px injected at the top, at a 900px
viewport, and scrolled 8 and 40px:

=========================================  ===  =====  ======  =====
class                                        n    min  median    max
=========================================  ===  =====  ======  =====
SAME page, reload or fresh browser          76  0.335   1.000  1.000
SAME page, notice pushed it down 60-200px  117  0.342   0.780  0.871
SAME page, viewport 800 vs 900 high         34  0.305   0.870  0.894
SAME page, scrolled 8 or 40px               32  0.834   0.903  0.961
-----------------------------------------  ---  -----  ------  -----
DIFFERENT page, same site and template      42  0.019   0.042  0.189
DIFFERENT page, unrelated site              38  0.000   0.000  0.042
=========================================  ===  =====  ======  =====

Every page but one reloads at exactly 1.000. The 0.335 floor is MDN, whose right-hand
advertisement and top promotion strip are re-rolled on every load - the "an ad rotates"
case, and the hardest thing on this list that is still the same screen.

**The committed pairs** in ``tests/fixtures/shots/pairs``, which carry the contrived
worst cases a live corpus will not produce:

===============================  =====  =========================================
pair                             score  what differs
===============================  =====  =========================================
same_screen_twice                1.000  nothing
same_screen_clock_tick           1.000  a clock digit, and a ``?t=`` query
dense_text_scrolled_slightly     0.966  solid body text scrolled 8px
same_screen_caret_blink          0.965  the text caret's blink phase
same_screen_no_url               0.964  no URL at all, plus a caret blink
same_screen_retina               0.770  captured at 2x instead of 1x
same_screen_scrolled             0.674  scrolled 6px, pulling a new row into view
search_results_pushed_down       0.659  a 120px notice moved every hit down
-------------------------------  -----  -----------------------------------------
same_layout_different_content    0.213  every row's text; layout and URL identical
modal_open_vs_closed             0.198  a dialog and its scrim
dense_text_different_article     0.101  one template, entirely different prose
article_vs_search_results        0.054  one site, prose against a column of hits
different_screens                0.011  another page entirely
===============================  =====  =========================================

Taking the two together, same-state bottoms out at **0.305** and different-state tops out
at **0.213**, so 0.26 sits in the middle of that 0.09-wide gap. The last two rows above
are the same claim on one pair of pages: the results page moved scores 0.659, and the
article against that same results page scores 0.054. The binding pair is
cross-corpus - a live page whose advertisement rotated, against two contrived invoice
lists that share a URL, a chrome and a layout - which is the conservative way to combine
them.

The gap is narrow, and it is worth saying what it replaces: under the previous design the
same two corpora gave a same-state floor of **0.040** and a different-state ceiling of
**0.500**. The classes OVERLAPPED, so no threshold existed at all, and a live page pushed
down by a notice was indistinguishable from an unrelated website. A 0.09-wide gap is a
poor thing to have to defend; a negative one cannot be defended at all.

What is still NOT the same state
--------------------------------

**A notice that takes over the screen.** The live corpus caught a real Wikimedia
fundraising appeal arriving mid-session. It is not a strip: it displaces the article by
**555 of 800 pixels**, leaving 31% of the recorded screen visible. It scores 0.124, below
this cut, and refusing it is correct - 69% of what the skill was written against is gone.
The information-theoretic ceiling for that pair is about 0.18, so no fingerprint recovers
it; the answer is to DISMISS the banner and look again, not to loosen identity.

**Scrolling on purpose.** A caller that scrolls should track the state it scrolled from
rather than expect this cut to tie the two captures together. It holds much further than
it used to - 40px of dense body text now scores 0.834 where it used to score 0.160 - but
it is not a promise.

Re-check this value if :class:`StateFingerprinter` changes what it puts in ``parts``. A
threshold has to be read against the shape of the signal, never in the abstract.
"""

# --- pixel bands ----------------------------------------------------------------------

_BAND_PITCH = 4.0
"""Logical pixels between the centres of consecutive bands.

Bands overlap heavily - each is :data:`_BAND_HEIGHT` tall and they start every 4px - and
that redundancy is the point. A part is named by its band's PATTERN, so a page shifted
down by N pixels re-emits the same patterns as long as some band lands where the old one
did; the pitch is how finely that can be met. Measured on the live corpus, a pitch of 8px
drops the notice class from 0.780 to 0.168 because a 200px shift no longer lands on a
band, while 2px buys nothing the 4px pitch does not already have and doubles the part
count.

The redundancy also means one pattern usually repeats over a run of bands, which is what
``#<k>`` counts: a pattern seen ``c`` times emits parts ``#0`` up to ``#floor(log2(c))``.
Powers of two rather than a plain count, so a region growing from 120px to 145px costs at
most one part instead of six. Dropping occupancy altogether was measured and is wrong: it
lifts two contrived invoice lists that share a layout from 0.213 to 0.714, because their
band patterns are the same and only the amount of screen each covers differs.
"""

_BAND_HEIGHT = 24.0
"""Half-width, in logical pixels, of the triangular window averaged into one band.

Sets how much of the page a single band reports on, and therefore how far a local change
spreads: a notice arriving contaminates the bands within this distance of it, and
everything further away still matches. The previous design needed a window of 120px to
survive scrolling, because its bands were pegged to a grid and had to reproduce a
perturbed value EXACTLY; content addressing removes that need, and shrinking the window to
24px is most of why a notice now scores 0.780 instead of 0.040.

Averaging over a window rather than sampling is still what absorbs antialiasing and
sub-pixel rendering differences, which is why it is not smaller.
"""

_BAND_BITS = 32
"""Bits per band: the frame is reduced to ``_BAND_BITS + 1`` columns and neighbours
compared.

This is the discrimination knob, and it is set by the hardest DIFFERENT pair rather than
by any same pair. At 16 bits - 80 logical pixels per column - two contrived invoice lists
that share a layout score 0.615, because 80px of body text averages to the same value
whatever it says. At 32 bits each column is 40px, about three characters, and the same
pair scores 0.213. Past that it starts costing same-state pairs for nothing: at 48 bits a
2x Retina capture of one unchanged screen falls from 0.770 to 0.520 while the invoice
pair barely moves.
"""

_BAND_TOLERANCE = 2.0
"""Deadzone, in luma units (0..255), for the neighbour comparison.

A column is only called brighter than its neighbour when it beats it by this much, so the
flat regions of a UI - page background, a card's fill - resolve to a stable ``0`` instead
of a coin flip on rounding noise. Quantizing the column values instead would only move the
coin flip to the quantization boundaries.
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


def _smooth_rows(fine: np.ndarray, half: float) -> np.ndarray:
    """``fine`` smoothed down its first axis by a triangular window of half-width ``half``.

    Each output row is the weighted mean of the fine rows within ``half`` of it, the
    weight falling linearly to zero. Dividing by the same convolution of ones renormalizes
    the ends, so the top and bottom bands average over what is actually there rather than
    over an implied field of black.
    """
    reach = max(int(np.ceil(half)), 1)
    kernel = np.clip(1.0 - np.abs(np.arange(-reach, reach + 1)) / half, 0.0, None)
    # Zero-padded ``valid`` rather than ``same``: numpy returns max(len) for ``same``,
    # which is the wrong length whenever the frame is shorter than the window.
    pad = np.zeros((reach, fine.shape[1]), dtype=fine.dtype)
    padded = np.vstack((pad, fine, pad))
    mask = np.concatenate((np.zeros(reach), np.ones(fine.shape[0]), np.zeros(reach)))
    norm = np.convolve(mask, kernel, mode="valid")
    out = np.empty_like(fine)
    for column in range(fine.shape[1]):
        out[:, column] = np.convolve(padded[:, column], kernel, mode="valid") / norm
    return out


def _bands(screenshot: Screenshot) -> list[str]:
    """One hex bit pattern per band of ``screenshot``, top band first.

    Bands are :data:`_BAND_HEIGHT` tall and start every :data:`_BAND_PITCH` logical
    pixels, so consecutive entries overlap and a run of equal entries means a region of
    the page looks the same all the way down. The list is POSITIONAL; it is
    :meth:`StateFingerprinter.fingerprint` that throws the positions away.

    Raises:
        PerceptionError: if the screenshot cannot be decoded or hashed.
    """
    try:
        image = screenshot.to_array()
        if image.size == 0:
            return []
        gray = image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        rows = max(int(round(screenshot.height / _BAND_PITCH)), 1)
        fine = _area_downsample(gray, rows, _BAND_BITS + 1)
        cells = _smooth_rows(fine, max(_BAND_HEIGHT / _BAND_PITCH, 1.0))
        bits = (cells[:, 1:] - cells[:, :-1]) > _BAND_TOLERANCE
    except PerceptionError:
        raise
    except Exception as exc:  # numpy/Pillow surprises stay inside the perception boundary
        raise PerceptionError(f"band hash failed: {exc}") from exc
    width = (_BAND_BITS + 3) // 4
    return [format(int("".join("1" if b else "0" for b in row), 2), f"0{width}x") for row in bits]


def perceptual_hash(screenshot: Screenshot) -> str:
    """The whole band hash of ``screenshot`` as one hex string, bands joined by ``-``.

    Deterministic, and tolerant of antialiasing and of sub-pixel rendering differences.
    This is the POSITIONAL form, and so it is NOT what makes a fingerprint survive a page
    being pushed down - it changes completely when the page shifts. It is the whole-signal
    identity that goes into ``Fingerprint.value``, and it is useful for asking whether two
    frames are pixel-identical.

    It encodes STRUCTURE, not brightness or colour: two frames whose edges fall in the
    same places hash alike even when their palettes differ, so a recoloured banner reads
    as the same screen. That is usually what you want from a state identity; when it is
    not, the structural and URL signals are what separate the two.

    Raises:
        PerceptionError: if the screenshot cannot be decoded.
    """
    return "-".join(_bands(screenshot))


def band_parts(screenshot: Screenshot) -> dict[str, str]:
    """The content-addressed band parts of ``screenshot``: ``{"band.<pattern>#<k>": pattern}``.

    A band with no horizontal structure - flat background - contributes nothing, because
    two screens that are blank in the same places have agreed about nothing. A pattern
    covering ``c`` bands emits ``#0`` up to ``#floor(log2(c))``, so how much of the screen
    it covers still counts while a few pixels of it do not; see :data:`_BAND_PITCH`.

    The part's VALUE is the pattern itself rather than a hash of it, so two parts that
    agree by name are checked to agree in substance as well.

    Raises:
        PerceptionError: if the screenshot cannot be decoded or hashed.
    """
    return _parts_from_bands(_bands(screenshot))


def _parts_from_bands(bands: Sequence[str]) -> dict[str, str]:
    """:func:`band_parts`, over bands already computed. Never raises."""
    counts: dict[str, int] = {}
    for pattern in bands:
        if pattern.strip("0"):
            counts[pattern] = counts.get(pattern, 0) + 1
    return {
        f"band.{pattern}#{k}": pattern
        for pattern, count in counts.items()
        for k in range(count.bit_length())
    }


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
    """A :class:`~skillweaver.contracts.Fingerprinter` over content-addressed pixel bands
    and the normalized URL.

    Stateless and deterministic: the same screenshot, elements and URL always produce an
    equal fingerprint, and nothing is cached between calls.
    """

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        """Identify the screen. See the module docstring for what goes into each field.

        ``elements`` is hashed into ``value`` but contributes no part, so a screen whose
        detector or OCR read it differently is still the same screen by ``similarity``
        while remaining a different frame by ``==``.

        Raises:
            PerceptionError: if the screenshot cannot be decoded or hashed.
        """
        pattern = normalize_url(url)
        bands = _bands(screenshot)

        parts: dict[str, str] = _parts_from_bands(bands)
        if pattern is not None:
            parts["url"] = pattern

        # ``value`` hashes the WHOLE signal - the layout and the silent bands included,
        # and the bands in their original ORDER - so two frames are equal only when
        # everything about them matches.
        layout = structural_hash(elements, screenshot.width, screenshot.height)
        value = _digest(pattern or "", layout, *bands)
        return Fingerprint(value, parts)
