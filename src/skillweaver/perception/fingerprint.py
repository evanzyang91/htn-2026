"""UI state fingerprinting: the answer to "which screen am I on?".

:class:`StateFingerprinter` turns a frame into a ``Fingerprint`` whose ``value`` is a
graph node id and whose ``parts`` explain why two screens matched.

A part is never named by WHERE it is. A notice arriving at the top pushes the page down,
and under the previous design - a difference hash on a viewport-pegged grid, one part per
ROW - a 200px notice scored a page against itself at 0.040, the same as two unrelated
sites; since every graph node is keyed on a fingerprint, the per-site graph then grew a
fresh node per visit and never accumulated. So ``parts`` are:

* ``band.<pattern>#<k>`` - horizontal pixel bands named by their bit PATTERN, with
  ``#<k>`` carrying occupancy in powers of two (see :data:`_BAND_PITCH`). A flat band
  emits nothing: ``similarity`` divides by the UNION of part names, so two screens blank
  in the same places collect no credit for it.
* ``url`` - normalized, and omitted entirely when absent rather than matching falsely.

The element index is hashed into ``value`` but contributes NO part, measured rather than
assumed: over the 88 live captures behind :data:`SAME_STATE_THRESHOLD` the previous
design's four quadrant parts moved the same-state floor from 0.305 to 0.301 and the
different-state ceiling from 0.213 to 0.225, because detection and OCR do not reproduce
across loads. A text sketch is worse: floor 0.151, ceiling unmoved.

Bands span the FULL width on purpose. Tiling into columns lifts a page with a rotating ad
from 0.335 to 0.692 - and two different Wikipedia articles from 0.034 to 0.505, because
half-width bands carry too few bits to tell one body of text from another. The second
costs more than the first is worth.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from skillweaver.contracts import Element, Fingerprint, Screenshot
from skillweaver.errors import PerceptionError

if TYPE_CHECKING:
    # At RUN time numpy is imported by the three functions that compute with it. This
    # module is imported by every command, ``skills ls`` and ``--help`` included, and none
    # of them fingerprints anything: 35ms of their start, and 390ms on the first command
    # after a ``uv sync`` (measured 2026-09-20). A run pays it once, at its first fingerprint -
    # or not at all, when the controller had it imported while it waited on the browser.
    import numpy as np

SAME_STATE_THRESHOLD = 0.26
"""Recommended cut for "these two frames are the same UI state".

Calibrated 2026-09-19 against TWO corpora at once, because either alone misleads: the
contrived pairs reproduce exactly and live pages never reproduce at all. Live: 88
captures through the shipped pipeline, Chromium headless at 1280x800, eight pages of four
sites, each reloaded, re-browsered, pushed down 60/120/200px, resized and scrolled.

Taking both together, same-state bottoms out at **0.305** (MDN, whose ad and promo strip
re-roll every load) and different-state tops out at **0.213** (two contrived invoice
lists sharing URL, chrome and layout), so 0.26 sits in a 0.09-wide gap. Narrow, and worth
saying what it replaces: the previous design gave a floor of 0.040 and a ceiling of
0.500, classes that OVERLAPPED, so no threshold existed at all.

Two things are still NOT the same state, correctly. A full-screen takeover: Wikipedia's
fundraising appeal displaces 555 of 800 pixels and scores 0.124, and the
information-theoretic ceiling for that pair is ~0.18, so the answer is to dismiss the
banner, not to loosen identity. And scrolling on purpose - 40px of dense body text scores
0.834 where it used to score 0.160, but that is not a promise.

Re-derive this if :class:`StateFingerprinter` changes what it puts in ``parts``: a
threshold is only meaningful against the shape of the signal it judges.
"""

_BAND_PITCH = 4.0
"""Logical pixels between consecutive band centres; bands are :data:`_BAND_HEIGHT` tall,
so they overlap heavily and a page shifted by N pixels re-emits its patterns as long as
some band lands where an old one did. A pitch of 8px drops the pushed-down class from
0.780 to 0.168; 2px buys nothing and doubles the part count.

A pattern seen ``c`` times emits ``#0``..``#floor(log2(c))``: powers of two, so a region
growing 120px to 145px costs at most one part. Dropping occupancy entirely lifts two
contrived invoice lists that share a layout from 0.213 to 0.714."""

_BAND_HEIGHT = 24.0
"""Half-width of the triangular window averaged into one band, in logical pixels: how far
a local change spreads. The previous design needed 120px because its grid-pegged bands
had to reproduce a perturbed value EXACTLY; content addressing removes that, and 24px is
most of why a notice now scores 0.780 instead of 0.040. Averaging rather than sampling is
what absorbs antialiasing, which is why it is not smaller."""

_BAND_BITS = 32
"""Bits per band: the frame is reduced to ``_BAND_BITS + 1`` columns and neighbours
compared. The discrimination knob, set by the hardest DIFFERENT pair: at 16 bits (80px
per column) two contrived invoice lists sharing a layout score 0.615, at 32 (40px, about
three characters) 0.213, and at 48 a 2x Retina capture of one unchanged screen falls from
0.770 to 0.520 while that pair barely moves."""

_BAND_TOLERANCE = 2.0
"""Deadzone in luma units (0..255) for the neighbour comparison, so flat regions resolve
to a stable ``0`` instead of a coin flip on rounding noise. Quantizing the column values
instead would only move the coin flip to the quantization boundaries."""

_LAYOUT_GRID = 4
"""Element centres are binned to a ``_LAYOUT_GRID`` square grid. Deliberately coarse -
200x150 logical pixels per cell at 800x600 - because the structural signal answers "what
kinds of thing, roughly where" and the pixel signal answers the rest. Widening cells from
50px to 200px lifted the worst same-state pair from 0.667 to 0.750 and moved no
different-state pair at all."""

_LAYOUT_BANDS = 2
"""Quadrant split per axis, so ``_LAYOUT_BANDS ** 2`` ``layout.qN`` parts."""

_SIZE_BUCKET = 16
"""Element width/height are rounded to this many logical pixels before hashing."""

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
    pixels of scroll.
    """
    import numpy as np

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

    Dividing by the same convolution of ones renormalizes the ends, so the top and bottom
    bands average over what is there rather than over an implied field of black.
    """
    import numpy as np

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
    """One hex bit pattern per band, top first. POSITIONAL; it is
    :meth:`StateFingerprinter.fingerprint` that throws the positions away.

    A run of equal entries means a region of the page looks the same all the way down.

    Raises:
        PerceptionError: the screenshot cannot be decoded or hashed.
    """
    import numpy as np

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
    """The whole band hash as one hex string, bands joined by ``-``.

    The POSITIONAL form, so it does NOT survive a page being pushed down; it is the
    whole-signal identity behind ``Fingerprint.value``, and answers "are these two frames
    pixel-identical". It encodes STRUCTURE, not colour, so a recoloured banner reads as
    the same screen.

    Raises:
        PerceptionError: the screenshot cannot be decoded.
    """
    return "-".join(_bands(screenshot))


def band_parts(screenshot: Screenshot) -> dict[str, str]:
    """The content-addressed band parts: ``{"band.<pattern>#<k>": pattern}``.

    A flat band contributes nothing. The part's VALUE is the pattern itself rather than a
    hash of it, so two parts agreeing by name are checked to agree in substance too.

    Raises:
        PerceptionError: the screenshot cannot be decoded or hashed.
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

    An empty bucket is the absence of evidence, not a piece of it.
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
    """Digest of the element layout: sorted kinds, quantized positions, text ignored."""
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
    """A ``Fingerprinter`` over content-addressed pixel bands and the normalized URL.

    Stateless, deterministic and uncached.
    """

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        """Identify the screen.

        ``elements`` is hashed into ``value`` but contributes no part, so a screen whose
        detector or OCR read it differently is still the same screen by ``similarity``
        while remaining a different frame by ``==``.

        Raises:
            PerceptionError: the screenshot cannot be decoded or hashed.
        """
        pattern = normalize_url(url)
        bands = _bands(screenshot)

        parts: dict[str, str] = _parts_from_bands(bands)
        if pattern is not None:
            parts["url"] = pattern

        # ``value`` hashes the WHOLE signal, silent bands and band ORDER included, so two
        # frames are equal only when everything about them matches.
        layout = structural_hash(elements, screenshot.width, screenshot.height)
        value = _digest(pattern or "", layout, *bands)
        return Fingerprint(value, parts)
