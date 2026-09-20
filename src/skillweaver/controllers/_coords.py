"""Coordinate arithmetic for the browser controller: the ONE place a logical pixel
becomes a physical one, or back. Pure - no Playwright, no I/O, no state.

Little of it is needed on the ACTION path, because Playwright's ``page.mouse`` already
speaks CSS pixels: :func:`within_viewport` is all the controller needs. It matters on the
CAPTURE path, where a PNG is at physical resolution, so :func:`scale_for` derives
``Screenshot.scale`` from the bytes rather than from what the browser was asked for. A
click that lands at exactly twice the intended coordinates is a missing call to this file.
"""

from __future__ import annotations

import math
import struct

from skillweaver.contracts import Box, Point

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def check_scale(scale: float) -> float:
    """``scale`` if it is a usable physical-per-logical ratio.

    Raises:
        ValueError: not finite or not positive. A zero or negative scale would collapse
            every coordinate to the origin, so it is refused rather than clamped.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return scale


def png_size(png: bytes) -> tuple[int, int]:
    """``(width, height)`` in PHYSICAL pixels from a PNG's IHDR header, without decoding.

    Raises:
        ValueError: not a PNG, or the header is truncated.
    """
    if len(png) < 24 or not png.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG image")
    if png[12:16] != b"IHDR":
        raise ValueError("PNG is missing its IHDR header chunk")
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def scale_for(physical: int, logical: int) -> float:
    """The physical-per-logical ratio, measured from the PNG actually captured so that
    ``Screenshot.scale`` and the bytes cannot disagree.

    Raises:
        ValueError: either side is not positive.
    """
    if physical <= 0 or logical <= 0:
        raise ValueError(f"sizes must be positive, got physical={physical}, logical={logical}")
    return physical / logical


def physical_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """The PNG size in physical pixels for a ``width`` x ``height`` logical viewport."""
    check_scale(scale)
    return round(width * scale), round(height * scale)


def point_to_physical(point: Point, scale: float) -> Point:
    """Convert a logical :class:`Point` to physical image pixels."""
    check_scale(scale)
    return Point(round(point.x * scale), round(point.y * scale))


def point_to_logical(x: float, y: float, scale: float) -> Point:
    """A raw physical image coordinate as a logical :class:`Point`."""
    check_scale(scale)
    return Point(round(x / scale), round(y / scale))


def box_to_physical(box: Box, scale: float) -> Box:
    """A logical :class:`Box` in physical image pixels.

    Both edges convert before the size is taken, so the result spans the same pixels the
    corners do.
    """
    check_scale(scale)
    x0, y0 = round(box.x * scale), round(box.y * scale)
    x1, y1 = round((box.x + box.w) * scale), round((box.y + box.h) * scale)
    return Box(x0, y0, x1 - x0, y1 - y0)


def box_to_logical(x: float, y: float, w: float, h: float, scale: float) -> Box:
    """A raw physical image rectangle as a logical :class:`Box`.

    Edges convert independently and the size derives from them, so adjacent boxes stay
    adjacent and no box gains or loses a pixel to rounding twice.
    """
    check_scale(scale)
    x0, y0 = round(x / scale), round(y / scale)
    x1, y1 = round((x + w) / scale), round((y + h) / scale)
    return Box(x0, y0, x1 - x0, y1 - y0)


def within_viewport(viewport: Box, point: Point) -> bool:
    """Whether a logical action coordinate falls inside ``viewport``.

    Action coordinates are viewport-relative, so only the SIZE is consulted, never the
    offset. Right and bottom edges are exclusive, matching ``Box.contains``.
    """
    return 0 <= point.x < viewport.w and 0 <= point.y < viewport.h


def clip_to_viewport(box: Box, viewport: Box) -> Box:
    """Trim ``box`` to the part of it on screen; zero area when none of it is.

    Ground truth uses this so an element hanging off the edge is labelled with the
    rectangle a detector could plausibly see.
    """
    x0 = max(box.x, 0)
    y0 = max(box.y, 0)
    x1 = min(box.x + box.w, viewport.w)
    y1 = min(box.y + box.h, viewport.h)
    return Box(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))
