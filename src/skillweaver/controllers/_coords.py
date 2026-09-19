"""Coordinate arithmetic for the browser controller.

This module is the ONE place where a logical pixel becomes a physical one, or the
other way round. It is pure: no Playwright, no I/O, no state, so every rule here
can be unit-tested on its own.

Why so little of it is needed on the action path
------------------------------------------------
Playwright's ``page.mouse`` speaks CSS pixels, which are exactly the LOGICAL pixels
of :mod:`skillweaver.contracts`. A click target therefore travels from the agent to
the browser unconverted, and :func:`within_viewport` is all the controller needs
before handing a point over.

The conversion matters on the *capture* path instead. A screenshot's PNG is at the
display's PHYSICAL resolution - twice the logical size on a Retina-style page - so
:func:`scale_for` derives ``Screenshot.scale`` from the PNG the capture actually
produced rather than from what the browser was asked for. Anything that then works
on those native pixels (a YOLO detector, OCR) converts back with
:func:`point_to_logical` or :func:`box_to_logical` before building a contract value.

A click that lands at exactly twice the intended coordinates is a missing call to
something in this file.
"""

from __future__ import annotations

import math
import struct

from skillweaver.contracts import Box, Point

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def check_scale(scale: float) -> float:
    """Return ``scale`` if it is a usable physical-per-logical ratio.

    Raises:
        ValueError: if ``scale`` is not finite or is not greater than zero. A zero
            or negative scale would silently collapse every coordinate to the
            origin, so it is refused rather than clamped.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return scale


def png_size(png: bytes) -> tuple[int, int]:
    """Read ``(width, height)`` in PHYSICAL pixels out of a PNG's IHDR header.

    Cheaper than decoding the image, and used by the controller to check that the
    frame it reports matches the bytes it captured.

    Raises:
        ValueError: if ``png`` is not a PNG or its header is truncated.
    """
    if len(png) < 24 or not png.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG image")
    if png[12:16] != b"IHDR":
        raise ValueError("PNG is missing its IHDR header chunk")
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def scale_for(physical: int, logical: int) -> float:
    """The physical-per-logical ratio that turns ``logical`` into ``physical``.

    This is how ``Screenshot.scale`` is derived: measured from the PNG that was
    actually captured, so the reported scale and the bytes can never disagree.

    Raises:
        ValueError: if either side is not positive.
    """
    if physical <= 0 or logical <= 0:
        raise ValueError(f"sizes must be positive, got physical={physical}, logical={logical}")
    return physical / logical


def physical_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """The PNG size in physical pixels for a ``width`` x ``height`` logical viewport.

    Matches the ``Screenshot`` docstring: ``round(width * scale)`` by
    ``round(height * scale)``.
    """
    check_scale(scale)
    return round(width * scale), round(height * scale)


def point_to_physical(point: Point, scale: float) -> Point:
    """Convert a logical :class:`Point` to physical image pixels."""
    check_scale(scale)
    return Point(round(point.x * scale), round(point.y * scale))


def point_to_logical(x: float, y: float, scale: float) -> Point:
    """Convert a raw physical image coordinate to a logical :class:`Point`.

    Use this on anything read out of ``Screenshot.to_array(logical=False)``.
    """
    check_scale(scale)
    return Point(round(x / scale), round(y / scale))


def box_to_physical(box: Box, scale: float) -> Box:
    """Convert a logical :class:`Box` to physical image pixels.

    Both edges are converted before the size is taken, so the result always spans
    the same pixels the corners do.
    """
    check_scale(scale)
    x0, y0 = round(box.x * scale), round(box.y * scale)
    x1, y1 = round((box.x + box.w) * scale), round((box.y + box.h) * scale)
    return Box(x0, y0, x1 - x0, y1 - y0)


def box_to_logical(x: float, y: float, w: float, h: float, scale: float) -> Box:
    """Convert a raw physical image rectangle to a logical :class:`Box`.

    Edges are converted independently and the size derived from them, so adjacent
    boxes stay adjacent and a box never gains or loses a pixel to rounding twice.
    """
    check_scale(scale)
    x0, y0 = round(x / scale), round(y / scale)
    x1, y1 = round((x + w) / scale), round((y + h) / scale)
    return Box(x0, y0, x1 - x0, y1 - y0)


def within_viewport(viewport: Box, point: Point) -> bool:
    """Whether a logical action coordinate falls inside ``viewport``.

    Action coordinates are relative to the viewport's top-left corner (see
    ``Controller.viewport``), so only ``viewport``'s size is consulted, never its
    offset. The right and bottom edges are exclusive, matching ``Box.contains``.
    """
    return 0 <= point.x < viewport.w and 0 <= point.y < viewport.h


def clip_to_viewport(box: Box, viewport: Box) -> Box:
    """Trim ``box`` to the part of it that is actually on screen.

    Returns a zero-area box when nothing of it is visible. Ground truth uses this
    so an element hanging off the edge is labelled with the rectangle a detector
    could plausibly see, not with one reaching outside the frame.
    """
    x0 = max(box.x, 0)
    y0 = max(box.y, 0)
    x1 = min(box.x + box.w, viewport.w)
    y1 = min(box.y + box.h, viewport.h)
    return Box(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))
