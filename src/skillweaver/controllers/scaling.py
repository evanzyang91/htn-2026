"""Every coordinate conversion a desktop controller needs, in one pure place.

Three coordinate spaces, of which only the first two leave this package:

``local logical``
    Logical pixels with the origin at the top-left of ``Controller.viewport()``. Every
    ``Point`` and ``Box`` in the codebase is in this space.
``global logical``
    Logical pixels with the origin at the primary display. What the OS pointer APIs
    want; a display left of the primary has a negative ``x``. :func:`to_global` /
    :func:`to_local` convert.
``physical``
    Raw bitmap pixels. Deliberately NOT ``Point`` or ``Box`` - plain ints and tuples -
    so a physical coordinate cannot be mistaken for a logical one.

Rounding is half-up (``floor(v + 0.5)``) everywhere, never bankers' rounding, so a test
can state the expected number outright. Boxes converting DOWN from physical are the
exception and grow outwards (floor the near edge, ceil the far one), so a detection never
loses the pixels at its border.
"""

from __future__ import annotations

import math

from skillweaver.contracts import Box, Point

__all__ = [
    "box_from_physical",
    "box_to_physical",
    "clamp_box",
    "clamp_point",
    "logical_to_physical",
    "physical_to_logical",
    "point_from_physical",
    "point_to_physical",
    "scale_for",
    "to_global",
    "to_local",
]


def _round_half_up(value: float) -> int:
    """Ties up, positives and negatives: ``2.5`` -> ``3``, ``-2.5`` -> ``-2``. Built-in
    ``round`` gives ``2`` and ``-2``; a rule you can apply in your head matters here."""
    return math.floor(value + 0.5)


def _check_scale(scale: float) -> float:
    """Reject a scale that would make every conversion nonsense.

    Raises:
        ValueError: ``scale`` is not a finite number greater than zero.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return float(scale)


def scale_for(physical: int, logical: int) -> float:
    """The backing scale factor implied by a capture: physical / logical.

    Measured from the bitmap actually got, never assumed to be ``2.0`` because the machine
    is a Mac: a non-Retina scale factor, a nominal-resolution backend and an external
    1080p monitor all report a real ``1.0``.

    Raises:
        ValueError: either size is not a positive integer.
    """
    if physical <= 0 or logical <= 0:
        raise ValueError(f"sizes must be positive, got physical={physical}, logical={logical}")
    return physical / logical


def logical_to_physical(value: float, scale: float) -> int:
    """One logical pixel coordinate or length as a physical one."""
    return _round_half_up(value * _check_scale(scale))


def physical_to_logical(value: float, scale: float) -> int:
    """One physical pixel coordinate or length as a logical one.

    The conversion a detector owes the rest of the system: raw image coordinates are
    physical, and nothing above the control layer may see them.
    """
    return _round_half_up(value / _check_scale(scale))


def point_to_physical(point: Point, scale: float) -> tuple[int, int]:
    """A logical ``Point`` as a physical ``(x, y)`` pair."""
    scale = _check_scale(scale)
    return (_round_half_up(point.x * scale), _round_half_up(point.y * scale))


def point_from_physical(x: float, y: float, scale: float) -> Point:
    """A physical ``(x, y)`` pair as a logical ``Point``."""
    scale = _check_scale(scale)
    return Point(_round_half_up(x / scale), _round_half_up(y / scale))


def box_to_physical(box: Box, scale: float) -> tuple[int, int, int, int]:
    """A logical ``Box`` as a physical ``(x, y, w, h)``.

    Both edges convert and the width is their difference, so two boxes sharing an edge
    logically still share it physically - scaling the width alone would open a one-pixel
    gap. Negative sizes are treated as zero.
    """
    scale = _check_scale(scale)
    left = _round_half_up(box.x * scale)
    top = _round_half_up(box.y * scale)
    right = _round_half_up((box.x + max(box.w, 0)) * scale)
    bottom = _round_half_up((box.y + max(box.h, 0)) * scale)
    return (left, top, right - left, bottom - top)


def box_from_physical(x: float, y: float, w: float, h: float, scale: float) -> Box:
    """A physical ``(x, y, w, h)`` rectangle as a logical ``Box``.

    Near edges floor and far edges ceil, so the logical box covers every physical pixel
    the detection touched: exact at an integer scale, at most one logical pixel larger per
    side at a fractional one, which is the safe direction for a click target. Negative
    sizes are treated as zero.
    """
    scale = _check_scale(scale)
    left = math.floor(x / scale)
    top = math.floor(y / scale)
    right = math.ceil((x + max(w, 0.0)) / scale)
    bottom = math.ceil((y + max(h, 0.0)) / scale)
    return Box(left, top, right - left, bottom - top)


def clamp_point(point: Point, bounds: Box) -> Point:
    """The point nearest ``point`` that is actually inside ``bounds``.

    ``bounds`` covers ``x <= px < x + w``, so clamping ``Point(1512, 0)`` into a 1512-wide
    display gives ``1511``. A degenerate bound collapses that axis onto its origin. This
    is a CORRECTION, not a check - use ``bounds.contains`` to refuse instead.
    """
    max_x = bounds.x + bounds.w - 1 if bounds.w > 0 else bounds.x
    max_y = bounds.y + bounds.h - 1 if bounds.h > 0 else bounds.y
    return Point(
        min(max(point.x, bounds.x), max_x),
        min(max(point.y, bounds.y), max_y),
    )


def clamp_box(box: Box, bounds: Box) -> Box:
    """The part of ``box`` that lies inside ``bounds``.

    Edges are half-open here, unlike :func:`clamp_point`: a box may END at
    ``bounds.x + bounds.w``, while a POINT may not sit there because there is no pixel.
    A box entirely outside comes back with zero area, flush against the edge.
    """
    left = min(max(box.x, bounds.x), bounds.x + max(bounds.w, 0))
    top = min(max(box.y, bounds.y), bounds.y + max(bounds.h, 0))
    right = min(max(box.x + max(box.w, 0), bounds.x), bounds.x + max(bounds.w, 0))
    bottom = min(max(box.y + max(box.h, 0), bounds.y), bounds.y + max(bounds.h, 0))
    return Box(left, top, max(right - left, 0), max(bottom - top, 0))


def to_global(point: Point, viewport: Box) -> Point:
    """A viewport-relative logical point as a display-global one.

    The OS pointer APIs address the whole desktop, so a controller on the second display
    must add that origin before it moves the cursor.
    """
    return Point(point.x + viewport.x, point.y + viewport.y)


def to_local(point: Point, viewport: Box) -> Point:
    """A display-global logical point as a viewport-relative one; inverse of :func:`to_global`."""
    return Point(point.x - viewport.x, point.y - viewport.y)
