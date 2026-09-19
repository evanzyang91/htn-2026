"""Every coordinate conversion a desktop controller needs, in one pure place.

Nothing here imports a screen, a backend or a model - only :mod:`math` and the
shared value types - so it can be read, reasoned about and tested by hand. That is
the point: a click that lands at exactly twice the intended offset is an arithmetic
bug, and arithmetic bugs belong in a file small enough to check line by line.

Three coordinate spaces exist, and only the first two ever leave this package:

``local logical``
    What the rest of skillweaver speaks: logical pixels with the origin at the
    top-left of ``Controller.viewport()``. Every :class:`~skillweaver.contracts.Point`
    and :class:`~skillweaver.contracts.Box` in the codebase is in this space.

``global logical``
    Logical pixels with the origin at the top-left of the primary display. This is
    what the OS pointer APIs want, and what a display's bounds are expressed in. A
    display left of the primary one has a negative ``x``; negatives are handled
    throughout. Convert with :func:`to_global` / :func:`to_local`.

``physical``
    Raw bitmap pixels of a capture. On a Retina Mac there are two of them per
    logical pixel in each direction, so a 1512x982 display grabs as 3024x1964.
    Physical values are deliberately **not** ``Point`` or ``Box`` - they are plain
    ints and tuples - so that a physical coordinate cannot be mistaken for a logical
    one by a type checker or by a reader.

Rounding is half-up (``floor(v + 0.5)``) everywhere, never Python's bankers'
rounding, so that the same input always gives the same output and the tests can
state the expected number outright. Boxes are the exception: converting a box
*down* from physical grows it outwards (floor the near edge, ceil the far edge) so
that a detection never loses the pixels at its border. Both directions round-trip
exactly at integer scales - see the tests.
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
    """Round to the nearest integer with ties going up, for positives and negatives.

    ``2.5`` becomes ``3`` and ``-2.5`` becomes ``-2``. Python's built-in ``round``
    would give ``2`` and ``-2``; the difference matters only at half-pixels, but a
    rule you can apply in your head matters everywhere.
    """
    return math.floor(value + 0.5)


def _check_scale(scale: float) -> float:
    """Reject a scale that would make every conversion nonsense.

    Raises:
        ValueError: if ``scale`` is not a finite number greater than zero.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return float(scale)


def scale_for(physical: int, logical: int) -> float:
    """The backing scale factor implied by a capture: physical / logical.

    Measure the scale from the bitmap you actually got rather than assuming ``2.0``
    because the machine is a Mac. A display connected at a non-Retina scale factor,
    a screen-recording backend configured for nominal resolution, and an external
    1080p monitor all report a real ``1.0``, and a wrong assumption here is exactly
    the doubling bug this module exists to prevent.

    Raises:
        ValueError: if either size is not a positive integer.
    """
    if physical <= 0 or logical <= 0:
        raise ValueError(f"sizes must be positive, got physical={physical}, logical={logical}")
    return physical / logical


def logical_to_physical(value: float, scale: float) -> int:
    """One logical pixel coordinate or length as a physical one.

    ``logical_to_physical(100, 2.0) == 200``.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    return _round_half_up(value * _check_scale(scale))


def physical_to_logical(value: float, scale: float) -> int:
    """One physical pixel coordinate or length as a logical one.

    ``physical_to_logical(200, 2.0) == 100``. This is the conversion a detector owes
    the rest of the system: raw image coordinates are physical, and nothing above
    the control layer may see them.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    return _round_half_up(value / _check_scale(scale))


def point_to_physical(point: Point, scale: float) -> tuple[int, int]:
    """A logical :class:`~skillweaver.contracts.Point` as a physical ``(x, y)`` pair.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    scale = _check_scale(scale)
    return (_round_half_up(point.x * scale), _round_half_up(point.y * scale))


def point_from_physical(x: float, y: float, scale: float) -> Point:
    """A physical ``(x, y)`` pair as a logical :class:`~skillweaver.contracts.Point`.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    scale = _check_scale(scale)
    return Point(_round_half_up(x / scale), _round_half_up(y / scale))


def box_to_physical(box: Box, scale: float) -> tuple[int, int, int, int]:
    """A logical :class:`~skillweaver.contracts.Box` as a physical ``(x, y, w, h)``.

    Both edges are converted and the width is their difference, so two boxes that
    share an edge logically still share it physically - scaling the width on its own
    would let a rounding difference open a one-pixel gap. Negative widths and
    heights are treated as zero.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    scale = _check_scale(scale)
    left = _round_half_up(box.x * scale)
    top = _round_half_up(box.y * scale)
    right = _round_half_up((box.x + max(box.w, 0)) * scale)
    bottom = _round_half_up((box.y + max(box.h, 0)) * scale)
    return (left, top, right - left, bottom - top)


def box_from_physical(x: float, y: float, w: float, h: float, scale: float) -> Box:
    """A physical ``(x, y, w, h)`` rectangle as a logical
    :class:`~skillweaver.contracts.Box`.

    The near edges floor and the far edges ceil, so the logical box always covers
    every physical pixel the detection touched. At an integer scale this is exact
    and round-trips with :func:`box_to_physical`; at a fractional one it grows by at
    most one logical pixel per side, which is the safe direction for a click target.
    Negative widths and heights are treated as zero.

    Raises:
        ValueError: if ``scale`` is not finite and positive.
    """
    scale = _check_scale(scale)
    left = math.floor(x / scale)
    top = math.floor(y / scale)
    right = math.ceil((x + max(w, 0.0)) / scale)
    bottom = math.ceil((y + max(h, 0.0)) / scale)
    return Box(left, top, right - left, bottom - top)


def clamp_point(point: Point, bounds: Box) -> Point:
    """The point nearest ``point`` that is actually inside ``bounds``.

    ``bounds`` covers ``x <= px < x + w``, so the largest addressable coordinate is
    ``bounds.x + bounds.w - 1``: clamping ``Point(1512, 0)`` into a 1512-wide display
    gives ``1511``, the rightmost real pixel, not ``1512``, which is off the screen.
    A degenerate bound (zero or negative width or height) collapses that axis onto
    its origin.

    This is a *correction*, not a check. Use ``bounds.contains(point)`` when you
    need to refuse an out-of-range action instead of quietly moving it.
    """
    max_x = bounds.x + bounds.w - 1 if bounds.w > 0 else bounds.x
    max_y = bounds.y + bounds.h - 1 if bounds.h > 0 else bounds.y
    return Point(
        min(max(point.x, bounds.x), max_x),
        min(max(point.y, bounds.y), max_y),
    )


def clamp_box(box: Box, bounds: Box) -> Box:
    """The part of ``box`` that lies inside ``bounds``.

    Edges are half-open here, unlike :func:`clamp_point`: a box may end at
    ``bounds.x + bounds.w`` because that edge is exclusive, while a *point* may not
    sit on it because there is no pixel there. A box entirely outside ``bounds``
    comes back with zero area, flush against the edge it was beyond.
    """
    left = min(max(box.x, bounds.x), bounds.x + max(bounds.w, 0))
    top = min(max(box.y, bounds.y), bounds.y + max(bounds.h, 0))
    right = min(max(box.x + max(box.w, 0), bounds.x), bounds.x + max(bounds.w, 0))
    bottom = min(max(box.y + max(box.h, 0), bounds.y), bounds.y + max(bounds.h, 0))
    return Box(left, top, max(right - left, 0), max(bottom - top, 0))


def to_global(point: Point, viewport: Box) -> Point:
    """A viewport-relative logical point as a display-global logical point.

    The OS pointer APIs address the whole desktop, so a controller driving the
    second display - whose bounds might be ``Box(1512, 0, 1920, 1080)`` - must add
    that origin before it moves the cursor. Inside skillweaver, ``Point(0, 0)`` is
    always the top-left of the viewport.
    """
    return Point(point.x + viewport.x, point.y + viewport.y)


def to_local(point: Point, viewport: Box) -> Point:
    """A display-global logical point as a viewport-relative one; inverse of
    :func:`to_global`."""
    return Point(point.x - viewport.x, point.y - viewport.y)
