"""Cropping a screenshot to a region, and moving coordinates between the two frames.

Why this module exists: re-detecting inside a region is how the agent gets a close
look at a toolbar, a table cell or a dialog. Cropping is the easy half; the half that
silently breaks everything is the coordinate bookkeeping afterwards, because a box
found in a crop is measured from the CROP's top-left, not the screen's.

:func:`crop` therefore never returns a bare screenshot. It returns a :class:`Crop`
that remembers where it came from, and the region's actual bounds after clamping, so
:meth:`Crop.to_parent` can put detections back where they belong:

    region = crop(shot, Box(300, 200, 400, 120))
    found = detector.detect(region.screenshot)     # boxes local to the region
    on_screen = region.to_parent_elements(found)   # boxes on the original screen

All coordinates here are LOGICAL pixels, in and out.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillweaver.contracts import Box, Element, Point, Screenshot
from skillweaver.errors import PerceptionError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "Crop",
    "clamp_box",
    "crop",
    "crop_around",
    "grow_box",
    "translate_box",
    "translate_elements",
    "translate_point",
]


def clamp_box(box: Box, width: int, height: int) -> Box:
    """``box`` trimmed to the ``width`` x ``height`` frame at the origin.

    Negative coordinates are pulled in to ``0``, an overhanging right or bottom edge is
    cut at the frame, and a box wholly outside the frame clamps to zero width or
    height rather than to something negative. Never raises.
    """
    x0 = min(max(box.x, 0), width)
    y0 = min(max(box.y, 0), height)
    x1 = min(max(box.x + max(box.w, 0), 0), width)
    y1 = min(max(box.y + max(box.h, 0), 0), height)
    return Box(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))


def grow_box(box: Box, margin: int) -> Box:
    """``box`` expanded by ``margin`` logical pixels on every side.

    Useful before cropping, since a detector reads a control more reliably with a
    little context around it. A negative ``margin`` shrinks, never past zero size.
    Clamp the result to the screenshot yourself, or let :func:`crop` do it.
    """
    w = max(box.w + 2 * margin, 0)
    h = max(box.h + 2 * margin, 0)
    return Box(box.x - margin, box.y - margin, w, h)


def translate_point(point: Point, dx: int, dy: int) -> Point:
    """``point`` moved by ``(dx, dy)`` logical pixels."""
    return Point(point.x + dx, point.y + dy)


def translate_box(box: Box, dx: int, dy: int) -> Box:
    """``box`` moved by ``(dx, dy)`` logical pixels; its size is unchanged."""
    return Box(box.x + dx, box.y + dy, box.w, box.h)


def translate_elements(elements: Iterable[Element], dx: int, dy: int) -> list[Element]:
    """Every element's box moved by ``(dx, dy)``; everything else, ``stable_id``
    included, is left alone.

    ``stable_id`` survives deliberately: a translated element is the same element seen
    in a different frame, not a new one.
    """
    return [dataclasses.replace(e, box=translate_box(e.box, dx, dy)) for e in elements]


@dataclass(frozen=True, slots=True)
class Crop:
    """A cropped screenshot together with the offset needed to undo the crop.

    Attributes:
        screenshot: The cropped region as its own ``Screenshot``, with logical size
            equal to ``box.w`` x ``box.h`` and the same ``scale`` as the original.
            Detectors and OCR can be pointed at it with no special handling.
        box: The region actually cropped, in the ORIGINAL screenshot's logical
            coordinates, AFTER clamping. This is the offset :meth:`to_parent_box` uses.
        requested: The region that was asked for, before clamping and including any
            ``margin``. Differs from ``box`` exactly when the request ran off an edge.
        source: The screenshot that was cropped.
    """

    screenshot: Screenshot
    box: Box
    requested: Box
    source: Screenshot = dataclasses.field(repr=False)

    @property
    def offset(self) -> Point:
        """The crop's top-left in the original's coordinates; add it to go outward."""
        return Point(self.box.x, self.box.y)

    @property
    def clamped(self) -> bool:
        """Whether the region requested had to be trimmed to fit the screenshot."""
        return self.box != self.requested

    def to_parent_point(self, point: Point) -> Point:
        """A point measured in the crop, re-expressed on the original screenshot."""
        return translate_point(point, self.box.x, self.box.y)

    def to_parent_box(self, box: Box) -> Box:
        """A box measured in the crop, re-expressed on the original screenshot."""
        return translate_box(box, self.box.x, self.box.y)

    def to_parent_elements(self, elements: Iterable[Element]) -> list[Element]:
        """Elements detected in the crop, re-expressed on the original screenshot.

        This is the call that makes re-detection inside a region usable by the rest of
        the system; skip it and every click is off by the crop's offset.
        """
        return translate_elements(elements, self.box.x, self.box.y)

    def to_local_point(self, point: Point) -> Point:
        """A point on the original screenshot, re-expressed inside the crop.

        The result may fall outside the crop; that is the caller's business.
        """
        return translate_point(point, -self.box.x, -self.box.y)

    def to_local_box(self, box: Box) -> Box:
        """A box on the original screenshot, re-expressed inside the crop."""
        return translate_box(box, -self.box.x, -self.box.y)


def crop(shot: Screenshot, box: Box, *, margin: int = 0) -> Crop:
    """Crop ``shot`` to ``box`` (optionally grown by ``margin``), clamping at the edges.

    The requested region is clamped to the screenshot, so asking for a box that hangs
    off the right edge gives you the visible part instead of an error - check
    ``result.box`` (or ``result.clamped``) when the difference matters. The PNG is cut
    at PHYSICAL resolution and ``scale`` is carried over unchanged, so the crop is just
    as sharp as the original and its logical coordinates keep the same units.

    Raises:
        PerceptionError: if the clamped region has zero area - there is no such image
            to return - or if the PNG cannot be decoded.
    """
    import io

    from PIL import Image, UnidentifiedImageError

    requested = grow_box(box, margin) if margin else box
    region = clamp_box(requested, shot.width, shot.height)
    if region.area <= 0:
        raise PerceptionError(
            f"crop region {requested} does not overlap the {shot.width}x{shot.height} screenshot"
        )

    scale = shot.scale
    try:
        image = Image.open(io.BytesIO(shot.png)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PerceptionError(f"screenshot PNG could not be decoded for cropping: {exc}") from exc

    # Cut in physical pixels, then clamp again against the real bytes: a PNG whose size
    # disagrees with width*scale must not produce an out-of-range crop.
    px0 = min(max(round(region.x * scale), 0), image.width)
    py0 = min(max(round(region.y * scale), 0), image.height)
    px1 = min(max(round((region.x + region.w) * scale), px0 + 1), image.width)
    py1 = min(max(round((region.y + region.h) * scale), py0 + 1), image.height)
    if px1 <= px0 or py1 <= py0:
        raise PerceptionError(
            f"crop region {region} maps to an empty area of the {image.width}x{image.height} PNG"
        )

    buffer = io.BytesIO()
    image.crop((px0, py0, px1, py1)).save(buffer, format="PNG")
    cropped = Screenshot(
        png=buffer.getvalue(),
        width=region.w,
        height=region.h,
        scale=scale,
        captured_at=shot.captured_at,
    )
    return Crop(screenshot=cropped, box=region, requested=requested, source=shot)


def crop_around(shot: Screenshot, elements: Sequence[Element], *, margin: int = 8) -> Crop:
    """Crop to the union of ``elements``' boxes plus ``margin``.

    The obvious way to take a closer look at a group of controls - a row, a toolbar -
    without computing the union by hand.

    Raises:
        PerceptionError: if ``elements`` is empty, or the union does not overlap the
            screenshot.
    """
    if not elements:
        raise PerceptionError("crop_around needs at least one element")
    x0 = min(e.box.x for e in elements)
    y0 = min(e.box.y for e in elements)
    x1 = max(e.box.x + e.box.w for e in elements)
    y1 = max(e.box.y + e.box.h for e in elements)
    return crop(shot, Box(x0, y0, x1 - x0, y1 - y0), margin=margin)
