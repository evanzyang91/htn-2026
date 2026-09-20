"""Loading, saving, resizing and converting ``Screenshot``.

Every helper keeps ONE invariant: the PNG is at PHYSICAL resolution while
``width``/``height`` are LOGICAL pixels and ``scale = physical / logical``. Get it wrong
and every click lands at twice or half the intended coordinates.

``shot.width``/``height`` are logical; ``physical_size(shot)`` is what the bytes decode to;
``shot.to_array()`` is logical-sized by default, so ``array[y, x]`` IS ``Point(x, y)``.

:func:`rescale` changes pixel density and keeps the logical frame; :func:`resize` changes
the frame and keeps the density.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from skillweaver.contracts import Screenshot, utcnow
from skillweaver.errors import PerceptionError

if TYPE_CHECKING:  # heavy imports stay out of module import time
    import numpy as np

__all__ = [
    "decode_png",
    "encode_png",
    "from_array",
    "from_png",
    "load_screenshot",
    "logical_size",
    "physical_size",
    "rescale",
    "resize",
    "save_screenshot",
    "to_array",
]


def _check_scale(scale: float) -> float:
    if not isinstance(scale, int | float) or scale <= 0 or scale != scale:
        raise PerceptionError(f"screenshot scale must be a positive number, got {scale!r}")
    return float(scale)


def decode_png(png: bytes) -> np.ndarray:
    """PNG bytes as an ``H x W x 3`` ``uint8`` RGB array at native size.

    PHYSICAL resolution, so coordinates from it must be divided by ``scale``.
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(png)).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PerceptionError(f"PNG bytes could not be decoded: {exc}") from exc
    return np.asarray(image, dtype=np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    """An ``H x W x 3``, ``H x W x 4`` or grayscale ``uint8`` array as PNG bytes."""
    import numpy as np
    from PIL import Image

    data = np.asarray(array)
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[2] not in (3, 4)):
        raise PerceptionError(f"cannot encode array of shape {data.shape} as PNG")
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    buffer = io.BytesIO()
    try:
        Image.fromarray(data).save(buffer, format="PNG")
    except (OSError, ValueError) as exc:
        raise PerceptionError(f"array could not be encoded as PNG: {exc}") from exc
    return buffer.getvalue()


def physical_size(shot: Screenshot) -> tuple[int, int]:
    """The ``(width, height)`` the PNG is EXPECTED to decode to, derived from ``scale``."""
    return (round(shot.width * shot.scale), round(shot.height * shot.scale))


def logical_size(shot: Screenshot) -> tuple[int, int]:
    """The ``(width, height)`` of the screenshot in LOGICAL pixels."""
    return (shot.width, shot.height)


def to_array(shot: Screenshot, *, logical: bool = True) -> np.ndarray:
    """``shot.to_array(logical=logical)``, spelled as a function.

    ``logical=False`` gives the sharper native-resolution array, which is what OCR and
    detectors run on before dividing their coordinates by ``shot.scale``.
    """
    return shot.to_array(logical=logical)


def from_array(
    array: np.ndarray,
    *,
    scale: float = 1.0,
    captured_at: datetime | None = None,
) -> Screenshot:
    """A ``Screenshot`` from an array of PHYSICAL pixels.

    The array's size IS the physical size, so passing a logical-resolution array with
    ``scale=2.0`` is a bug: the screenshot would claim half the size it looks.
    """
    scale = _check_scale(scale)
    png = encode_png(array)
    height, width = int(array.shape[0]), int(array.shape[1])
    return Screenshot(
        png=png,
        width=max(1, round(width / scale)),
        height=max(1, round(height / scale)),
        scale=scale,
        captured_at=captured_at or utcnow(),
    )


def from_png(
    png: bytes,
    *,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    captured_at: datetime | None = None,
) -> Screenshot:
    """A ``Screenshot`` from PNG bytes of PHYSICAL pixels.

    Pass ``width``/``height`` only when the true logical size is known independently (a
    viewport the controller reported) and rounding would otherwise be ambiguous.
    """
    scale = _check_scale(scale)
    array = decode_png(png)
    physical_h, physical_w = int(array.shape[0]), int(array.shape[1])
    return Screenshot(
        png=png,
        width=width if width is not None else max(1, round(physical_w / scale)),
        height=height if height is not None else max(1, round(physical_h / scale)),
        scale=scale,
        captured_at=captured_at or utcnow(),
    )


def load_screenshot(
    path: str | Path,
    *,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
    captured_at: datetime | None = None,
) -> Screenshot:
    """Read a PNG file into a ``Screenshot``.

    The file is PHYSICAL pixels, exactly as a capture is, so a 2x Retina fixture must be
    loaded with ``scale=2.0`` for its boxes to come out logical.
    """
    file = Path(path)
    try:
        png = file.read_bytes()
    except OSError as exc:
        raise PerceptionError(f"screenshot could not be read from {file}: {exc}") from exc
    return from_png(png, scale=scale, width=width, height=height, captured_at=captured_at)


def save_screenshot(shot: Screenshot, path: str | Path) -> Path:
    """Write ``shot.png`` to ``path`` verbatim.

    Only the pixels: ``scale`` lives outside the file, so a reader must pass the same
    ``scale`` to :func:`load_screenshot`.
    """
    file = Path(path)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(shot.png)
    except OSError as exc:
        raise PerceptionError(f"screenshot could not be written to {file}: {exc}") from exc
    return file


def rescale(shot: Screenshot, scale: float) -> Screenshot:
    """Change pixel density, keeping the LOGICAL frame identical.

    Every ``Box`` computed against the result is still valid against ``shot``, so
    ``rescale(shot, 1.0)`` is the cheap way to hand a Retina capture to something that
    ignores ``scale``.
    """
    scale = _check_scale(scale)
    if scale == shot.scale:
        return shot
    target = (max(1, round(shot.width * scale)), max(1, round(shot.height * scale)))
    return Screenshot(
        png=_resized_png(shot, target),
        width=shot.width,
        height=shot.height,
        scale=scale,
        captured_at=shot.captured_at,
    )


def resize(
    shot: Screenshot,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Screenshot:
    """Change the LOGICAL frame, keeping ``scale`` and so the pixel density.

    One of ``width``/``height`` scales proportionally, both force a size. Boxes do NOT
    survive: coordinates from the result are in the new frame and must be scaled by
    ``shot.width / result.width`` to mean anything on the original.

    Raises:
        PerceptionError: neither dimension given, a non-positive dimension, or an
            undecodable PNG.
    """
    if width is None and height is None:
        raise PerceptionError("resize needs width, height or both")
    if width is None:
        assert height is not None
        width = max(1, round(shot.width * height / shot.height))
    elif height is None:
        height = max(1, round(shot.height * width / shot.width))
    if width <= 0 or height <= 0:
        raise PerceptionError(f"resize target must be positive, got {width}x{height}")
    if (width, height) == (shot.width, shot.height):
        return shot
    target = (max(1, round(width * shot.scale)), max(1, round(height * shot.scale)))
    return Screenshot(
        png=_resized_png(shot, target),
        width=width,
        height=height,
        scale=shot.scale,
        captured_at=shot.captured_at,
    )


def _resized_png(shot: Screenshot, target: tuple[int, int]) -> bytes:
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(shot.png)).convert("RGB")
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PerceptionError(f"screenshot PNG could not be resized: {exc}") from exc
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
