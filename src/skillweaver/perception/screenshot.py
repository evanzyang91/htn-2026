"""Loading, saving, resizing and converting :class:`~skillweaver.contracts.Screenshot`.

Every helper here exists to keep ONE invariant true: a ``Screenshot`` carries its PNG
at PHYSICAL resolution while ``width``/``height`` are LOGICAL pixels, and
``scale = physical / logical``. Get that wrong and every click in the system lands at
twice (or half) the intended coordinates, so the conversions live in one place instead
of being re-derived by each caller.

The three sizes to keep straight:

``shot.width``, ``shot.height``
    Logical. What a ``Box`` or ``Point`` is measured in.
``physical_size(shot)``
    What the PNG bytes actually decode to.
``shot.to_array()``
    Logical-sized by default, so ``array[y, x]`` IS ``Point(x, y)``.

Use :func:`rescale` (change pixel density, keep the logical frame) and :func:`resize`
(change the logical frame, keep the density) rather than resizing arrays by hand.
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
    """Decode PNG bytes into an ``H x W x 3`` ``uint8`` RGB array at native size.

    No resizing happens: the result is at PHYSICAL resolution, so coordinates taken
    from it must be divided by the screenshot's ``scale``.

    Raises:
        PerceptionError: if the bytes cannot be decoded as an image.
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
    """Encode an ``H x W x 3`` (or ``H x W x 4``, or grayscale ``H x W``) ``uint8``
    array as PNG bytes.

    Raises:
        PerceptionError: if the array is not an encodable image.
    """
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
    """The ``(width, height)`` the PNG decodes to, derived from ``scale``.

    This is what the bytes are *expected* to be; :func:`load_screenshot` is what
    guarantees the stored bytes agree with it.
    """
    return (round(shot.width * shot.scale), round(shot.height * shot.scale))


def logical_size(shot: Screenshot) -> tuple[int, int]:
    """The ``(width, height)`` of the screenshot in LOGICAL pixels."""
    return (shot.width, shot.height)


def to_array(shot: Screenshot, *, logical: bool = True) -> np.ndarray:
    """``shot.to_array(logical=logical)``, spelled as a function.

    ``logical=True`` gives a logical-sized array whose indices are logical pixels;
    ``logical=False`` gives the sharper native-resolution array, which is what OCR and
    detectors should run on before dividing their coordinates by ``shot.scale``.
    """
    return shot.to_array(logical=logical)


def from_array(
    array: np.ndarray,
    *,
    scale: float = 1.0,
    captured_at: datetime | None = None,
) -> Screenshot:
    """Build a ``Screenshot`` from an array of PHYSICAL pixels.

    The array's own size is the physical size, so the logical size becomes
    ``array.shape[1] / scale`` by ``array.shape[0] / scale``. Passing a
    logical-resolution array with ``scale=2.0`` is therefore a bug: the resulting
    screenshot would claim to be half the size it looks.

    Raises:
        PerceptionError: if ``scale`` is not positive or the array is not an image.
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
    """Build a ``Screenshot`` from PNG bytes of PHYSICAL pixels.

    The logical size is derived from the decoded size and ``scale``. Pass ``width``
    and ``height`` only when the true logical size is known independently (a viewport
    reported by the controller) and rounding would otherwise be ambiguous.

    Raises:
        PerceptionError: if ``scale`` is not positive or the bytes are not a decodable
            image.
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
    """Read a PNG file into a ``Screenshot`` whose ``scale`` is ``scale``.

    The file is treated as PHYSICAL pixels, exactly as a capture would be, so a 2x
    Retina fixture must be loaded with ``scale=2.0`` for its boxes to come out in
    logical pixels.

    Raises:
        PerceptionError: if the file is missing or is not a decodable image.
    """
    file = Path(path)
    try:
        png = file.read_bytes()
    except OSError as exc:
        raise PerceptionError(f"screenshot could not be read from {file}: {exc}") from exc
    return from_png(png, scale=scale, width=width, height=height, captured_at=captured_at)


def save_screenshot(shot: Screenshot, path: str | Path) -> Path:
    """Write ``shot.png`` to ``path`` verbatim and return the path.

    Only the pixels are written: ``scale`` lives outside the file, so whoever reads it
    back must pass the same ``scale`` to :func:`load_screenshot`. Parent directories
    are created.

    Raises:
        PerceptionError: if the file cannot be written.
    """
    file = Path(path)
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(shot.png)
    except OSError as exc:
        raise PerceptionError(f"screenshot could not be written to {file}: {exc}") from exc
    return file


def rescale(shot: Screenshot, scale: float) -> Screenshot:
    """Change pixel density while keeping the LOGICAL frame identical.

    Every ``Box`` computed against the result is still valid against ``shot``, which is
    what makes this safe: ``rescale(shot, 1.0)`` is the cheap way to hand a Retina
    capture to something that ignores ``scale``.

    Raises:
        PerceptionError: if ``scale`` is not positive or the PNG cannot be decoded.
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
    """Change the LOGICAL frame, keeping ``scale`` - and therefore pixel density - as
    it was.

    Give one of ``width``/``height`` to scale proportionally, or both to force a size.
    Boxes do NOT survive this: coordinates from the result are in the new frame and
    must be scaled by ``shot.width / result.width`` to mean anything on the original.

    Raises:
        PerceptionError: if neither dimension is given, a dimension is not positive, or
            the PNG cannot be decoded.
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
