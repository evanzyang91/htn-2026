"""Translation between ``Element`` and YOLO labels.

skillweaver speaks integer LOGICAL pixels as ``(x, y, w, h)``; YOLO speaks normalized
``0..1`` floats as ``(cx, cy, w, h)`` after an integer class id. Everything here is pure,
because an off-by-one does not crash anything - it quietly trains a model that aims half a
button to the left.

Normalized labels are SCALE-FREE, so a dataset can mix device scale factors as long as the
LOGICAL size is what normalized them; that is why :func:`to_label_text` takes it.

The class map is FROZEN. :data:`CLASS_NAMES` is a literal rather than derived from
``ElementKind`` because trained weights encode class IDS: reordering the enum would turn
every detected button into a checkbox. Appending is safe; anything else invalidates
existing weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from skillweaver.contracts import Box, Element, ElementKind, ElementSource

CLASS_NAMES: tuple[str, ...] = (
    "button",
    "text_field",
    "checkbox",
    "radio",
    "link",
    "icon",
    "menu",
    "tab",
    "row",
    "image",
    "text",
    "other",
)
"""Index is the YOLO class id, value the ``ElementKind``. The order is part of the trained
weights - see the module docstring before touching it."""

_COORD_DECIMALS = 6
"""The YOLO convention, and enough for an exact round trip: the worst rounding error is
``5e-7``, under half a pixel for any image narrower than 500,000 pixels."""

_CLASS_ID_BY_KIND: dict[ElementKind, int] = {
    ElementKind(name): index for index, name in enumerate(CLASS_NAMES)
}


def class_id(kind: ElementKind) -> int:
    """The YOLO class id for ``kind``.

    Raises:
        KeyError: ``kind`` is missing from :data:`CLASS_NAMES`, which can only happen if a
            new ``ElementKind`` was added without extending the map.
    """
    return _CLASS_ID_BY_KIND[kind]


def kind_of(class_id_: int) -> ElementKind:
    """The ``ElementKind`` a YOLO class id stands for.

    Raises:
        ValueError: ``class_id_`` is outside the class map.
    """
    if not 0 <= class_id_ < len(CLASS_NAMES):
        raise ValueError(
            f"class id {class_id_} is outside the {len(CLASS_NAMES)}-class map; "
            f"the weights were trained against a different CLASS_NAMES"
        )
    return ElementKind(CLASS_NAMES[class_id_])


def box_to_yolo(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    """Normalize ``box`` to YOLO's ``(cx, cy, w, h)``, all four in ``0.0..1.0``.

    The center is the GEOMETRIC centre, ``x + w / 2``, not the integer ``Box.center`` -
    halving an odd width has to stay fractional or the round trip loses a pixel. Values
    are clamped, so a box hanging off the edge still yields a legal label.

    Raises:
        ValueError: ``width`` or ``height`` is not positive.
    """
    _check_size(width, height)
    cx = (box.x + box.w / 2) / width
    cy = (box.y + box.h / 2) / height
    nw = box.w / width
    nh = box.h / height
    return (_clamp01(cx), _clamp01(cy), _clamp01(nw), _clamp01(nh))


def box_from_yolo(cx: float, cy: float, nw: float, nh: float, width: int, height: int) -> Box:
    """YOLO's normalized ``(cx, cy, nw, nh)`` back as an integer logical :class:`Box`.

    The inverse of :func:`box_to_yolo` for any box inside the frame: the size is recovered
    FIRST and the corner derived from it, so rounding a half-pixel centre cannot leak into
    the width.

    Raises:
        ValueError: ``width`` or ``height`` is not positive.
    """
    _check_size(width, height)
    w = round(nw * width)
    h = round(nh * height)
    x = round(cx * width - nw * width / 2)
    y = round(cy * height - nh * height / 2)
    return Box(x=x, y=y, w=max(w, 0), h=max(h, 0))


def to_label_line(element: Element, width: int, height: int) -> str:
    """One YOLO label line: ``"<class> <cx> <cy> <w> <h>"``.

    ``width``/``height`` are the LOGICAL size the box was measured in, never a Retina PNG's.
    """
    cx, cy, nw, nh = box_to_yolo(element.box, width, height)
    numbers = " ".join(f"{value:.{_COORD_DECIMALS}f}" for value in (cx, cy, nw, nh))
    return f"{class_id(element.kind)} {numbers}"


def to_label_text(elements: Iterable[Element], width: int, height: int) -> str:
    """One YOLO ``.txt`` label file, one line per element.

    Empty for no elements, which is a legal label file meaning "no objects in this image".
    """
    lines = [to_label_line(element, width, height) for element in elements]
    return "".join(f"{line}\n" for line in lines)


def parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    """Split one label line into ``(class_id, cx, cy, nw, nh)``.

    Raises:
        ValueError: not exactly five numbers, or a non-integer class id.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"a YOLO label line needs 5 fields, got {len(parts)}: {line!r}")
    try:
        cls = int(parts[0])
        cx, cy, nw, nh = (float(value) for value in parts[1:])
    except ValueError as exc:
        raise ValueError(f"malformed YOLO label line {line!r}: {exc}") from exc
    return cls, cx, cy, nw, nh


def element_from_label_line(
    line: str,
    width: int,
    height: int,
    *,
    confidence: float = 1.0,
    source: ElementSource = ElementSource.dom,
) -> Element:
    """One label line as an :class:`Element` with a logical-pixel box.

    A label file carries no text and no confidence, so ``text`` comes back empty and
    ``confidence`` is whatever the caller passes.

    Raises:
        ValueError: a malformed line, unknown class id or non-positive size.
    """
    cls, cx, cy, nw, nh = parse_label_line(line)
    return Element(
        box=box_from_yolo(cx, cy, nw, nh, width, height),
        kind=kind_of(cls),
        text="",
        confidence=confidence,
        source=source,
    )


def elements_from_label_text(
    text: str,
    width: int,
    height: int,
    *,
    confidence: float = 1.0,
    source: ElementSource = ElementSource.dom,
) -> list[Element]:
    """A whole label file as elements, in file order; blank lines are skipped.

    Raises:
        ValueError: a malformed line, unknown class id or non-positive size.
    """
    return [
        element_from_label_line(line, width, height, confidence=confidence, source=source)
        for line in text.splitlines()
        if line.strip()
    ]


def dataset_yaml(root: str, *, train: str = "images/train", val: str = "images/val") -> str:
    """The ``data.yaml`` an ultralytics training run reads.

    ``root`` should be absolute, so the file stays usable from any working directory. The
    ``names`` block is :data:`CLASS_NAMES` in class-id order, which is what ties the
    weights back to ``ElementKind``; written by hand rather than through ``yaml.dump`` so
    the ids line up in a column a human can check.
    """
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return (
        "# Generated by scripts/build_ui_dataset.py - do not edit by hand.\n"
        "# Class ids come from skillweaver.perception.labeling.CLASS_NAMES and are\n"
        "# frozen: reordering them invalidates every trained weight file.\n"
        f"path: {root}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )


def label_stem(image_name: str) -> str:
    """``"shot_003.png"`` -> ``"shot_003.txt"``; ultralytics pairs them by this stem."""
    stem, _, _ = image_name.rpartition(".")
    return f"{stem or image_name}.txt"


def recall_at_iou(
    expected: Sequence[Element],
    detected: Sequence[Element],
    *,
    iou: float = 0.5,
    match_kind: bool = True,
) -> float:
    """Fraction of ``expected`` elements a detection matched, at an IoU threshold.

    Each expected element claims the best still-unclaimed detection, so one big detection
    covering three buttons cannot count three times. ``1.0`` for empty ``expected``.

    """
    if not expected:
        return 1.0
    claimed: set[int] = set()
    hits = 0
    for want in expected:
        best_index, best_score = -1, -1.0
        for index, got in enumerate(detected):
            if index in claimed or (match_kind and got.kind is not want.kind):
                continue
            score = want.box.iou(got.box)
            if score >= iou and score > best_score:
                best_index, best_score = index, score
        if best_index >= 0:
            claimed.add(best_index)
            hits += 1
    return hits / len(expected)


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _check_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
