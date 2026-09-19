"""The YOLO detector and the label arithmetic underneath it.

Two halves, and they fail for different reasons.

The **labeling** half is pure arithmetic and is tested hard, because its failure
mode is silence: a half-pixel drift in the normalization does not raise, it trains
a model that aims slightly wrong, and by the time anyone notices, the weights are
the suspect and the converter is not. Every expected number below is worked out by
hand in the test and written as a literal - calling the converter to check the
converter would agree with itself about anything.

The **detector** half needs weights. They are committed, so these tests normally
run; they skip loudly, with the two commands that rebuild them, when
``SKILLWEAVER_DATA_DIR`` points at a tree without them. What can be tested with no
model at all - lazy loading, the missing-weights error, the physical-to-logical
conversion against a stub - is tested unconditionally either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from skillweaver.contracts import (
    Box,
    Detector,
    Element,
    ElementKind,
    ElementSource,
    Screenshot,
    utcnow,
)
from skillweaver.errors import PerceptionError
from skillweaver.perception.detect_yolo import YoloDetector, default_weights_path
from skillweaver.perception.labeling import (
    CLASS_NAMES,
    box_from_yolo,
    box_to_yolo,
    class_id,
    dataset_yaml,
    element_from_label_line,
    elements_from_label_text,
    kind_of,
    label_stem,
    parse_label_line,
    recall_at_iou,
    to_label_line,
    to_label_text,
)

FIXTURES = Path(__file__).parent / "fixtures"

MEAN_RECALL_FLOOR = 0.85
"""Mean fraction of ground-truth elements the shipped detector must find across the
committed fixture frames, at IoU 0.5 and the same element kind.

Pinned from what the shipped weights actually measure - 0.908 - not from what would
be nice. A floor nobody can hit is a test that gets deleted; a floor far below the
truth never catches a regression. ``fixtures/README.md`` records the run. Raise it
when a better model lands.
"""

FRAME_RECALL_FLOOR = 0.70
"""No single frame may fall below this, so a good average cannot hide one blind
screen. The weakest fixture is the settings dialog at 0.78: its modal scrim dims
the page behind it and the detector loses a third of the greyed-out body text."""

CONTROL_RECALL_FLOOR = 0.90
"""Pooled recall over the INTERACTIVE kinds only - buttons, fields, checkboxes,
menus, rows - which is what the agent actually clicks. It is higher than the
all-kinds floor on purpose: missing a paragraph of text costs nothing, because OCR
reads text anyway, and missing the Send button ends the run. Measured: 0.957.
"""

CONTROL_KINDS = frozenset(
    {
        ElementKind.button,
        ElementKind.text_field,
        ElementKind.checkbox,
        ElementKind.radio,
        ElementKind.link,
        ElementKind.menu,
        ElementKind.tab,
        ElementKind.row,
    }
)


# --------------------------------------------------------------------------------------
# The class map
# --------------------------------------------------------------------------------------


def test_class_map_is_frozen() -> None:
    """Class ids are baked into trained weights, so this literal is a contract.

    Reordering it silently turns every detected button into a checkbox: the model
    still reports class 0 and this module still says class 0 means something, but
    the two no longer agree. Appending a new kind at the end is safe; anything else
    means retraining.
    """
    assert CLASS_NAMES == (
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


def test_class_map_covers_every_element_kind() -> None:
    assert set(CLASS_NAMES) == {kind.value for kind in ElementKind}
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))


def test_class_id_and_kind_round_trip_for_every_kind() -> None:
    for kind in ElementKind:
        assert kind_of(class_id(kind)) is kind
    for index in range(len(CLASS_NAMES)):
        assert class_id(kind_of(index)) == index


def test_class_id_matches_the_literal_positions() -> None:
    assert class_id(ElementKind.button) == 0
    assert class_id(ElementKind.text_field) == 1
    assert class_id(ElementKind.checkbox) == 2
    assert class_id(ElementKind.other) == 11


@pytest.mark.parametrize("bad", [-1, 12, 999])
def test_unknown_class_id_is_refused(bad: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        kind_of(bad)


# --------------------------------------------------------------------------------------
# Box normalization, checked by hand
# --------------------------------------------------------------------------------------


def test_box_to_yolo_arithmetic_on_a_1000x500_image() -> None:
    """A 100x50 box at (200, 100) on a 1000x500 image.

    center x = 200 + 100/2 = 250 -> 250/1000 = 0.25
    center y = 100 +  50/2 = 125 -> 125/500  = 0.25
    width    = 100/1000 = 0.1
    height   =  50/500  = 0.1
    """
    assert box_to_yolo(Box(200, 100, 100, 50), 1000, 500) == (0.25, 0.25, 0.1, 0.1)


def test_box_to_yolo_arithmetic_on_a_1280x800_image() -> None:
    """A 64x40 box at (128, 80) on a 1280x800 image.

    center x = 128 + 32 = 160 -> 160/1280 = 0.125
    center y =  80 + 20 = 100 -> 100/800  = 0.125
    width    =  64/1280 = 0.05
    height   =  40/800  = 0.05
    """
    assert box_to_yolo(Box(128, 80, 64, 40), 1280, 800) == (0.125, 0.125, 0.05, 0.05)


def test_box_to_yolo_keeps_an_odd_size_centered_on_a_half_pixel() -> None:
    """A 3x5 box at (0, 0) on a 100x100 image has its center at (1.5, 2.5).

    Rounding that to the integer ``Box.center`` would lose half a pixel each way,
    and the box would come back one pixel narrower than it went in.
    """
    assert box_to_yolo(Box(0, 0, 3, 5), 100, 100) == (0.015, 0.025, 0.03, 0.05)


def test_box_from_yolo_arithmetic() -> None:
    """The inverse, also by hand: cx=0.25 on 1000 wide is a center at 250, and a
    normalized width of 0.1 is 100 pixels, so the left edge is 250 - 50 = 200."""
    assert box_from_yolo(0.25, 0.25, 0.1, 0.1, 1000, 500) == Box(200, 100, 100, 50)
    assert box_from_yolo(0.5, 0.5, 1.0, 1.0, 1280, 800) == Box(0, 0, 1280, 800)


def test_box_to_yolo_clamps_a_box_hanging_off_the_frame() -> None:
    """Normalized coordinates outside 0..1 are not a legal YOLO label."""
    cx, cy, nw, nh = box_to_yolo(Box(-40, -20, 80, 40), 800, 600)
    assert (cx, cy) == (0.0, 0.0)
    assert 0.0 <= nw <= 1.0 and 0.0 <= nh <= 1.0


@pytest.mark.parametrize("size", [(1000, 500), (1280, 800), (1920, 1080), (640, 640)])
def test_boxes_round_trip_exactly(size: tuple[int, int]) -> None:
    """Every hand-written box survives normalize-then-denormalize unchanged.

    The list is chosen for the cases that break naive arithmetic: the origin, a
    single pixel, odd widths and heights, a box flush against the right and bottom
    edges, and the whole frame.
    """
    width, height = size
    boxes = [
        Box(0, 0, 1, 1),
        Box(0, 0, width, height),
        Box(1, 1, 3, 5),
        Box(7, 13, 17, 23),
        Box(width // 2, height // 2, 1, 1),
        Box(width - 1, height - 1, 1, 1),
        Box(width - 37, height - 41, 37, 41),
        Box(width // 3, height // 3, width // 3, height // 3),
        Box(0, height - 2, width, 2),
    ]
    for box in boxes:
        cx, cy, nw, nh = box_to_yolo(box, width, height)
        assert box_from_yolo(cx, cy, nw, nh, width, height) == box, box


def test_normalized_labels_do_not_depend_on_device_scale() -> None:
    """The same box normalized against a logical frame and against its 2x physical
    twin gives the same four numbers - which is why the dataset can mix Retina and
    ordinary captures without a special case anywhere."""
    logical = box_to_yolo(Box(100, 50, 200, 40), 1000, 500)
    retina = box_to_yolo(Box(200, 100, 400, 80), 2000, 1000)
    assert logical == retina


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-5, 100)])
def test_a_non_positive_image_size_is_refused(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        box_to_yolo(Box(0, 0, 1, 1), *bad)
    with pytest.raises(ValueError, match="positive"):
        box_from_yolo(0.5, 0.5, 0.1, 0.1, *bad)


# --------------------------------------------------------------------------------------
# Label files
# --------------------------------------------------------------------------------------


def test_label_line_is_the_expected_text() -> None:
    element = Element(box=Box(200, 100, 100, 50), kind=ElementKind.button)
    assert to_label_line(element, 1000, 500) == "0 0.250000 0.250000 0.100000 0.100000"


def test_label_line_uses_the_element_kind_class_id() -> None:
    element = Element(box=Box(0, 0, 10, 10), kind=ElementKind.checkbox)
    assert to_label_line(element, 100, 100).split()[0] == "2"


def test_label_text_round_trips_a_whole_frame() -> None:
    elements = [
        Element(box=Box(0, 0, 40, 24), kind=ElementKind.button),
        Element(box=Box(311, 97, 17, 17), kind=ElementKind.checkbox),
        Element(box=Box(640, 400, 320, 33), kind=ElementKind.text_field),
        Element(box=Box(1279, 799, 1, 1), kind=ElementKind.icon),
    ]
    text = to_label_text(elements, 1280, 800)
    assert text.count("\n") == 4
    read_back = elements_from_label_text(text, 1280, 800)
    assert [(e.box, e.kind) for e in read_back] == [(e.box, e.kind) for e in elements]


def test_an_empty_frame_is_an_empty_label_file() -> None:
    """A legal YOLO label file with no objects, not a missing one."""
    assert to_label_text([], 800, 600) == ""
    assert elements_from_label_text("", 800, 600) == []


def test_blank_lines_are_ignored() -> None:
    text = "0 0.5 0.5 0.1 0.1\n\n  \n2 0.25 0.25 0.02 0.02\n"
    assert len(elements_from_label_text(text, 800, 600)) == 2


def test_a_label_line_read_back_carries_the_callers_confidence_and_source() -> None:
    element = element_from_label_line(
        "5 0.5 0.5 0.1 0.1", 800, 600, confidence=0.42, source=ElementSource.yolo
    )
    assert element.kind is ElementKind.icon
    assert element.confidence == pytest.approx(0.42)
    assert element.source is ElementSource.yolo
    assert element.text == ""


@pytest.mark.parametrize(
    "bad",
    ["0 0.5 0.5 0.1", "0 0.5 0.5 0.1 0.1 0.1", "button 0.5 0.5 0.1 0.1", "0 a b c d"],
)
def test_a_malformed_label_line_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_label_line(bad)


def test_parse_label_line_returns_the_five_fields() -> None:
    assert parse_label_line("3 0.5 0.25 0.125 0.0625") == (3, 0.5, 0.25, 0.125, 0.0625)


def test_dataset_yaml_lists_every_class_in_id_order() -> None:
    text = dataset_yaml("/tmp/ui-dataset")
    assert "path: /tmp/ui-dataset" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert f"nc: {len(CLASS_NAMES)}" in text
    for index, name in enumerate(CLASS_NAMES):
        assert f"  {index}: {name}\n" in text


def test_label_stem_pairs_an_image_with_its_labels() -> None:
    assert label_stem("shot_003.png") == "shot_003.txt"
    assert label_stem("mail_inbox__1280x800@1x__y0.png") == "mail_inbox__1280x800@1x__y0.txt"


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_recall_counts_a_match_only_above_the_threshold() -> None:
    want = [Element(box=Box(0, 0, 100, 100), kind=ElementKind.button)]
    close = [Element(box=Box(5, 5, 100, 100), kind=ElementKind.button)]
    far = [Element(box=Box(60, 60, 100, 100), kind=ElementKind.button)]
    assert recall_at_iou(want, close) == 1.0
    assert recall_at_iou(want, far) == 0.0


def test_recall_requires_the_right_kind_unless_told_otherwise() -> None:
    want = [Element(box=Box(0, 0, 100, 100), kind=ElementKind.button)]
    wrong_kind = [Element(box=Box(0, 0, 100, 100), kind=ElementKind.checkbox)]
    assert recall_at_iou(want, wrong_kind) == 0.0
    assert recall_at_iou(want, wrong_kind, match_kind=False) == 1.0


def test_one_detection_cannot_satisfy_two_expectations() -> None:
    want = [
        Element(box=Box(0, 0, 100, 100), kind=ElementKind.button),
        Element(box=Box(2, 2, 100, 100), kind=ElementKind.button),
    ]
    detected = [Element(box=Box(1, 1, 100, 100), kind=ElementKind.button)]
    assert recall_at_iou(want, detected) == 0.5


def test_nothing_expected_is_perfect_recall() -> None:
    assert recall_at_iou([], []) == 1.0


# --------------------------------------------------------------------------------------
# The detector, without weights
# --------------------------------------------------------------------------------------


def blank_screenshot(width: int = 1280, height: int = 800, scale: float = 1.0) -> Screenshot:
    """A real PNG of the right physical size, so ``to_array`` works on it."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (round(width * scale), round(height * scale)), "white").save(
        buffer, format="PNG"
    )
    return Screenshot(
        png=buffer.getvalue(),
        width=width,
        height=height,
        scale=scale,
        captured_at=utcnow(),
    )


def test_yolo_detector_satisfies_the_detector_protocol() -> None:
    assert isinstance(YoloDetector(), Detector)


def test_constructing_a_detector_loads_nothing(tmp_path: Path) -> None:
    """Import and construction must stay cheap: a missing weights file is only an
    error for code that actually detects something."""
    detector = YoloDetector(tmp_path / "definitely-absent.pt")
    assert detector.weights_path() == tmp_path / "definitely-absent.pt"


def test_missing_weights_raise_with_somewhere_to_go(tmp_path: Path) -> None:
    """Never an empty list: to an agent that is indistinguishable from a blank
    screen, and it would go looking for the bug on the wrong side of the code."""
    missing = tmp_path / "ui_detector.pt"
    with pytest.raises(PerceptionError) as caught:
        YoloDetector(missing).detect(blank_screenshot())
    message = str(caught.value)
    assert str(missing) in message
    assert "build_ui_dataset.py" in message
    assert "train_detector.py" in message


def test_importing_ultralytics_leaves_pil_alone() -> None:
    """Loading the detector must not change how the rest of the process decodes an
    image.

    Ultralytics patches ``PIL.Image.open`` so a failed decode tries to install a HEIF
    plugin - which runs pip, and raises ``ModuleNotFoundError`` where
    ``Screenshot.to_array`` promises ``PerceptionError``. One import of this module
    would otherwise break an unrelated test and put a package install in a test run.
    """
    import PIL.Image

    from skillweaver.perception.detect_yolo import _import_yolo

    _import_yolo()
    assert not getattr(PIL.Image.open, "__module__", "").startswith("ultralytics")
    with pytest.raises(PerceptionError):
        Screenshot(b"not a png", 1, 1, 1.0, utcnow()).to_array()


def test_the_default_weights_path_sits_under_the_models_directory() -> None:
    assert default_weights_path().name == "ui_detector.pt"
    assert default_weights_path().parent.name == "models"


# -- conversion, against a stub model instead of real weights --------------------------


@dataclass
class _StubBoxes:
    """The shape ultralytics hands back: xyxy in the pixels it was given."""

    rows: list[tuple[float, float, float, float]]
    scores: list[float]
    classes: list[int]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def xyxy(self) -> _StubTensor:
        return _StubTensor(self.rows)

    @property
    def conf(self) -> _StubTensor:
        return _StubTensor(self.scores)

    @property
    def cls(self) -> _StubTensor:
        return _StubTensor(self.classes)


@dataclass
class _StubTensor:
    values: list

    def cpu(self) -> _StubTensor:
        return self

    def tolist(self) -> list:
        return self.values


@dataclass
class _StubResult:
    boxes: _StubBoxes


class _StubModel:
    """Stands in for a loaded YOLO. Records what it was asked, returns fixed boxes."""

    names = dict(enumerate(CLASS_NAMES))

    def __init__(self, boxes: _StubBoxes) -> None:
        self._boxes = boxes
        self.seen_shape: tuple[int, ...] | None = None

    def predict(self, source, **kwargs):  # noqa: ANN001, ANN003 - mirrors ultralytics
        self.seen_shape = source.shape
        return [_StubResult(self._boxes)]


def detector_with(boxes: _StubBoxes) -> tuple[YoloDetector, _StubModel]:
    detector = YoloDetector()
    model = _StubModel(boxes)
    detector._model = model  # noqa: SLF001 - substituting the one thing we cannot ship
    return detector, model


def test_detections_come_back_in_logical_pixels_on_a_retina_frame() -> None:
    """The bug this guards against is a click landing at exactly twice the intended
    coordinates: a 2x screenshot is inferred on at 2x, so every box must be divided
    by ``Screenshot.scale`` before it becomes an ``Element``."""
    boxes = _StubBoxes(
        rows=[(200.0, 100.0, 400.0, 180.0)], scores=[0.9], classes=[class_id(ElementKind.button)]
    )
    detector, model = detector_with(boxes)
    elements = detector.detect(blank_screenshot(1280, 800, scale=2.0))

    assert model.seen_shape == (1600, 2560, 3)  # inference ran on the physical pixels
    assert len(elements) == 1
    assert elements[0].box == Box(100, 50, 100, 40)
    assert elements[0].kind is ElementKind.button
    assert elements[0].source is ElementSource.yolo
    assert elements[0].confidence == pytest.approx(0.9)


def test_detections_at_scale_one_are_unchanged() -> None:
    boxes = _StubBoxes(rows=[(10.0, 20.0, 60.0, 44.0)], scores=[0.5], classes=[0])
    detector, _ = detector_with(boxes)
    assert detector.detect(blank_screenshot(800, 600))[0].box == Box(10, 20, 50, 24)


def test_detections_are_ordered_by_confidence() -> None:
    boxes = _StubBoxes(
        rows=[(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 40.0, 40.0), (50.0, 50.0, 70.0, 70.0)],
        scores=[0.30, 0.95, 0.60],
        classes=[0, 2, 5],
    )
    detector, _ = detector_with(boxes)
    scores = [element.confidence for element in detector.detect(blank_screenshot(800, 600))]
    assert scores == sorted(scores, reverse=True)


def test_a_box_running_off_the_frame_is_clipped_to_the_viewport() -> None:
    """A model may predict past the edge; a click target never may."""
    boxes = _StubBoxes(rows=[(-20.0, -10.0, 60.0, 40.0)], scores=[0.8], classes=[0])
    detector, _ = detector_with(boxes)
    assert detector.detect(blank_screenshot(800, 600))[0].box == Box(0, 0, 60, 40)


def test_a_box_entirely_outside_the_frame_is_dropped() -> None:
    boxes = _StubBoxes(rows=[(900.0, 700.0, 950.0, 750.0)], scores=[0.8], classes=[0])
    detector, _ = detector_with(boxes)
    assert detector.detect(blank_screenshot(800, 600)) == []


def test_nothing_detected_is_an_empty_list_not_an_error() -> None:
    detector, _ = detector_with(_StubBoxes(rows=[], scores=[], classes=[]))
    assert detector.detect(blank_screenshot()) == []


def test_a_class_id_outside_the_map_is_a_perception_error() -> None:
    """Weights trained against a different class map would otherwise report
    confident nonsense."""
    boxes = _StubBoxes(rows=[(0.0, 0.0, 10.0, 10.0)], scores=[0.8], classes=[99])
    detector, _ = detector_with(boxes)
    with pytest.raises(PerceptionError, match="outside"):
        detector.detect(blank_screenshot())


# --------------------------------------------------------------------------------------
# The detector, with weights
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One committed screenshot and the ground truth that was captured with it."""

    name: str
    screenshot: Screenshot
    expected: list[Element]


def load_fixture_frames() -> list[Frame]:
    """Read ``fixtures/frames.json`` and the PNG and label file each entry names.

    The expectations are stored as ordinary YOLO label files, so this also puts the
    label reader through a few thousand real boxes on the way past.
    """
    manifest = json.loads((FIXTURES / "frames.json").read_text(encoding="utf-8"))
    frames: list[Frame] = []
    for entry in manifest["frames"]:
        png = (FIXTURES / entry["image"]).read_bytes()
        screenshot = Screenshot(
            png=png,
            width=entry["width"],
            height=entry["height"],
            scale=entry["scale"],
            captured_at=utcnow(),
        )
        expected = elements_from_label_text(
            (FIXTURES / entry["labels"]).read_text(encoding="utf-8"),
            entry["width"],
            entry["height"],
        )
        frames.append(Frame(entry["image"], screenshot, expected))
    return frames


requires_weights = pytest.mark.skipif(
    not default_weights_path().is_file(),
    reason=(
        f"no detector weights at {default_weights_path()} - they ship at "
        f"data/models/ui_detector.pt, so this means SKILLWEAVER_DATA_DIR points "
        f"somewhere else. Rebuild them with:\n"
        f"  uv run python scripts/build_ui_dataset.py --out data/models/ui-dataset\n"
        f"  uv run python scripts/train_detector.py --data data/models/ui-dataset/data.yaml"
    ),
)


def test_the_fixture_frames_are_the_ones_the_floor_was_measured_on() -> None:
    frames = load_fixture_frames()
    assert [frame.name for frame in frames] == [
        "mail_inbox.png",
        "rec_selected.png",
        "set_dialog.png",
    ]
    assert all(frame.expected for frame in frames)


@requires_weights
def test_detection_meets_the_pinned_recall_floor() -> None:
    """The real gate: on frames at a viewport and device scale the model never
    trained on, it has to find most of what is actually there, at the same kind and
    within IoU 0.5.

    Both floors are what the shipped weights measure with headroom, not numbers
    chosen to make this pass.
    """
    detector = YoloDetector()
    scores: dict[str, float] = {}
    for frame in load_fixture_frames():
        detected = detector.detect(frame.screenshot)
        assert detected, f"{frame.name}: detected nothing at all"
        scores[frame.name] = recall_at_iou(frame.expected, detected, iou=0.5)

    mean = sum(scores.values()) / len(scores)
    assert mean >= MEAN_RECALL_FLOOR, f"mean recall {mean:.3f} below {MEAN_RECALL_FLOOR}: {scores}"
    for name, score in scores.items():
        assert score >= FRAME_RECALL_FLOOR, f"{name} recall {score:.3f} is a blind screen"


@requires_weights
def test_the_controls_the_agent_clicks_are_found_more_reliably_than_prose() -> None:
    """Pooled over every fixture, because a frame with two menus and a frame with
    fifty rows should not weigh the same when the question is "can it see a
    control"."""
    detector = YoloDetector()
    wanted = 0
    found = 0.0
    for frame in load_fixture_frames():
        controls = [element for element in frame.expected if element.kind in CONTROL_KINDS]
        recall = recall_at_iou(controls, detector.detect(frame.screenshot), iou=0.5)
        wanted += len(controls)
        found += recall * len(controls)

    pooled = found / wanted
    assert pooled >= CONTROL_RECALL_FLOOR, f"control recall {pooled:.3f} below the floor"


@requires_weights
def test_detections_stay_inside_the_viewport_and_carry_a_real_confidence() -> None:
    detector = YoloDetector()
    for frame in load_fixture_frames():
        shot = frame.screenshot
        for element in detector.detect(shot):
            assert element.source is ElementSource.yolo
            assert 0.0 < element.confidence <= 1.0
            assert element.box.area > 0
            assert 0 <= element.box.x and element.box.x + element.box.w <= shot.width
            assert 0 <= element.box.y and element.box.y + element.box.h <= shot.height
