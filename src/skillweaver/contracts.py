"""The shared contracts of skillweaver: every value type and every behavior Protocol.

This module is the project's shared surface. Every other module imports from it and
none of them import from each other's internals, so changing anything here is a
coordination decision, not a local edit (see ``AGENTS.md``).

Conventions that hold for everything in this file
--------------------------------------------------

**COORDINATES ARE LOGICAL PIXELS. ALWAYS. EVERYWHERE.**
    Every :class:`Point` and :class:`Box` in this codebase - element boxes, click
    targets, viewports, scroll anchors - is expressed in *logical* pixels (CSS pixels
    in a browser, "points" on macOS), with the origin at the top-left of the
    controller's viewport, x growing right and y growing down.

    A Retina display has 2 physical pixels per logical pixel, so a raw screenshot of
    a 1440x900 logical screen is 2880x1800 physical pixels. Anything that works on
    raw image pixels (a detector, OCR, template matching) MUST convert back to
    logical pixels before building a ``Box`` - divide by :attr:`Screenshot.scale`.
    ``Screenshot.to_array()`` returns a logical-size image by default precisely so
    the obvious code is the correct code. A click that lands at exactly twice the
    intended coordinates is this bug.

**Values are immutable.** Value types are ``@dataclass(frozen=True, slots=True)``.
    Sequences inside them are tuples. Build a modified copy with
    ``dataclasses.replace``. The one deliberate exception is :class:`Spend`.

**Times.** Durations are milliseconds (``*_ms``, ``ms``) as ``float`` unless the
    name says seconds. Timestamps are timezone-aware UTC ``datetime`` - use
    :func:`utcnow`.

**Lookups.** Methods returning a list return it ordered best-first, return an empty
    list when nothing matches, and never return ``None``. Methods documented as
    returning ``X | None`` use ``None`` for "no such thing", not an exception.

**Failures.** Protocol methods raise subclasses of
    :class:`skillweaver.errors.SkillWeaverError`; each docstring names which.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

from skillweaver.errors import BudgetExceeded

if TYPE_CHECKING:
    import numpy as np


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC ``datetime``.

    Use this for every timestamp stored in a contract type, so that timestamps are
    always comparable and never naive.
    """
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# Geometry and observation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    """A position in LOGICAL pixels, relative to the top-left of the viewport.

    ``x`` grows to the right, ``y`` grows downward. Never physical pixels: on a
    Retina display divide raw image coordinates by ``Screenshot.scale`` first.
    """

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle in LOGICAL pixels.

    ``(x, y)`` is the top-left corner, ``w`` and ``h`` are the width and height.
    The rectangle covers ``x <= px < x + w`` and ``y <= py < y + h``. ``w`` and ``h``
    are expected to be non-negative; a box with zero width or height has zero area.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        """The center of the box in logical pixels, rounded down to whole pixels."""
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        """Area in square logical pixels; ``0`` for a degenerate box."""
        return max(self.w, 0) * max(self.h, 0)

    def contains(self, point: Point) -> bool:
        """Whether ``point`` lies inside the box (left/top edges inclusive, right/bottom
        exclusive)."""
        return self.x <= point.x < self.x + self.w and self.y <= point.y < self.y + self.h

    def iou(self, other: Box) -> float:
        """Intersection over union with ``other``, in ``0.0..1.0``.

        ``1.0`` for identical non-degenerate boxes, ``0.0`` when the boxes do not
        overlap or when both have zero area. Never raises.
        """
        ix = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One captured frame of the controller's viewport.

    Attributes:
        png: The encoded PNG bytes, at the capture's native (PHYSICAL) resolution.
        width: Viewport width in LOGICAL pixels.
        height: Viewport height in LOGICAL pixels.
        scale: Physical pixels divided by logical pixels (``2.0`` on a Retina
            display, ``1.0`` for a default Playwright page). The PNG is therefore
            ``round(width * scale)`` by ``round(height * scale)`` physical pixels.
        captured_at: Timezone-aware UTC time of capture.
    """

    png: bytes = field(repr=False)
    width: int
    height: int
    scale: float
    captured_at: datetime

    def to_array(self, *, logical: bool = True) -> np.ndarray:
        """Decode the PNG into an ``H x W x 3`` ``uint8`` RGB numpy array.

        With ``logical=True`` (the default) the image is resized to ``height`` x
        ``width``, so array index ``[y, x]`` IS the logical pixel ``Point(x, y)`` and
        boxes found in the array need no conversion. When ``scale == 1.0`` no
        resize happens.

        With ``logical=False`` the array is at native PHYSICAL resolution (sharper,
        better for OCR). Coordinates found in it MUST be divided by ``scale`` before
        becoming a ``Point`` or ``Box``.

        Raises:
            PerceptionError: if the PNG bytes cannot be decoded.
        """
        import io

        import numpy as np
        from PIL import Image, UnidentifiedImageError

        from skillweaver.errors import PerceptionError

        try:
            image = Image.open(io.BytesIO(self.png)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise PerceptionError(f"screenshot PNG could not be decoded: {exc}") from exc
        if logical and image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


class ElementKind(enum.StrEnum):
    """What sort of UI element something is. ``other`` is the catch-all."""

    button = "button"
    text_field = "text_field"
    checkbox = "checkbox"
    radio = "radio"
    link = "link"
    icon = "icon"
    menu = "menu"
    tab = "tab"
    row = "row"
    image = "image"
    text = "text"
    other = "other"


class ElementSource(enum.StrEnum):
    """Where an :class:`Element` came from.

    ``yolo`` is the visual detector, ``ocr`` the text reader, ``dom`` a ground-truth
    source (offline only, see :class:`GroundTruthSource`), and ``merged`` an element
    fused from more than one source (typically a YOLO box with OCR text).
    """

    yolo = "yolo"
    ocr = "ocr"
    dom = "dom"
    merged = "merged"


@dataclass(frozen=True, slots=True)
class Element:
    """One UI element seen on a screenshot.

    Attributes:
        box: Bounding box in LOGICAL pixels.
        kind: The element's kind.
        text: Visible text or label, ``""`` when there is none or it is unknown.
        confidence: Detection confidence in ``0.0..1.0`` (``1.0`` for ground truth).
        stable_id: An identifier expected to stay the same for the same element
            across observations of the same screen (for example a hash of kind,
            text and coarse position), or ``None`` when no such identity is known.
            It is NOT stable across different screens.
        source: Which producer emitted this element.
    """

    box: Box
    kind: ElementKind
    text: str = ""
    confidence: float = 1.0
    stable_id: str | None = None
    source: ElementSource = ElementSource.merged


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A compact identity for "which screen is this".

    Attributes:
        value: The canonical identity string. Two fingerprints are EQUAL (``==``,
            ``hash``) exactly when their ``value`` is equal, so a ``Fingerprint`` is
            usable as a dict key or graph node id.
        parts: Named sub-hashes the value was derived from (for example ``url``,
            ``layout``, ``text``). Used only by :meth:`similarity`; excluded from
            equality and hashing. Treat it as read-only.
    """

    value: str
    parts: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def similarity(self, other: Fingerprint) -> float:
        """How alike two screens are, in ``0.0..1.0``. Symmetric; never raises.

        ``1.0`` when the ``value`` strings are equal. Otherwise the fraction of part
        names, over the union of both fingerprints' part names, whose sub-hashes are
        equal; ``0.0`` when neither fingerprint has parts.
        """
        if self.value == other.value:
            return 1.0
        names = set(self.parts) | set(other.parts)
        if not names:
            return 0.0
        same = sum(
            1
            for name in names
            if name in self.parts and name in other.parts and self.parts[name] == other.parts[name]
        )
        return same / len(names)


@runtime_checkable
class ElementIndex(Protocol):
    """A queryable view over the elements of ONE observation.

    Return contract for every method: a ``list[Element]`` ordered best-first, an
    EMPTY list when nothing matches, never ``None``, and never an exception for "no
    match". All geometry is in LOGICAL pixels. An index is immutable once built.
    """

    def all(self) -> list[Element]:
        """Every element, in reading order (top-to-bottom, then left-to-right)."""
        ...

    def by_kind(self, kind: ElementKind) -> list[Element]:
        """Elements of exactly ``kind``, in reading order."""
        ...

    def find_text(
        self, query: str, kind: ElementKind | None = None, fuzzy: bool = True
    ) -> list[Element]:
        """Elements whose ``text`` matches ``query``, best match first.

        Matching is case-insensitive. With ``fuzzy=False`` only elements whose text
        equals or contains ``query`` match. With ``fuzzy=True`` near matches (OCR
        typos, partial words) are also returned, ranked below exact ones. ``kind``
        restricts the search to one element kind.
        """
        ...

    def nearest(self, point: Point, kind: ElementKind | None = None) -> list[Element]:
        """Elements ordered by distance from ``point`` to the element's box (``0``
        when the point is inside it), nearest first. ``kind`` restricts the kind."""
        ...

    def containing(self, point: Point) -> list[Element]:
        """Elements whose box contains ``point``, smallest box (most specific) first."""
        ...

    def best(self, description: str) -> list[Element]:
        """Elements matching a free-form description such as ``"blue Submit button"``
        or ``"search field"``, best first. Considers kind words as well as text.
        The first item is the index's best guess; callers should still check it."""
        ...


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the agent knows about the screen at one instant.

    Attributes:
        screenshot: The frame this observation was built from.
        elements: All detected elements, in reading order, boxes in LOGICAL pixels.
        index: Query interface over ``elements`` (excluded from equality).
        fingerprint: Identity of this screen.
        url: Current URL when the controller has one, else ``None``.
        taken_at: Timezone-aware UTC time the observation was completed.
    """

    screenshot: Screenshot
    elements: tuple[Element, ...]
    index: ElementIndex = field(compare=False, repr=False)
    fingerprint: Fingerprint
    url: str | None
    taken_at: datetime


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------

ActionKind = Literal[
    "click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate", "back"
]
"""The ``kind`` tag of every action, as used by ``Controller.supports``."""

MouseButton = Literal["left", "right", "middle"]


@dataclass(frozen=True, slots=True)
class Click:
    """Click at ``point`` (LOGICAL pixels). ``clicks=2`` is a double click."""

    kind: ClassVar[Literal["click"]] = "click"
    point: Point
    button: MouseButton = "left"
    clicks: int = 1


@dataclass(frozen=True, slots=True)
class Move:
    """Move the pointer to ``point`` (LOGICAL pixels) without pressing anything."""

    kind: ClassVar[Literal["move"]] = "move"
    point: Point


@dataclass(frozen=True, slots=True)
class Drag:
    """Press the left button at ``start``, move to ``end``, release. LOGICAL pixels."""

    kind: ClassVar[Literal["drag"]] = "drag"
    start: Point
    end: Point


@dataclass(frozen=True, slots=True)
class TypeText:
    """Type ``text`` literally into whatever currently has keyboard focus.

    Does not click first and does not press Enter; use :class:`Click` and
    :class:`PressKey` for that.
    """

    kind: ClassVar[Literal["type_text"]] = "type_text"
    text: str


@dataclass(frozen=True, slots=True)
class PressKey:
    """Press a key chord. ``keys`` are held together in order, then released.

    Key names follow Playwright's vocabulary: ``"Enter"``, ``"Tab"``, ``"Escape"``,
    ``"Backspace"``, ``"ArrowDown"``, ``"Control"``, ``"Meta"``, ``"Shift"``,
    ``"Alt"``, and single characters such as ``"a"``. ``("Meta", "a")`` is Cmd+A.
    Controllers translate to their own backend's names.
    """

    kind: ClassVar[Literal["press_key"]] = "press_key"
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Scroll:
    """Scroll with the pointer over ``point`` (LOGICAL pixels).

    ``dx`` and ``dy`` are LOGICAL pixels of content movement; positive ``dy`` scrolls
    DOWN (reveals content further down the page), positive ``dx`` scrolls right.
    """

    kind: ClassVar[Literal["scroll"]] = "scroll"
    point: Point
    dx: int = 0
    dy: int = 0


@dataclass(frozen=True, slots=True)
class Wait:
    """Do nothing for ``ms`` milliseconds."""

    kind: ClassVar[Literal["wait"]] = "wait"
    ms: int


@dataclass(frozen=True, slots=True)
class Navigate:
    """Load ``url`` directly.

    OPTIONAL: only some controllers support it (a browser does, a raw desktop
    controller does not). Check ``controller.supports("navigate")`` first; an
    unsupporting controller returns ``ActionResult(ok=False, ...)``.
    """

    kind: ClassVar[Literal["navigate"]] = "navigate"
    url: str


@dataclass(frozen=True, slots=True)
class Back:
    """Go back one entry in the browser's own session history.

    NOT a ``Navigate`` to a remembered address, and deliberately not a mode of one.
    ``Navigate`` takes a URL and always lands on it; this takes no argument and lands
    wherever the history stack says, which is a different thing to record, to replay
    and to refuse. Folding it into ``Navigate`` would leave ``url`` meaningless in one
    mode, and every reader that already trusts ``Navigate.url`` - the dashboard's edge
    labels, the hardcoded-navigation pass in ``skills.refactor`` - would read that
    empty string as an address.

    OPTIONAL, like ``Navigate``: only a controller with session history supports it (a
    browser does, a raw desktop controller does not). Check
    ``controller.supports("back")`` first; an unsupporting controller returns
    ``ActionResult(ok=False, ...)``. A browser with nothing behind the current page
    refuses it the same way, which is why it is only ever OFFERED to a policy that has
    been told this page has somewhere to go back to - see ``DomSnapshot.can_go_back``.
    """

    kind: ClassVar[Literal["back"]] = "back"


Action = Click | Move | Drag | TypeText | PressKey | Scroll | Wait | Navigate | Back
"""The closed set of things a controller can be asked to do. Match on ``.kind`` or
with ``match``/``isinstance``; do not add a member without updating every controller."""

ACTION_TYPES: Mapping[str, type] = {
    cls.kind: cls for cls in (Click, Move, Drag, TypeText, PressKey, Scroll, Wait, Navigate, Back)
}
"""Maps each ``kind`` tag to its action class."""


def action_to_dict(action: Action) -> dict[str, Any]:
    """Serialize an action to a JSON-safe dict, e.g.
    ``{"kind": "click", "point": {"x": 1, "y": 2}, "button": "left", "clicks": 1}``.

    Points become ``{"x", "y"}`` dicts and tuples become lists. The inverse is
    :func:`action_from_dict`.
    """
    data: dict[str, Any] = {"kind": action.kind}
    for f in dataclasses.fields(action):
        value = getattr(action, f.name)
        if isinstance(value, Point):
            value = {"x": value.x, "y": value.y}
        elif isinstance(value, tuple):
            value = list(value)
        data[f.name] = value
    return data


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Rebuild an action from :func:`action_to_dict` output.

    Raises:
        ValueError: if ``kind`` is missing or unknown, or the fields do not fit.
    """
    kind = data.get("kind")
    cls = ACTION_TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ValueError(f"unknown action kind: {kind!r}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name == "kind":
            continue
        if isinstance(value, Mapping):
            value = Point(int(value["x"]), int(value["y"]))
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"bad fields for action {kind!r}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The outcome of ``Controller.perform``.

    ``ok`` means the input was delivered, NOT that the UI did what was hoped - that
    is a :class:`Critic`'s job. ``error`` is ``None`` when ``ok`` and a short
    human-readable reason otherwise. ``elapsed_ms`` is wall-clock milliseconds the
    controller spent performing the action, including any settle wait.
    """

    ok: bool
    error: str | None = None
    elapsed_ms: float = 0.0


# --------------------------------------------------------------------------------------
# Controllers and perception
# --------------------------------------------------------------------------------------


@runtime_checkable
class Controller(Protocol):
    """Eyes and hands on one screen: a browser page or a real desktop.

    All coordinates in and out are LOGICAL pixels relative to the top-left of
    :meth:`viewport`. A controller is NOT thread-safe; use it from one thread.
    """

    def capture(self) -> Screenshot:
        """Grab the current frame.

        ``Screenshot.width``/``height`` equal the viewport's logical size and
        ``scale`` reports the physical-to-logical ratio of the PNG.

        Raises:
            ControllerError: if the screen cannot be captured or the controller is
                closed.
        """
        ...

    def perform(self, action: Action) -> ActionResult:
        """Execute one action and wait for the UI to settle.

        Failures to deliver the action (unsupported kind, point outside the
        viewport, backend error) are REPORTED as ``ActionResult(ok=False, error=...)``
        and not raised, so an exploring agent can carry on.

        Raises:
            ControllerError: only when the controller itself is unusable (closed or
                crashed).
        """
        ...

    def viewport(self) -> Box:
        """The controllable area in LOGICAL pixels. ``x`` and ``y`` are ``0`` for a
        browser page; a desktop controller restricted to a window may report an
        offset, but action coordinates are still relative to this box's top-left."""
        ...

    def supports(self, action_kind: ActionKind) -> bool:
        """Whether this controller can perform actions of ``action_kind``.
        ``"navigate"`` and ``"back"`` are the kinds commonly unsupported: both need a
        session history, which only a browser has."""
        ...

    def url(self) -> str | None:
        """The current page URL, or ``None`` for a controller with no notion of one
        (a desktop). This is what a user could read from the address bar; it is not
        ground truth about page contents."""
        ...

    def describe(self) -> str:
        """One human-readable line for logs and prompts, such as
        ``"playwright chromium 1280x800 @1x"``."""
        ...

    def close(self) -> None:
        """Release the browser or OS resources. Idempotent; never raises."""
        ...


@runtime_checkable
class GroundTruthSource(Protocol):
    """Perfect knowledge of the screen, read from the DOM or accessibility tree.

    THIS IS AN OFFLINE TEACHER ONLY. It exists to label detector training data and
    to score evaluations. The agent's action path - perceiver, explorer, planner,
    skill runner, skill code - MUST NEVER call it: the whole point of the project is
    an agent that works from pixels. Code that needs it takes it as an explicit
    argument so the dependency is visible.
    """

    def elements(self) -> list[Element]:
        """Every interactable or visible element, boxes in LOGICAL pixels,
        ``source=ElementSource.dom``, ``confidence=1.0``. Empty list if none.

        Raises:
            ControllerError: if the underlying page or tree cannot be read.
        """
        ...

    def url(self) -> str:
        """The exact current URL (``""`` when there is none)."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Finds UI elements in pixels (YOLO). Does not read text."""

    def detect(self, screenshot: Screenshot) -> list[Element]:
        """Return detected elements with boxes in LOGICAL pixels (divide raw image
        coordinates by ``screenshot.scale``), ``source=ElementSource.yolo``, highest
        confidence first. Empty list when nothing is found.

        Raises:
            PerceptionError: if the model cannot be loaded or inference fails.
        """
        ...


@runtime_checkable
class TextReader(Protocol):
    """Reads text in pixels (OCR)."""

    def read(self, screenshot: Screenshot) -> list[Element]:
        """Return one ``ElementKind.text`` element per recognized line or word, with
        ``text`` filled, boxes in LOGICAL pixels, ``source=ElementSource.ocr``, in
        reading order. Empty list when no text is found.

        Raises:
            PerceptionError: if the OCR engine cannot be loaded or fails.
        """
        ...


@runtime_checkable
class Fingerprinter(Protocol):
    """Turns a screen into a :class:`Fingerprint`."""

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        """Compute the identity of a screen. Deterministic: the same inputs always
        give an equal fingerprint. Should be robust to cosmetic change (a blinking
        caret, a clock) and sensitive to structural change (a dialog opening).

        Raises:
            PerceptionError: if the screenshot cannot be processed.
        """
        ...


@runtime_checkable
class Perceiver(Protocol):
    """Composes capture, detect, read, merge, index and fingerprint into one call."""

    def observe(self, controller: Controller) -> Observation:
        """Capture the controller's screen and return a full :class:`Observation`.

        Reads only ``controller.capture()`` and ``controller.url()``; never performs
        an action and never touches a :class:`GroundTruthSource`.

        Raises:
            ControllerError: if capture fails.
            PerceptionError: if detection, OCR or fingerprinting fails.
        """
        ...


# --------------------------------------------------------------------------------------
# Memory: skills
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a skill came from.

    ``trajectory_id`` is the ``Trajectory.run_id`` it was synthesized from,
    ``task_text`` the task that run was solving, ``model`` the ``LLMClient.name()``
    that wrote the code, ``created_at`` a UTC timestamp.
    """

    trajectory_id: str
    task_text: str
    model: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SkillStats:
    """Running record of how a skill performs.

    ``runs`` counts every recorded execution and ``successes`` the ones that
    succeeded. ``mean_ms`` is the mean wall-clock milliseconds of SUCCESSFUL runs
    (``0.0`` before the first success). ``last_ok_at`` is the UTC time of the most
    recent success, or ``None``.
    """

    runs: int = 0
    successes: int = 0
    mean_ms: float = 0.0
    last_ok_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Skill:
    """A reusable, versioned piece of automation stored as Python source.

    Attributes:
        name: Snake-case identifier, unique within ``domain`` (``"search_invoice"``).
        domain: The site or app it belongs to (``"example.com"``, ``"desktop:finder"``).
        summary: One line for retrieval and planning prompts.
        docstring: Full description: what it does, its parameters, its end state.
        params: Parameter name to a JSON-schema-like description
            (``{"query": {"type": "string"}}``). Treat as read-only.
        code: Python source defining ``def run(ctx, **params)``. It may touch the
            world ONLY through ``ctx`` (see :class:`SkillContext`).
        requires: Names of other skills in the same domain that ``code`` calls.
        precondition: The screen the skill expects to start on, or ``None`` when it
            can start anywhere.
        verifier_code: Python source defining ``def verify(ctx, result) -> bool``
            that checks the skill achieved its end state, or ``None``.
        provenance: Where the skill came from.
        version: ``0`` until stored; ``SkillStore.put`` assigns ``1, 2, 3, ...``.
        stats: Performance record.
        demoted_reason: ``None`` for a healthy skill; the reason string once
            ``SkillStore.demote`` has retired it from retrieval.
    """

    name: str
    domain: str
    summary: str
    docstring: str
    params: Mapping[str, Any] = field(hash=False)
    code: str
    requires: tuple[str, ...]
    precondition: Fingerprint | None
    verifier_code: str | None
    provenance: Provenance
    version: int = 0
    stats: SkillStats = field(default_factory=SkillStats)
    demoted_reason: str | None = None


@runtime_checkable
class SkillStore(Protocol):
    """Durable, versioned storage of skills, keyed by ``(name, domain)``."""

    def put(self, skill: Skill) -> Skill:
        """Store ``skill`` as the NEXT version of ``(name, domain)`` and return the
        stored copy. The incoming ``version`` is ignored: the first ``put`` yields
        version ``1`` and each later one increments it. Older versions are kept.
        Does not run admission checks; that happens before ``put``."""
        ...

    def get(self, name: str, domain: str, version: int | None = None) -> Skill:
        """Return one skill; ``version=None`` means the latest. Demoted skills are
        still returned (check ``demoted_reason``).

        Raises:
            SkillNotFound: if the name, domain or version does not exist.
        """
        ...

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[Skill]:
        """The LATEST version of every skill, optionally restricted to one domain,
        sorted by ``(domain, name)``. Demoted skills are omitted unless asked for.
        Empty list when there are none."""
        ...

    def record_run(self, name: str, domain: str, ok: bool, ms: float) -> Skill:
        """Fold one execution into the latest version's :class:`SkillStats` and
        return the updated skill. ``ms`` is wall-clock milliseconds.

        Raises:
            SkillNotFound: if the skill does not exist.
        """
        ...

    def demote(self, name: str, domain: str, reason: str) -> Skill:
        """Retire the latest version from retrieval by setting ``demoted_reason``,
        and return the updated skill. A later ``put`` of a fixed version is healthy
        again.

        Raises:
            SkillNotFound: if the skill does not exist.
        """
        ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieval hit: the skill, a relevance ``score`` in ``0.0..1.0`` (higher is
    better) and ``why`` - a short human-readable explanation for logs and prompts."""

    skill: Skill
    score: float
    why: str = ""


@runtime_checkable
class SkillRetriever(Protocol):
    """Finds stored skills relevant to a task."""

    def search(self, task: str, domain: str | None = None, k: int = 5) -> list[Candidate]:
        """At most ``k`` candidates for the task text, best score first. ``domain``
        restricts the search. Demoted skills are never returned. Empty list when
        nothing is relevant.

        Raises:
            ProviderError: if an embedding backend fails.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors for retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input text, in order, all of the same length, each
        L2-normalized so a dot product is cosine similarity. Deterministic for a
        given text. Empty input gives an empty list.

        Raises:
            ProviderError: if the embedding backend fails.
        """
        ...


# --------------------------------------------------------------------------------------
# Memory: site graph
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UIState:
    """A node of the site graph: one recognizable screen of one site or app.

    ``fingerprint`` is the node id. ``label`` is a short human name (``"inbox"``).
    ``url_pattern`` is a URL or glob the state was seen at, or ``None``.
    ``first_seen`` is UTC. ``thumbnail`` is small PNG bytes for the dashboard, or
    ``None``.
    """

    fingerprint: Fingerprint
    domain: str
    label: str = ""
    url_pattern: str | None = None
    first_seen: datetime = field(default_factory=utcnow)
    thumbnail: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class Transition:
    """An edge of the site graph: performing ``actions`` on ``src`` leads to ``dst``.

    An edge is identified by ``(src, dst, actions)``. ``attempts`` and ``successes``
    count observations; ``mean_ms`` is the mean wall-clock milliseconds of
    SUCCESSFUL traversals (``0.0`` before the first); ``last_verified`` is the UTC
    time of the latest success, or ``None``.
    """

    src: Fingerprint
    dst: Fingerprint
    actions: tuple[Action, ...]
    attempts: int = 0
    successes: int = 0
    mean_ms: float = 0.0
    last_verified: datetime | None = None


@dataclass(frozen=True, slots=True)
class Route:
    """A path through the site graph.

    ``steps`` is the flattened action sequence to perform, ``edges`` the
    transitions it follows in order, and ``cost`` the expected total milliseconds
    (each edge's ``mean_ms`` divided by its success rate). A route from a state to
    itself has no steps, no edges and cost ``0.0``.
    """

    steps: tuple[Action, ...]
    cost: float
    edges: tuple[Transition, ...]


@runtime_checkable
class GraphView(Protocol):
    """The read-only part of a :class:`SiteGraph`; this is what skill code sees."""

    def route(self, src_fp: Fingerprint, dst_fp: Fingerprint) -> Route | None:
        """The lowest-cost known route, or ``None`` when no path is known. Only
        edges with at least one success are used. Matching is exact on
        ``Fingerprint.value``; never raises for unknown fingerprints."""
        ...

    def neighbors(self, fp: Fingerprint) -> list[Transition]:
        """Outgoing edges of ``fp``, most reliable first. Empty list when the state
        is unknown or has no edges."""
        ...

    def states(self, domain: str) -> list[UIState]:
        """Every known state of ``domain``, oldest first. Empty list when none."""
        ...


@runtime_checkable
class SiteGraph(GraphView, Protocol):
    """Per-domain memory of screens and how to move between them."""

    def upsert_state(self, state: UIState) -> UIState:
        """Add a state, or update the label, URL pattern and thumbnail of a known
        one (keyed by ``state.fingerprint``). The original ``first_seen`` is kept.
        Returns the stored state."""
        ...

    def observe_transition(
        self,
        src: Fingerprint,
        actions: Sequence[Action],
        dst: Fingerprint,
        ok: bool,
        ms: float,
    ) -> Transition:
        """Record one attempt at the edge ``(src, dst, actions)`` and return the
        updated edge. ``dst`` is the state the actions were expected to reach; ``ok``
        says whether it was actually reached; ``ms`` is the wall-clock milliseconds
        the attempt took. Unknown states are added implicitly under ``src``'s domain
        when known, else ``""``."""
        ...

    def save(self) -> None:
        """Persist every loaded domain. A no-op for in-memory graphs.

        Raises:
            SkillWeaverError: if the data directory cannot be written.
        """
        ...

    def load(self, domain: str) -> None:
        """Load one domain's graph from storage, replacing what is in memory for it.
        A domain with nothing stored loads as empty; never raises for "missing"."""
        ...


# --------------------------------------------------------------------------------------
# Reasoning: LLM, critic, budget
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and money accounting for LLM calls. Add two with ``+``.

    ``calls`` is the number of provider requests; ``cost_usd`` is US dollars.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.calls + other.calls,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool offered to the model: its ``name``, a ``description``, and
    ``input_schema`` as a JSON Schema object. Treat the schema as read-only."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(hash=False)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asking for a tool: ``name``, JSON-like ``args``, and the provider's
    call ``id`` (``""`` when the provider has none) used to match the result."""

    name: str
    args: Mapping[str, Any] = field(hash=False)
    id: str = ""


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One turn of a conversation, provider-neutral.

    Attributes:
        role: ``"user"``, ``"assistant"``, or ``"tool"`` for a tool result.
        text: The text content (``""`` allowed). For ``"tool"`` it is the result.
        images: PNG bytes attached to the turn, in order.
        tool_calls: For an ``"assistant"`` turn being replayed, the calls it made.
        tool_call_id: For a ``"tool"`` turn, the ``ToolCall.id`` it answers.
    """

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    images: tuple[bytes, ...] = field(default=(), repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One model reply: ``text`` (``""`` when it only called tools), ``tool_calls``
    in order, the ``usage`` of this single call, and the provider's ``stop_reason``
    normalized to ``"end"``, ``"tool_use"``, ``"max_tokens"`` or ``"other"``."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end"


@runtime_checkable
class LLMClient(Protocol):
    """A chat model behind a provider-neutral interface (Claude, Gemini, a fake)."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send the whole conversation and return one reply. Stateless: nothing is
        remembered between calls except the usage total. Blocking.

        ``max_tokens`` caps the reply length in tokens. ``temperature=None`` means
        the provider default; an adapter whose model rejects sampling parameters
        (current Claude models do) MUST silently drop a non-``None`` value rather
        than fail.

        Raises:
            ProviderError: on any network, auth, rate-limit or malformed-reply
                failure, after the client's own retries are exhausted.
        """
        ...

    def total_usage(self) -> Usage:
        """The sum of the usage of every ``complete`` call made on this client."""
        ...

    def name(self) -> str:
        """The model identifier, such as ``"claude-opus-5"``; recorded in provenance."""
        ...


@dataclass(frozen=True, slots=True)
class Verdict:
    """A judgment on whether a step or a task achieved its goal.

    ``confidence`` is ``0.0..1.0``. ``source`` says who decided: ``"programmatic"``
    (a fingerprint match, a verifier, a rule - free and trustworthy) or ``"model"``
    (an LLM looked at the screens).
    """

    ok: bool
    reason: str = ""
    confidence: float = 1.0
    source: Literal["programmatic", "model"] = "programmatic"


@runtime_checkable
class Critic(Protocol):
    """Decides whether something worked by comparing two observations."""

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> Verdict:
        """Judge whether ``goal`` was achieved going from ``before`` to ``after``.
        ``expectation`` optionally describes what ``after`` should look like.
        Implementations try programmatic checks first and consult a model only when
        those are inconclusive.

        Raises:
            ProviderError: if a model was needed and the call failed.
        """
        ...


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one run. ``max_seconds`` is wall-clock seconds and
    ``max_usd`` is US dollars of LLM spend."""

    max_steps: int = 40
    max_seconds: float = 300.0
    max_usd: float = 2.0
    max_llm_calls: int = 60


@dataclass(slots=True)
class Spend:
    """MUTABLE running total charged against a :class:`Budget`.

    The one non-frozen type in this module. Call :meth:`check` BEFORE each unit of
    work: it raises once any limit has been reached, meaning nothing remains.
    ``seconds`` holds explicitly added time; after :meth:`start`, wall-clock time
    since then is counted too (see :meth:`elapsed_seconds`). Not thread-safe.
    """

    budget: Budget = field(default_factory=Budget)
    steps: int = 0
    seconds: float = 0.0
    usd: float = 0.0
    llm_calls: int = 0
    _started: float | None = field(default=None, repr=False)

    def start(self) -> Spend:
        """Start the wall clock (idempotent). Returns ``self`` for chaining."""
        if self._started is None:
            self._started = time.monotonic()
        return self

    def elapsed_seconds(self) -> float:
        """``seconds`` plus wall-clock seconds since :meth:`start`, if started."""
        running = 0.0 if self._started is None else time.monotonic() - self._started
        return self.seconds + running

    def add_step(self, n: int = 1) -> None:
        """Charge ``n`` agent steps (one step is one performed action)."""
        self.steps += n

    def add_usage(self, usage: Usage) -> None:
        """Charge the ``calls`` and ``cost_usd`` of one or more LLM calls."""
        self.llm_calls += usage.calls
        self.usd += usage.cost_usd

    def check(self) -> None:
        """Raise if any limit has been reached; otherwise return ``None``.

        Raises:
            BudgetExceeded: naming the exhausted limit, when ``steps >= max_steps``,
                ``elapsed_seconds() >= max_seconds``, ``usd >= max_usd`` or
                ``llm_calls >= max_llm_calls``.
        """
        b = self.budget
        elapsed = self.elapsed_seconds()
        if self.steps >= b.max_steps:
            raise BudgetExceeded(f"max_steps reached: {self.steps}/{b.max_steps}")
        if elapsed >= b.max_seconds:
            raise BudgetExceeded(f"max_seconds reached: {elapsed:.1f}/{b.max_seconds:.1f}")
        if self.usd >= b.max_usd:
            raise BudgetExceeded(f"max_usd reached: {self.usd:.4f}/{b.max_usd:.4f}")
        if self.llm_calls >= b.max_llm_calls:
            raise BudgetExceeded(f"max_llm_calls reached: {self.llm_calls}/{b.max_llm_calls}")


# --------------------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One action of a run with the screens around it.

    ``index`` counts from ``0``. ``before`` and ``after`` are the observations either
    side of ``action``; ``result`` is what the controller reported; ``verdict`` is a
    critic's judgment of this step or ``None`` if it was not judged; ``note`` is
    free text (the agent's stated reason, typically).
    """

    index: int
    action: Action
    before: Observation
    after: Observation
    result: ActionResult
    verdict: Verdict | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The full record of one run: the raw material skills are synthesized from.

    ``run_id`` is unique; ``task`` is the task text; ``ok`` is whether the run
    achieved the task; timestamps are UTC; ``note`` is the closing remark given to
    ``TrajectoryRecorder.finish``.
    """

    run_id: str
    task: str
    domain: str
    steps: tuple[TrajectoryStep, ...]
    ok: bool
    started_at: datetime
    finished_at: datetime
    note: str = ""


@runtime_checkable
class TrajectoryRecorder(Protocol):
    """Builds one :class:`Trajectory` at a time. Call ``start``, then ``step`` per
    action, then ``finish``."""

    def start(self, task: str, domain: str) -> str:
        """Begin a new run and return its fresh unique ``run_id``.

        Raises:
            SkillWeaverError: if a run is already in progress.
        """
        ...

    def step(
        self,
        action: Action,
        before: Observation,
        after: Observation,
        result: ActionResult,
        verdict: Verdict | None = None,
        note: str = "",
    ) -> TrajectoryStep:
        """Append one step (its ``index`` is assigned here) and return it.

        Raises:
            SkillWeaverError: if no run is in progress.
        """
        ...

    def finish(self, ok: bool, note: str = "") -> Trajectory:
        """Close the run and return the finished trajectory. Does not persist it;
        pass it to a :class:`TrajectoryStore`.

        Raises:
            SkillWeaverError: if no run is in progress.
        """
        ...


@runtime_checkable
class TrajectoryStore(Protocol):
    """Durable storage of finished trajectories."""

    def save(self, trajectory: Trajectory) -> None:
        """Persist a trajectory, replacing any with the same ``run_id``.

        Raises:
            SkillWeaverError: if the data directory cannot be written.
        """
        ...

    def load(self, run_id: str) -> Trajectory:
        """Return one trajectory.

        Raises:
            SkillWeaverError: if ``run_id`` is unknown or its data is unreadable.
        """
        ...

    def list(self) -> list[str]:
        """Every stored ``run_id``, oldest first. Empty list when there are none."""
        ...


# --------------------------------------------------------------------------------------
# Tasks, skills at run time, and the agent
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A task for the agent.

    ``text`` is the natural-language instruction, ``domain`` the site or app it
    concerns, ``target`` which controller family runs it (``"browser"`` or
    ``"desktop"``), and ``params`` free-form task parameters (a start URL under
    ``"start_url"``, values to fill in). Treat ``params`` as read-only.
    """

    text: str
    domain: str
    target: Literal["browser", "desktop"] = "browser"
    params: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """The result of running one task end to end.

    ``trajectory`` is what happened, ``verdict`` the final judgment, ``spend`` the
    resources used, ``skill_used`` the name of the stored skill that carried the run
    (``None`` when it was solved by exploration), ``note`` a closing remark.
    """

    ok: bool
    trajectory: Trajectory
    verdict: Verdict
    spend: Spend
    skill_used: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class SkillResult:
    """The result of executing one skill.

    ``value`` is whatever ``run`` returned (``None`` on failure); ``steps`` is the
    number of controller actions performed, including by nested skills; ``ms`` is
    wall-clock milliseconds; ``error`` is ``None`` when ``ok``, else a short reason;
    ``trace`` is the ordered ``ctx.log`` lines and action descriptions.
    """

    ok: bool
    value: Any = None
    steps: int = 0
    ms: float = 0.0
    error: str | None = None
    trace: tuple[str, ...] = ()


@runtime_checkable
class ActionSurface(Protocol):
    """The narrowed, action-only view of a controller that skill code receives as
    ``ctx.ctl``. No ``capture``, no ``close``, no ground truth.

    Unlike ``Controller.perform``, every method here RAISES ``ControllerError`` when
    the action could not be delivered, so skill code need not check results. All
    coordinates are LOGICAL pixels. A target may be a :class:`Point`, a
    :class:`Box` (its center is used) or an :class:`Element` (its box's center).
    """

    def perform(self, action: Action) -> ActionResult:
        """Perform any action. Raises ``ControllerError`` if it is not delivered."""
        ...

    def supports(self, action_kind: ActionKind) -> bool:
        """Whether the underlying controller supports ``action_kind``."""
        ...

    def click(
        self, target: Point | Box | Element, button: MouseButton = "left", clicks: int = 1
    ) -> ActionResult:
        """Click the target."""
        ...

    def type_text(self, text: str) -> ActionResult:
        """Type into whatever has focus."""
        ...

    def press(self, *keys: str) -> ActionResult:
        """Press a key chord, e.g. ``press("Enter")`` or ``press("Meta", "a")``."""
        ...

    def scroll(self, target: Point | Box | Element, dx: int = 0, dy: int = 0) -> ActionResult:
        """Scroll over the target; positive ``dy`` scrolls down."""
        ...

    def wait(self, ms: int) -> ActionResult:
        """Pause for ``ms`` milliseconds."""
        ...


@runtime_checkable
class SkillContext(Protocol):
    """The ONLY object skill code receives: ``def run(ctx, **params)``.

    This is the complete published surface. Nothing else - no imports, no files, no
    network, no controller internals, no ground truth - is reachable from skill
    code, and the sandbox raises ``SandboxViolation`` on any attempt.
    """

    @property
    def ctl(self) -> ActionSurface:
        """Hands: the action-only view of the controller."""
        ...

    @property
    def see(self) -> ElementIndex:
        """Eyes: an index of the screen AS IT IS NOW. Implementations re-observe
        lazily after any action performed through ``ctl``, so never cache it across
        actions. Raises ``PerceptionError`` if observing fails."""
        ...

    @property
    def graph(self) -> GraphView:
        """Read-only view of the current domain's site graph."""
        ...

    def call(self, name: str, **kwargs: Any) -> Any:
        """Run another skill of the same domain and return its value.

        Raises:
            SkillNotFound: if no such skill exists.
            ExpectationFailed, ControllerError: propagated from the callee.
        """
        ...

    def expect(self, condition: bool, why: str) -> None:
        """Assert something about the screen. Raises ``ExpectationFailed(why)`` when
        ``condition`` is false, which fails the skill cleanly."""
        ...

    def log(self, msg: str) -> None:
        """Append a line to the run's trace. Never raises."""
        ...


@runtime_checkable
class SkillRunner(Protocol):
    """Executes stored skill code inside the sandbox."""

    def run(self, skill: Skill, args: Mapping[str, Any], ctx: SkillContext) -> SkillResult:
        """Execute ``skill.code`` with ``args`` and, when present, its verifier.

        Skill failures of every sort (exception, failed expectation, failed
        verifier, sandbox violation) are REPORTED as ``SkillResult(ok=False,
        error=...)`` and not raised.

        Raises:
            BudgetExceeded: the one exception allowed to escape, so a run stops.
        """
        ...


@dataclass(frozen=True, slots=True)
class SkillCall:
    """A plan step that invokes a stored skill with ``args``."""

    name: str
    domain: str
    args: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class Plan:
    """A model-free way to do a task from what is already known.

    ``steps`` are performed in order and are either a :class:`SkillCall` or a raw
    :data:`Action` (typically site-graph route steps between skills).
    ``skills_used`` names the skills involved; ``estimated_ms`` is the expected
    wall-clock milliseconds from recorded stats.
    """

    steps: tuple[SkillCall | Action, ...]
    skills_used: tuple[str, ...] = ()
    estimated_ms: float = 0.0


@runtime_checkable
class Explorer(Protocol):
    """The slow path: solve a task by trial and error with a computer-use model."""

    def explore(self, task: TaskSpec, controller: Controller, budget: Budget) -> RunOutcome:
        """Attempt the task within ``budget`` and return what happened. Running out
        of budget or failing the task gives ``RunOutcome(ok=False, ...)`` with the
        partial trajectory; neither is raised.

        Raises:
            ControllerError: if the controller breaks mid-run.
        """
        ...


@runtime_checkable
class Planner(Protocol):
    """The fast path: compose stored skills and known routes, without exploring."""

    def plan(self, task: TaskSpec, observation: Observation) -> Plan | None:
        """A plan for the task from the current screen, or ``None`` when the library
        and graph do not cover it (the caller then falls back to an
        :class:`Explorer`)."""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """Turns a successful trajectory into a reusable skill."""

    def synthesize(self, trajectory: Trajectory) -> Skill | None:
        """Write a skill (``version=0``, not yet stored) from the trajectory, or
        ``None`` when it is not worth keeping (failed run, trivial, no model
        answer). Does not admit or store the skill.

        Raises:
            ProviderError: if the model call fails.
        """
        ...
