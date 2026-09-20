"""Every value type and behavior Protocol the project shares. Changing this file is a
coordination decision, not a local edit (see ``AGENTS.md``).

Conventions that hold throughout:

* **Coordinates are LOGICAL pixels, always** - CSS pixels in a browser, points on macOS,
  origin at the viewport's top-left. Anything working on raw image pixels (a detector,
  OCR) must divide by ``Screenshot.scale`` before building a ``Box``; a click landing at
  exactly twice the intended coordinates is that bug. ``Screenshot.to_array()`` returns a
  logical-size image by default so the obvious code is the correct code.
* **Values are immutable** frozen slotted dataclasses holding tuples; ``Spend`` is the one
  deliberate exception.
* **Times** are milliseconds as ``float`` unless the name says seconds; timestamps are
  timezone-aware UTC from ``utcnow``.
* **Lookups** return a best-first list, empty when nothing matches, never ``None``; a
  documented ``X | None`` uses ``None`` for "no such thing" rather than raising.
* **Failures** are ``SkillWeaverError`` subclasses, named per docstring.
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
    """Now, as a timezone-aware UTC ``datetime`` - use for every stored timestamp, so
    none is ever naive."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Point:
    """A position in LOGICAL pixels from the viewport's top-left; ``y`` grows downward."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned rectangle in LOGICAL pixels, covering ``x <= px < x + w`` and
    ``y <= py < y + h`` from its top-left ``(x, y)``."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        """The centre, rounded down to whole pixels."""
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        """Area in square logical pixels; ``0`` for a degenerate box."""
        return max(self.w, 0) * max(self.h, 0)

    def contains(self, point: Point) -> bool:
        """Left/top edges inclusive, right/bottom exclusive."""
        return self.x <= point.x < self.x + self.w and self.y <= point.y < self.y + self.h

    def iou(self, other: Box) -> float:
        """Intersection over union, ``0.0..1.0``; ``0.0`` for no overlap or zero area."""
        ix = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One captured frame: ``png`` at PHYSICAL resolution, ``width``/``height`` LOGICAL,
    and ``scale`` the ratio between them (``2.0`` on Retina)."""

    png: bytes = field(repr=False)
    width: int
    height: int
    scale: float
    captured_at: datetime

    def to_array(self, *, logical: bool = True) -> np.ndarray:
        """Decode to an ``H x W x 3`` ``uint8`` RGB array, raising ``PerceptionError``.

        ``logical=True`` resizes so index ``[y, x]`` IS ``Point(x, y)`` and no conversion
        is needed. ``logical=False`` keeps native PHYSICAL resolution - sharper for OCR,
        and every coordinate out of it must be divided by ``scale``.
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
    """The producer of an ``Element``; ``merged`` is fused from several (typically a YOLO
    box carrying OCR text)."""

    yolo = "yolo"
    ocr = "ocr"
    dom = "dom"
    merged = "merged"


@dataclass(frozen=True, slots=True)
class Element:
    """One UI element seen on a screenshot; ``box`` is LOGICAL pixels.

    ``stable_id`` holds across observations of the SAME screen and is not stable across
    different ones; ``None`` when no such identity is known.
    """

    box: Box
    kind: ElementKind
    text: str = ""
    confidence: float = 1.0
    stable_id: str | None = None
    source: ElementSource = ElementSource.merged


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """A compact identity for "which screen is this"; equal exactly when ``value`` is, so
    it serves as a dict key or graph node id. ``parts`` are the named sub-hashes it came
    from, read by ``similarity`` alone and excluded from equality."""

    value: str
    parts: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False)

    def similarity(self, other: Fingerprint) -> float:
        """``0.0..1.0``, symmetric: ``1.0`` on equal ``value``, else the fraction of the
        union of part names whose sub-hashes agree, and ``0.0`` when neither has parts."""
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
    """A queryable view over ONE observation's elements, immutable once built. Every
    method returns a best-first list, empty when nothing matches, and never raises for
    "no match"."""

    def all(self) -> list[Element]:
        """Every element, in reading order (top-to-bottom, then left-to-right)."""
        ...

    def by_kind(self, kind: ElementKind) -> list[Element]:
        """Elements of exactly ``kind``, in reading order."""
        ...

    def find_text(
        self, query: str, kind: ElementKind | None = None, fuzzy: bool = True
    ) -> list[Element]:
        """Elements whose ``text`` matches, best first, case-insensitively. ``fuzzy=False``
        is equality or containment only; ``fuzzy=True`` also admits OCR typos and partial
        words, ranked below exact matches."""
        ...

    def nearest(self, point: Point, kind: ElementKind | None = None) -> list[Element]:
        """Nearest first by distance to the element's box, ``0`` when ``point`` is inside it."""
        ...

    def containing(self, point: Point) -> list[Element]:
        """Elements whose box contains ``point``, smallest box (most specific) first."""
        ...

    def best(self, description: str) -> list[Element]:
        """Elements matching a free-form description (``"blue Submit button"``), best first,
        reading kind words as well as text. A RANKING: it has a winner even when nothing
        fits, so the caller must still check."""
        ...


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the agent knows about the screen at one instant. ``elements`` are in
    reading order with LOGICAL boxes; ``index`` queries them and is out of equality."""

    screenshot: Screenshot
    elements: tuple[Element, ...]
    index: ElementIndex = field(compare=False, repr=False)
    fingerprint: Fingerprint
    url: str | None
    taken_at: datetime


ActionKind = Literal[
    "click", "move", "drag", "type_text", "press_key", "scroll", "wait", "navigate", "back"
]
"""The ``kind`` tag of every action, as ``Controller.supports`` takes it."""

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
    """Type ``text`` literally into whatever has focus; does not click first or press Enter."""

    kind: ClassVar[Literal["type_text"]] = "type_text"
    text: str


@dataclass(frozen=True, slots=True)
class PressKey:
    """A key chord: ``keys`` held together in order, then released. Names follow
    Playwright's vocabulary (``"Enter"``, ``"Meta"``, ``"ArrowDown"``, ``"a"``) and each
    controller translates to its own backend's."""

    kind: ClassVar[Literal["press_key"]] = "press_key"
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Scroll:
    """Scroll with the pointer over ``point``; positive ``dy`` scrolls DOWN, positive
    ``dx`` right, both in LOGICAL pixels of content movement."""

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
    """Load ``url`` directly. OPTIONAL: check ``supports("navigate")``, since a desktop
    controller returns ``ActionResult(ok=False)``."""

    kind: ClassVar[Literal["navigate"]] = "navigate"
    url: str


@dataclass(frozen=True, slots=True)
class Back:
    """Go back one entry in the browser's session history.

    Deliberately not a mode of ``Navigate``: it takes no argument, and folding it in would
    leave ``url`` empty for every reader that already trusts it as an address (the
    dashboard's edge labels, ``skills.refactor``'s hardcoded-navigation pass).

    OPTIONAL: needs session history, so check ``supports("back")``. A browser with nothing
    behind it refuses too, which is why this is only OFFERED when ``DomSnapshot.can_go_back``
    says the page has somewhere to go.
    """

    kind: ClassVar[Literal["back"]] = "back"


Action = Click | Move | Drag | TypeText | PressKey | Scroll | Wait | Navigate | Back
"""Closed set: do not add a member without updating every controller."""

ACTION_TYPES: Mapping[str, type] = {
    cls.kind: cls for cls in (Click, Move, Drag, TypeText, PressKey, Scroll, Wait, Navigate, Back)
}


def action_to_dict(action: Action) -> dict[str, Any]:
    """JSON-safe dict: points become ``{"x", "y"}`` and tuples become lists."""
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
    """Rebuild an action from ``action_to_dict`` output, raising ``ValueError`` on an
    unknown ``kind`` or fields that do not fit."""
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
    """The outcome of ``Controller.perform``. ``ok`` means the input was DELIVERED, not
    that the UI did what was hoped - that is a ``Critic``'s job. ``elapsed_ms`` includes
    any settle wait."""

    ok: bool
    error: str | None = None
    elapsed_ms: float = 0.0


@runtime_checkable
class Controller(Protocol):
    """Eyes and hands on one screen: a browser page or a real desktop. Coordinates are
    LOGICAL pixels from ``viewport``'s top-left. NOT thread-safe."""

    def capture(self) -> Screenshot:
        """The current frame, sized to the viewport's logical size. Raises ``ControllerError``."""
        ...

    def perform(self, action: Action) -> ActionResult:
        """Execute one action and wait for the UI to settle. A failure to DELIVER it is
        reported as ``ActionResult(ok=False)`` so an exploring agent carries on;
        ``ControllerError`` means the controller itself is unusable."""
        ...

    def viewport(self) -> Box:
        """The controllable area in LOGICAL pixels. A desktop controller confined to a
        window may report an offset; action coordinates stay relative to this box's
        top-left regardless."""
        ...

    def supports(self, action_kind: ActionKind) -> bool:
        """``"navigate"`` and ``"back"`` are the commonly unsupported kinds: both need the
        session history only a browser has."""
        ...

    def url(self) -> str | None:
        """The address bar, or ``None`` for a desktop - not ground truth about contents."""
        ...

    def describe(self) -> str:
        """One line for logs and prompts: ``"playwright chromium 1280x800 @1x"``."""
        ...

    def close(self) -> None:
        """Release browser or OS resources. Idempotent; never raises."""
        ...


@runtime_checkable
class GroundTruthSource(Protocol):
    """Perfect knowledge of the screen from the DOM or accessibility tree.

    AN OFFLINE TEACHER ONLY - detector labels, eval scoring, and the world reset. The
    agent's action path (perceiver, explorer, planner, runner, skill code) must NEVER call
    it, and whatever may takes it as an explicit argument so the dependency is visible.
    """

    def elements(self) -> list[Element]:
        """Every visible or interactable element: LOGICAL boxes, ``source=dom``,
        ``confidence=1.0``. Raises ``ControllerError`` if the page cannot be read."""
        ...

    def url(self) -> str:
        """The exact current URL, ``""`` when there is none."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Finds UI elements in pixels (YOLO). Does not read text."""

    def detect(self, screenshot: Screenshot) -> list[Element]:
        """Detected elements, highest confidence first, ``source=yolo``, boxes converted
        to LOGICAL pixels. Raises ``PerceptionError``."""
        ...


@runtime_checkable
class TextReader(Protocol):
    """Reads text in pixels (OCR)."""

    def read(self, screenshot: Screenshot) -> list[Element]:
        """One ``ElementKind.text`` element per recognized line or word, in reading order,
        ``source=ocr``, boxes LOGICAL. Raises ``PerceptionError``."""
        ...


@runtime_checkable
class Fingerprinter(Protocol):
    """Turns a screen into a :class:`Fingerprint`."""

    def fingerprint(
        self, screenshot: Screenshot, elements: Sequence[Element], url: str | None = None
    ) -> Fingerprint:
        """The identity of a screen. Deterministic, robust to cosmetic change (a caret, a
        clock) and sensitive to structural change (a dialog opening)."""
        ...


@runtime_checkable
class Perceiver(Protocol):
    """Composes capture, detect, read, merge, index and fingerprint into one call."""

    def observe(self, controller: Controller) -> Observation:
        """A full ``Observation`` from ``capture()`` and ``url()`` alone: never performs an
        action, never touches a ``GroundTruthSource``."""
        ...


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a skill came from: the ``Trajectory.run_id`` it was synthesized from, that
    run's task, and the ``LLMClient.name()`` that wrote the code."""

    trajectory_id: str
    task_text: str
    model: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SkillStats:
    """Running record of how a skill performs; ``mean_ms`` averages SUCCESSFUL runs only.

    Says only what the sandbox was told, which the verifier told the sandbox - so it reads
    14/14 for a skill whose verifier passed on the wrong screen. Read the critic's verdict
    and the site when asking whether a replay did the job.
    """

    runs: int = 0
    successes: int = 0
    mean_ms: float = 0.0
    last_ok_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Precedent:
    """One request a skill is PROVEN to have served, with the arguments it passed under.
    The first is the admission gate's own re-run; later ones are warm runs under a
    different wording that passed BOTH the verifier and the critic."""

    task_text: str
    args: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class Skill:
    """A reusable, versioned piece of automation stored as Python source.

    Attributes:
        name: Snake-case, unique within ``domain``.
        domain: The site or app (``"example.com"``, ``"desktop:finder"``), namespaced by
            perception path - see ``perception_mode``.
        summary: One line, for retrieval and planning prompts.
        params: Parameter name to a JSON-schema-like description; read-only.
        code: ``def run(ctx, **params)``, which may touch the world ONLY through ``ctx``.
        requires: Other skills of the same domain that ``code`` calls.
        precondition: The start screen, or ``None`` to start anywhere.
        verifier_code: ``def verify(ctx, result) -> bool``, or ``None``.
        version: ``0`` until ``SkillStore.put`` assigns ``1, 2, 3, ...``.
        demoted_reason: Set once ``SkillStore.demote`` retires it from retrieval.
        action_signature: What the skill DOES with labels, values and URLs abstracted
            away; close signatures are one FAMILY (``skills.family``). Empty until a
            verifier-passed run earns it, and an empty signature has no family.
        precedents: Requests this skill is proven to have served, oldest first.
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
    action_signature: tuple[str, ...] = ()
    precedents: tuple[Precedent, ...] = ()


@runtime_checkable
class SkillStore(Protocol):
    """Durable, versioned storage of skills, keyed by ``(name, domain)``."""

    def put(self, skill: Skill) -> Skill:
        """Store as the NEXT version of ``(name, domain)``, ignoring the incoming
        ``version``, and return the stored copy. Older versions are kept, and admission
        checks happen before this."""
        ...

    def get(self, name: str, domain: str, version: int | None = None) -> Skill:
        """One skill, latest when ``version`` is ``None``. Demoted skills are still
        returned - check ``demoted_reason``. Raises ``SkillNotFound``."""
        ...

    def list(self, domain: str | None = None, *, include_demoted: bool = False) -> list[Skill]:
        """The LATEST version of every skill, sorted by ``(domain, name)``."""
        ...

    def record_run(self, name: str, domain: str, ok: bool, ms: float) -> Skill:
        """Fold one execution into the latest version's ``SkillStats``. Raises ``SkillNotFound``."""
        ...

    def demote(self, name: str, domain: str, reason: str) -> Skill:
        """Retire the latest version from retrieval; a later ``put`` is healthy again.
        Raises ``SkillNotFound``."""
        ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrieval hit: the skill, a ``0.0..1.0`` relevance ``score`` and a short ``why``."""

    skill: Skill
    score: float
    why: str = ""


@runtime_checkable
class SkillRetriever(Protocol):
    """Finds stored skills relevant to a task."""

    def search(self, task: str, domain: str | None = None, k: int = 5) -> list[Candidate]:
        """At most ``k`` candidates, best score first, never demoted ones. A RANKING: the
        planner still decides whether the winner accounts for the request."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors for retrieval."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One deterministic vector per text, in order, same length, L2-normalized so a dot
        product is cosine similarity."""
        ...


@dataclass(frozen=True, slots=True)
class UIState:
    """A node of the site graph - one recognizable screen - keyed by ``fingerprint``.
    ``thumbnail`` is small PNG bytes for the dashboard."""

    fingerprint: Fingerprint
    domain: str
    label: str = ""
    url_pattern: str | None = None
    first_seen: datetime = field(default_factory=utcnow)
    thumbnail: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class Transition:
    """An edge of the site graph, identified by ``(src, dst, actions)``: performing
    ``actions`` on ``src`` leads to ``dst``. ``mean_ms`` averages SUCCESSFUL traversals.

    ``attempts`` and ``successes`` SUM when two records merge, so a save must be handed a
    DELTA and never the whole record - see ``graph.model.InMemorySiteGraph.unsaved``.
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
    """A path through the site graph: the flattened ``steps`` to perform, the ``edges``
    they follow, and ``cost`` in expected milliseconds (each edge's ``mean_ms`` over its
    success rate). A route to the state you are on is empty at cost ``0.0``."""

    steps: tuple[Action, ...]
    cost: float
    edges: tuple[Transition, ...]


@runtime_checkable
class GraphView(Protocol):
    """The read-only part of a ``SiteGraph``; what skill code sees."""

    def route(self, src_fp: Fingerprint, dst_fp: Fingerprint) -> Route | None:
        """The lowest-cost known route over edges with at least one success, or ``None``.
        Matches EXACTLY on ``Fingerprint.value``, so a caller on a live page must settle
        "am I already there?" by similarity first."""
        ...

    def neighbors(self, fp: Fingerprint) -> list[Transition]:
        """Outgoing edges of ``fp``, most reliable first."""
        ...

    def states(self, domain: str) -> list[UIState]:
        """Every known state of ``domain``, oldest first."""
        ...


@runtime_checkable
class SiteGraph(GraphView, Protocol):
    """Per-domain memory of screens and how to move between them."""

    def upsert_state(self, state: UIState) -> UIState:
        """Add a state, or update a known one's label, URL pattern and thumbnail; the
        original ``first_seen`` is kept."""
        ...

    def observe_transition(
        self,
        src: Fingerprint,
        actions: Sequence[Action],
        dst: Fingerprint,
        ok: bool,
        ms: float,
    ) -> Transition:
        """Record one attempt at ``(src, dst, actions)``, where ``dst`` is the state the
        actions were EXPECTED to reach and ``ok`` whether it was. Unknown states are added
        implicitly under ``src``'s domain, else ``""``."""
        ...

    def save(self) -> None:
        """Persist every loaded domain; a no-op for in-memory graphs."""
        ...

    def load(self, domain: str) -> None:
        """Load one domain from storage, replacing what is in memory. Nothing stored loads
        as empty rather than raising."""
        ...


@dataclass(frozen=True, slots=True)
class Usage:
    """Token and money accounting for LLM calls; add two with ``+``."""

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
    """A tool offered to the model; ``input_schema`` is a read-only JSON Schema object."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(hash=False)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asking for a tool; ``id`` matches the result back and is ``""`` when the
    provider has none."""

    name: str
    args: Mapping[str, Any] = field(hash=False)
    id: str = ""


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One provider-neutral conversation turn. On a ``"tool"`` turn ``text`` is the result
    and ``tool_call_id`` names the ``ToolCall`` it answers; on a replayed ``"assistant"``
    turn ``tool_calls`` are the calls it made."""

    role: Literal["user", "assistant", "tool"]
    text: str = ""
    images: tuple[bytes, ...] = field(default=(), repr=False)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One model reply; ``usage`` is this call alone and ``stop_reason`` is normalized to
    ``"end"``, ``"tool_use"``, ``"max_tokens"`` or ``"other"``."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end"


@runtime_checkable
class LLMClient(Protocol):
    """A chat model behind a provider-neutral interface."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send the whole conversation and return one reply. Blocking and stateless apart
        from the usage total, and raises ``ProviderError`` once its own retries are spent.

        An adapter whose model rejects sampling parameters - current Claude models do -
        MUST silently drop a non-``None`` ``temperature`` rather than fail.
        """
        ...

    def total_usage(self) -> Usage:
        """Every ``complete`` call on this client, summed. An attempt's cost is read from
        HERE, never from what a path reports about itself."""
        ...

    def name(self) -> str:
        """The model identifier, recorded in provenance."""
        ...


@dataclass(frozen=True, slots=True)
class Verdict:
    """A judgment on whether a step or task achieved its goal. ``source`` is
    ``"programmatic"`` (a fingerprint, a verifier, a rule - free and trustworthy) or
    ``"model"`` (an LLM looked at the screens)."""

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
        """Whether ``goal`` was achieved going from ``before`` to ``after``, with
        ``expectation`` optionally describing ``after``. Programmatic checks first, a model
        only when those are inconclusive."""
        ...


@dataclass(frozen=True, slots=True)
class Budget:
    """Hard limits for one run; ``max_usd`` is US dollars of LLM spend."""

    max_steps: int = 40
    max_seconds: float = 300.0
    max_usd: float = 2.0
    max_llm_calls: int = 60


@dataclass(slots=True)
class Spend:
    """MUTABLE running total against a ``Budget`` - the one non-frozen type here, and not
    thread-safe. Call ``check`` BEFORE each unit of work: it raises once a limit is
    REACHED, meaning nothing remains."""

    budget: Budget = field(default_factory=Budget)
    steps: int = 0
    seconds: float = 0.0
    usd: float = 0.0
    llm_calls: int = 0
    _started: float | None = field(default=None, repr=False)

    def start(self) -> Spend:
        """Start the wall clock; idempotent, returns ``self`` for chaining."""
        if self._started is None:
            self._started = time.monotonic()
        return self

    def elapsed_seconds(self) -> float:
        """``seconds`` plus wall-clock seconds since ``start``, if started."""
        running = 0.0 if self._started is None else time.monotonic() - self._started
        return self.seconds + running

    def add_step(self, n: int = 1) -> None:
        """Charge ``n`` steps; one step is one performed ACTION, not one decision."""
        self.steps += n

    def add_usage(self, usage: Usage) -> None:
        """Charge the ``calls`` and ``cost_usd`` of one or more LLM calls."""
        self.llm_calls += usage.calls
        self.usd += usage.cost_usd

    def check(self) -> None:
        """Raise ``BudgetExceeded`` naming the exhausted limit, if any is reached.

        Enforced from Python, so it bounds no native call: anything entering a native
        library bounds its own work (see ``perception.ocr.OcrWorker``).
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


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One action of a run with the observations either side of it. ``note`` is free text,
    typically the agent's stated reason.

    A move that ran a code block is SEVERAL steps and both the reason and the verdict sit
    on its last one, so a reader must regroup - see ``trajectory.render.moves_of``.
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
    """The full record of one run - the raw material a skill is synthesized from."""

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
    """Builds one ``Trajectory`` at a time: ``start``, then ``step`` per action, then
    ``finish``."""

    def start(self, task: str, domain: str) -> str:
        """Begin a run and return its fresh unique ``run_id``."""
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
        """Append one step, assigning its ``index``."""
        ...

    def finish(self, ok: bool, note: str = "") -> Trajectory:
        """Close the run and return the trajectory; persisting it is a ``TrajectoryStore``'s job."""
        ...


@runtime_checkable
class TrajectoryStore(Protocol):
    """Durable storage of finished trajectories."""

    def save(self, trajectory: Trajectory) -> None:
        """Persist a trajectory, replacing any with the same ``run_id``."""
        ...

    def load(self, run_id: str) -> Trajectory:
        """One trajectory, raising ``SkillWeaverError`` for an unknown or unreadable id."""
        ...

    def list(self) -> list[str]:
        """Every stored ``run_id``, oldest first."""
        ...


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A task for the agent. ``params`` is free-form and read-only: a ``"start_url"``,
    values to fill in, and the reset hooks ``"reset_url"`` / ``"reset_actions"``."""

    text: str
    domain: str
    target: Literal["browser", "desktop"] = "browser"
    params: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """The result of running one task end to end; ``skill_used`` is ``None`` when
    exploration solved it."""

    ok: bool
    trajectory: Trajectory
    verdict: Verdict
    spend: Spend
    skill_used: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class SkillResult:
    """The result of executing one skill. ``steps`` counts controller actions including
    nested skills', and ``trace`` is the ordered ``ctx.log`` lines and action descriptions."""

    ok: bool
    value: Any = None
    steps: int = 0
    ms: float = 0.0
    error: str | None = None
    trace: tuple[str, ...] = ()


@runtime_checkable
class ActionSurface(Protocol):
    """The action-only view of a controller that skill code gets as ``ctx.ctl``: no
    ``capture``, no ``close``, no ground truth - and no ``navigate`` or ``back``, so a
    stored skill can never pop a history stack it did not build.

    Unlike ``Controller.perform``, every method RAISES ``ControllerError`` on an undelivered
    action, so skill code need not check results. A target may be a ``Point``, a ``Box`` or
    an ``Element``, the latter two by their centre.
    """

    def perform(self, action: Action) -> ActionResult:
        """Perform any action; raises ``ControllerError`` if it is not delivered."""
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
        """Press a key chord: ``press("Enter")``, ``press("Meta", "a")``."""
        ...

    def scroll(self, target: Point | Box | Element, dx: int = 0, dy: int = 0) -> ActionResult:
        """Scroll over the target; positive ``dy`` scrolls down."""
        ...

    def wait(self, ms: int) -> ActionResult:
        """Pause for ``ms``. A skill must never sleep for time the browser already spent:
        every action is settled before the controller returns, so wait for a THING (see
        ``ctx.wait_for_text``) rather than for a time."""
        ...


@runtime_checkable
class SkillContext(Protocol):
    """The ONLY object skill code receives: ``def run(ctx, **params)``, and the complete
    published surface. Nothing else - imports, files, network, controller internals, ground
    truth - is reachable, and the sandbox raises ``SandboxViolation`` on any attempt."""

    @property
    def ctl(self) -> ActionSurface:
        """Hands: the action-only view of the controller."""
        ...

    @property
    def see(self) -> ElementIndex:
        """Eyes: the screen AS IT IS NOW, re-observed lazily after any action through
        ``ctl``, so never cache it across actions."""
        ...

    @property
    def graph(self) -> GraphView:
        """Read-only view of the current domain's site graph."""
        ...

    def call(self, name: str, **kwargs: Any) -> Any:
        """Run another skill of the same domain and return its value; the callee's
        ``ExpectationFailed`` and ``ControllerError`` propagate."""
        ...

    def expect(self, condition: bool, why: str) -> None:
        """Raises ``ExpectationFailed(why)`` when false, which fails the skill cleanly."""
        ...

    def log(self, msg: str) -> None:
        """Append a line to the run's trace. Never raises."""
        ...


@runtime_checkable
class SkillRunner(Protocol):
    """Executes stored skill code inside the sandbox."""

    def run(self, skill: Skill, args: Mapping[str, Any], ctx: SkillContext) -> SkillResult:
        """Execute ``skill.code`` with ``args`` and its verifier if it has one. Every skill
        failure - exception, expectation, verifier, sandbox violation - is REPORTED as
        ``SkillResult(ok=False)``; ``BudgetExceeded`` is the one exception allowed out."""
        ...


@dataclass(frozen=True, slots=True)
class SkillCall:
    """A plan step that invokes a stored skill with ``args``."""

    name: str
    domain: str
    args: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class Plan:
    """A model-free way to do a task from what is already known: ordered ``SkillCall``s
    and raw ``Action``s (typically route steps between skills), with ``estimated_ms``
    from recorded stats."""

    steps: tuple[SkillCall | Action, ...]
    skills_used: tuple[str, ...] = ()
    estimated_ms: float = 0.0


@runtime_checkable
class Explorer(Protocol):
    """The slow path: solve a task by trial and error with a computer-use model."""

    def explore(self, task: TaskSpec, controller: Controller, budget: Budget) -> RunOutcome:
        """Attempt the task within ``budget``. Running out of budget and failing the task
        both give ``RunOutcome(ok=False)`` with the partial trajectory rather than raising."""
        ...


@runtime_checkable
class Planner(Protocol):
    """The fast path: compose stored skills and known routes, without exploring."""

    def plan(self, task: TaskSpec, observation: Observation) -> Plan | None:
        """A plan from the current screen, or ``None`` when the library and graph do not
        cover it and the caller must fall back to an ``Explorer``."""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """Turns a successful trajectory into a reusable skill."""

    def synthesize(self, trajectory: Trajectory) -> Skill | None:
        """An unstored ``version=0`` skill, or ``None`` when the trajectory is not worth
        keeping. Neither admits nor stores it."""
        ...
