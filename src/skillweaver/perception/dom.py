"""The DOM perceiver: ask the page what is on it, instead of looking at a picture of it.

A ``Perceiver`` that reads the browser's own accessibility information and emits ordinary
``Element`` values with ``source=dom``. It runs NO detector and NO OCR.

Read ``AGENTS.md`` first: the acting loop is pixels-only by design, and this is a
DELIBERATE, authorized relaxation for BROWSER USE ONLY, unreachable without
``--perception dom``. It is a separate class from ``BrowserGroundTruth`` so the
relaxation stays visible.

It buys the whole OCR bill - reading is 84-97% of a pixel observation - and costs
everything the page declines to say: text in an image, a canvas, a cross-origin iframe or
a closed shadow root. The two paths see genuinely different screens, which is why they
keep separate skill libraries; see :mod:`skillweaver.perception_mode`, and measure the
two paths with ``PerceptionCounts`` rather than claiming a speedup.

:meth:`DomPerceiver.observe` emits CONTROLS (actionable, carrying a :class:`DomControl` in
:attr:`DomPerceiver.last` for the policy's target table) and then visible TEXT, both in
reading order. The policy ignores the text; ``ctx.see.find_text`` in stored skills and
``structural_hash`` do not.

Geometry is LOGICAL pixels: ``getBoundingClientRect`` already reports CSS pixels, so the
conversion is at scale 1.0 through :mod:`skillweaver.controllers._coords`, and only the
rounding has to agree with everything else.

A control whose centre is outside the viewport is dropped, as in Jev's ``snapshot.js``:
actions here are POINT-based, so an element the camera could not see is one the mouse
cannot reach, and offering it would offer a target that silently misses. Scrolling is how
the rest of the page is reached.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from skillweaver.contracts import (
    Box,
    Controller,
    Element,
    ElementKind,
    ElementSource,
    Fingerprinter,
    Observation,
    Screenshot,
    utcnow,
)
from skillweaver.controllers import _coords
from skillweaver.errors import ControllerError, PerceptionError
from skillweaver.logging_ import get_logger
from skillweaver.perception.elements import build_index
from skillweaver.perception.fingerprint import StateFingerprinter
from skillweaver.perception.ocr import PerceptionCounters

log = get_logger(__name__)

__all__ = [
    "ARRIVAL_REST_MS",
    "CHANGED_REST_MS",
    "MAX_CONTROLS",
    "MAX_PAGE_TEXT",
    "MAX_TEXT_NODES",
    "NO_CHANGE_REST_MS",
    "SNAPSHOT_ATTEMPTS",
    "SNAPSHOT_RETRY_MS",
    "DomControl",
    "DomOption",
    "DomPerceiver",
    "DomScroller",
    "DomSnapshot",
    "ROLE_KINDS",
]

ARRIVAL_REST_MS: tuple[float, float] = (150.0, 1500.0)
"""``(quiet, cap)`` for a move that CHANGED THE ADDRESS: a navigation may still be
materializing its content, and a decision belongs on a quiet page rather than on
whatever rendered first. Upstream's numbers. See :meth:`DomPerceiver.rest_after`."""

NO_CHANGE_REST_MS: tuple[float, float] = (120.0, 1000.0)
"""``(quiet, cap)`` for a move after which the page reads EXACTLY as before: the effect may
land asynchronously - a cart badge, a toast - so look once more before "no change" is
what the policy is told. Upstream's numbers."""

CHANGED_REST_MS: tuple[float, float] = (150.0, 1500.0)
"""``(quiet, cap)`` for a move after which the page CHANGED IN PLACE. This one is not
upstream's - its two conditions leave this case unrested, and so did this project's for a
day - and it was paid for on a live site. splitkb.com, a product page's *More info*: the
click opened a modal, the page had visibly answered, so the next frame was taken at once -
and the policy was offered ONE control, the modal's close button, because the video player
inside it had not hydrated yet. It closed the popup it had been asked to play a video in.
The same page observed after a quiesce offers two. A page answers in PHASES
(``AGENTS.md``, where Walmart's *Add* button arrives 0.61s after its title), and a frame
taken between two of them is a skeleton whatever the address bar says."""

MAX_CONTROLS = 250
"""Controls reported from one frame, matching Jev's own cap: it bounds the policy's
target table, which is what grows the request. :attr:`DomSnapshot.omitted_controls` says
when it bit, so a cap that truncates a real page is findable rather than silent."""

MAX_TEXT_NODES = 400
"""Visible text elements per frame. Higher than the control cap because text is what a
stored skill reads and a dense article legitimately has hundreds of lines; still a cap,
because ``ElementIndex.best`` and ``find_text`` scan the list per query."""

SNAPSHOT_ATTEMPTS = 4
"""Asks of a document that answers ``null`` (see :meth:`DomPerceiver._read`). Four,
because a page that has committed a navigation gets a body within a frame or two and one
that never does is broken and worth reporting rather than waiting on."""

SNAPSHOT_RETRY_MS = 150.0
"""How long to wait between those attempts."""

MAX_PAGE_TEXT = 6000
"""Characters of visible page text on the snapshot, as in Jev's ``snapshot.js``. The blob
a policy is shown as context, NOT the element list, and capped because it goes into a
request body on every step."""

ROLE_KINDS: dict[str, ElementKind] = {
    "button": ElementKind.button,
    "link": ElementKind.link,
    "checkbox": ElementKind.checkbox,
    "switch": ElementKind.checkbox,
    "radio": ElementKind.radio,
    "menuitemradio": ElementKind.radio,
    "tab": ElementKind.tab,
    "menuitem": ElementKind.menu,
    "option": ElementKind.menu,
    "combobox": ElementKind.menu,
    "textbox": ElementKind.text_field,
    "searchbox": ElementKind.text_field,
    "spinbutton": ElementKind.text_field,
    "gridcell": ElementKind.row,
}
"""ARIA role to ``ElementKind``. Deliberately lossy in one direction - several roles
collapse onto ``menu`` and ``text_field`` - because ``ElementKind`` is the vocabulary a
PIXEL detector can support, and a kind it cannot emit is one stored skill code could only
find on one path. An editable ``combobox`` is re-mapped in :func:`_kind_of`."""


@dataclass(frozen=True, slots=True)
class DomOption:
    """One selectable value of a ``<select>``. ``label`` is what a person reads."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class DomControl:
    """One actionable thing on the page, as the page describes it.

    The richness ``Element`` deliberately does not carry: ``Element`` is source-agnostic
    and widening it would change the shared surface, so a policy that needs to know a
    checkbox is already ticked reads it from here, via :attr:`DomPerceiver.last`.

    Attributes:
        index: 1-based position in :attr:`DomSnapshot.controls`, the number an indexed
            target table quotes.
        element_id: The id this control's ``Element`` carries as ``stable_id``, and so
            the id ``ElementCatalog`` files it under - the ONE join between a policy's
            choice and the explorer's grounding, computed once, here.
        role: The ARIA role, as ``snapshot.js`` resolves it.
        name: The accessible name.
        kind: ``role`` mapped onto :data:`ROLE_KINDS`.
        box: Bounding rectangle in LOGICAL pixels, clipped to the viewport.
        value: The field's current value.
        editable: Whether text can be typed into it.
        checked / selected / expanded: The matching ARIA state. ``None`` means the page
            did not say, and must not be read as ``False``.
        options: The selectable values, for a ``<select>`` only.
        scope_text: Text of the nearest enclosing row, card or form - what tells a policy
            that this ``Add`` button belongs to THAT product.
    """

    index: int
    element_id: str
    role: str
    name: str
    kind: ElementKind
    box: Box
    value: str = ""
    editable: bool = False
    checked: bool | None = None
    selected: bool | None = None
    expanded: bool | None = None
    options: tuple[DomOption, ...] = ()
    scope_text: str = ""

    @property
    def label(self) -> str:
        """The name, or the role - never empty, so a target table can quote an icon-only
        control."""
        return self.name or self.role


@dataclass(frozen=True, slots=True)
class DomScroller:
    """The box a wheel at the middle of the viewport would scroll, when that is not the
    document: a dialog's option list, a results pane, a drawer.

    Found by walking up from ``elementFromPoint`` at the viewport centre to the nearest
    ancestor that overflows and may scroll, as upstream Jev's ``snapshot.js`` does. The
    start point is not arbitrary: it is where this path's targetless ``Scroll`` is
    delivered (``Explorer._resolve`` wheels at ``controller.viewport().center``, the same
    floor division), so the box found IS the box the wheel moves and the offer cannot
    disagree with the action. Aiming at the box's own centre instead would not be safer -
    a nested scroller can sit under that point and take the wheel.

    Attributes:
        can_down / can_up: Whether THIS box has content past its own fold.
        label: What the box calls itself, for the log - ``aria-label``, else its role,
            else its tag.
    """

    can_down: bool
    can_up: bool
    label: str = ""


@dataclass(frozen=True, slots=True)
class DomSnapshot:
    """Everything one ``page.evaluate`` returned, beside the observation it produced.

    Attributes:
        url / title: The document's own.
        text: Visible page text, capped at :data:`MAX_PAGE_TEXT`.
        controls: Actionable elements in reading order, 1-based via ``DomControl.index``.
        texts: Visible text runs as ready-made ``text`` elements, not in the policy's
            target table.
        viewport: The visible area in logical pixels.
        scroll_y / page_height: What decides whether scrolling is offered, when the
            DOCUMENT is what a wheel would move.
        scroller: The box a wheel would move instead, or ``None`` when that is the
            document. When set it ALONE decides the offer; see :attr:`can_scroll_down`.
        loading: The page saying it is still arriving - ``readyState`` short of
            ``complete``, or an ``aria-busy`` / ``progressbar`` on it. ADVISORY, as
            upstream has it: shown to the policy so a ``WAIT`` can be deliberate, and
            part of nothing that identifies the screen.
        omitted_controls: How many controls :data:`MAX_CONTROLS` cut.
        covered_controls: How many controls were left out because something else is at
            the point a click on them would land - clipped by a scrolling list, or under
            a banner. See :data:`_SNAPSHOT_JS`.
        can_go_back: Whether this tab has a previous session-history entry - the page's
            own answer from the Navigation API, not ``history.length``. It counts only
            entries contiguous and SAME-ORIGIN with this one, so the ``about:blank`` a run
            starts from does not make it true and a back never leaves the site.
        by_element_id: The controls keyed by ``element_id``, which is how a policy turns
            the id it chose back into what it knows about that control.
    """

    url: str
    title: str
    text: str
    controls: tuple[DomControl, ...]
    texts: tuple[Element, ...] = ()
    viewport: Box = Box(0, 0, 0, 0)
    scroll_y: float = 0.0
    page_height: float = 0.0
    omitted_controls: int = 0
    covered_controls: int = 0
    can_go_back: bool = False
    scroller: DomScroller | None = None
    loading: bool = False
    by_element_id: dict[str, DomControl] = field(default_factory=dict, compare=False, repr=False)

    @property
    def digest(self) -> str:
        """Whether the page is LITERALLY the page it was: upstream's ``fingerprint``.

        Address, visible text, every control with its value and state, and where things
        are scrolled to. A change DETECTOR and an exact one, which is a different job from
        identifying a screen: that stays ``StateFingerprinter`` and its calibrated
        similarity, and nothing stored is ever keyed on this. It sees what the pixel
        identity cannot - a ticked box, a list scrolled inside a dialog, a cart badge
        going from 0 to 1 - which are the changes a policy's history has to be honest
        about.
        """
        scroller = self.scroller
        content = [
            self.url,
            self.text,
            [(c.element_id, c.value, c.checked, c.selected, c.expanded) for c in self.controls],
            self.scroll_y,
            None if scroller is None else (scroller.can_down, scroller.can_up),
        ]
        return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()[:16]

    @property
    def can_scroll_down(self) -> bool:
        """Whether a wheel at the middle of the screen has anything below to reveal.

        The document's own extent answered this alone until 2026-09-20, and that is
        wrong under a dialog: the page behind is locked, so the document reports nothing
        to scroll while the option list in front of it has a fold of its own. Measured on
        a modal of 40 options over a locked 800px page: ``False`` here, NO scroll
        operation offered, and 14 controls the policy could never reach - while one wheel
        at the viewport centre moved that list 0 -> 560. With :attr:`scroller` the same
        screen offers ``SCROLL_DOWN``, and ``SCROLL_UP`` once it has moved.

        The box decides ALONE when there is one, as upstream does it: a locked document
        often still reports a tall ``scrollHeight``, and offering a scroll on the
        strength of that is offering a wheel that moves nothing.
        """
        if self.scroller is not None:
            return self.scroller.can_down
        return self.scroll_y + self.viewport.h < self.page_height - 2

    @property
    def can_scroll_up(self) -> bool:
        """Whether there is anything above to bring back. See :attr:`can_scroll_down`."""
        if self.scroller is not None:
            return self.scroller.can_up
        return self.scroll_y > 0


class DomPerceiver:
    """A ``Perceiver`` over the page's own control list.

    Fills ``Observation`` exactly as ``ComposedPerceiver`` does, so everything above
    ``Element`` is unaffected by which path produced it.

    Args:
        fingerprinter: Defaults to ``StateFingerprinter``, which every stored
            precondition was hashed with.
        counters: The tally to charge work to. ``ocr_reads`` staying at zero on this path
            is the measurement, not an omission.

    Not thread-safe: :attr:`last` is one slot, so one perceiver drives one browser.
    """

    __slots__ = (
        "_acted_ms",
        "_armed",
        "_resting",
        "_counters",
        "_fingerprinter",
        "_last",
        "_observed_ms",
        "_rested_ms",
        "_rests",
    )

    def __init__(
        self,
        fingerprinter: Fingerprinter | None = None,
        *,
        counters: PerceptionCounters | None = None,
    ) -> None:
        self._fingerprinter = fingerprinter if fingerprinter is not None else StateFingerprinter()
        self._counters = counters if counters is not None else PerceptionCounters()
        self._last: DomSnapshot | None = None
        self._armed: tuple[int, DomSnapshot | None, bool] | None = None
        self._resting: tuple[str, float, DomSnapshot] | None = None
        self._rests = 0
        self._rested_ms = 0.0
        self._observed_ms = 0.0
        self._acted_ms = 0.0

    def __repr__(self) -> str:
        return f"DomPerceiver(counts={self._counters.snapshot()})"

    @property
    def counters(self) -> PerceptionCounters:
        """The running tally of everything this perceiver has done."""
        return self._counters

    @property
    def last(self) -> DomSnapshot | None:
        """The snapshot behind the most recent :meth:`observe`, or ``None``.

        It belongs to the LAST observation only; anything holding an older
        ``Observation`` must not assume this still describes it.
        """
        return self._last

    @property
    def site_ms(self) -> float:
        """Milliseconds this run has spent on the SITE rather than on a model.

        Time inside :meth:`observe` - the capture, the page script, and any
        :meth:`rest_after` wait - plus the controller's own ``acted_ms``, read off it at
        each observation: input delivery and ``_settle``. Together that is everything a
        run waits on the page for, which is upstream's ``load_ms``.

        A DURATION, where ``PerceptionCounters`` is deliberately counts - seconds move
        with machine load, and OCR taught this project that twice. This one is kept
        because its question cannot be asked in counts: of a run's wall clock, how much
        was the models and how much was the website? It is dominated by the network and
        the page, not by this machine, and it is reported beside the counts, never
        instead of them.
        """
        return self._observed_ms + self._acted_ms

    @property
    def rests(self) -> tuple[int, float]:
        """``(how many times, total milliseconds)`` :meth:`rest_after` made a frame wait."""
        return self._rests, self._rested_ms

    def rest_after(self, actions: int, basis: DomSnapshot | None, *, waited: bool = False) -> None:
        """Arm ONE coming observation to be taken on a page that has come to rest.

        An acting policy calls this as it answers: ``basis`` is the screen it decided
        on, and ``actions`` how many controller actions its move performs - the explorer
        observes after each, and the one worth waiting for is the LAST, which is the
        frame the next decision is made on. That frame is taken on a page at rest,
        whichever of three things the move did to it: changed the address
        (:data:`ARRIVAL_REST_MS`), left it reading exactly as ``basis`` did
        (:data:`NO_CHANGE_REST_MS`, skipped when the move WAS a wait), or changed it in
        place (:data:`CHANGED_REST_MS`). The first two are upstream's; the third is the
        one a live run added, and its constant says what that cost. On a page that has
        finished answering, the wait is one quiet window and no more.

        ``basis=None`` is the FIRST frame of a run, which has nothing to be compared
        with and is rested as an arrival: the explorer takes it the moment ``Navigate``
        returns, which is the moment ``_settle`` saw the load event, and a real page is
        not the page yet (``AGENTS.md``, Walmart). An acting policy arms it as it is
        built, before anything has observed. It costs ONE quiet window per run - not per
        action, which is what ``AGENTS.md`` forbids - and the warm path pays it too when a
        policy is configured, where it is the start screen the planner matches a
        precondition against that gets to finish painting.

        Measured, 2026-09-20. A results page opened DIRECTLY that streams its rows in
        after ``load``: without this the step-0 decision was offered 5 of 10 controls and
        answered ``BLOCKED`` 0.92 in zero actions; with it, ``waited_ms=664
        controls_before=4 controls_after=10`` and the first move was the right one at
        1.00. The price on a page already at rest, live Wikipedia warm replay, 3 of 3:
        165-168ms, ``54 -> 54``, ``frame_changed=False``, still 0 model calls.

        Armed per move rather than switched on, because every other reader of this
        perceiver must stay untaxed: a warm replay, the admission gate's rest loop and a
        skill's ``wait_for_text`` all observe in a loop, and a quiet window inside each
        read is what ``AGENTS.md`` forbids ``_settle`` for. Re-arming replaces whatever
        was armed, so a move that stopped early cannot leave a wait behind for long.
        """
        self._armed = (max(int(actions), 1), basis, waited)

    def observe(self, controller: Controller) -> Observation:
        """One frame: capture, ask the page, index, fingerprint.

        Raises:
            ControllerError: if the capture or the page script fails.
            PerceptionError: if the page answers with something unusable, or if
                fingerprinting fails.
        """
        began = time.monotonic()
        shot: Screenshot = controller.capture()
        self._counters.captures += 1
        snapshot = self._read(controller, shot)
        self._counters.detections += 1
        if self._rested(controller, snapshot):
            # Capture AGAIN, then read: the frame and the controls must be one moment,
            # and the frame taken before the wait is the moment being replaced.
            shot = controller.capture()
            self._counters.captures += 1
            snapshot = self._read(controller, shot)
            self._counters.detections += 1
            self._report_rest(snapshot)
        self._last = snapshot
        elements = _elements_of(snapshot)
        url = controller.url()
        observation = Observation(
            screenshot=shot,
            elements=elements,
            index=build_index(elements),
            fingerprint=self._fingerprinter.fingerprint(shot, elements, url),
            url=url,
            taken_at=utcnow(),
        )
        self._counters.observations += 1
        self._observed_ms += (time.monotonic() - began) * 1000.0
        acted = getattr(controller, "acted_ms", None)
        if isinstance(acted, (int, float)):
            self._acted_ms = float(acted)
        return observation

    def _rested(self, controller: Controller, snapshot: DomSnapshot) -> bool:
        """Whether this frame was the armed one AND the page was then made to rest."""
        if self._armed is None:
            return False
        remaining, basis, waited = self._armed
        if remaining > 1:
            self._armed = (remaining - 1, basis, waited)
            return False
        self._armed = None
        if basis is None:
            why, (quiet, cap) = "first", ARRIVAL_REST_MS
        elif snapshot.url != basis.url:
            why, (quiet, cap) = "arrived", ARRIVAL_REST_MS
        elif snapshot.digest != basis.digest:
            why, (quiet, cap) = "changed", CHANGED_REST_MS
        elif not waited:
            why, (quiet, cap) = "unchanged", NO_CHANGE_REST_MS
        else:
            return False
        quiesce = getattr(controller, "quiesce", None)
        if not callable(quiesce):
            return False
        started = time.monotonic()
        quiesce(quiet, cap)
        spent = (time.monotonic() - started) * 1000.0
        self._rests += 1
        self._rested_ms += spent
        self._resting = (why, spent, snapshot)
        return True

    def _report_rest(self, rested: DomSnapshot) -> None:
        """Say what the wait bought: the frame that WOULD have been decided on, against
        the one that will be. Logged after the re-read so one line carries both, which is
        what lets a single live run be its own before-and-after."""
        if self._resting is None:
            return
        why, spent, before = self._resting
        self._resting = None
        log.info(
            "dom.rest",
            why=why,
            waited_ms=round(spent),
            controls_before=len(before.controls),
            controls_after=len(rested.controls),
            frame_changed=before.digest != rested.digest,
        )

    def _read(self, controller: Controller, shot: Screenshot) -> DomSnapshot:
        """Run :data:`_SNAPSHOT_JS` on the controller's page and parse the result.

        Handed to the controller through a duck-typed ``evaluate`` rather than by
        importing ``BrowserController``, so this module stays out of that package.

        Raises:
            ControllerError: the controller cannot run page script at all.
            PerceptionError: the script returned something unusable.
        """
        evaluate = getattr(controller, "evaluate", None)
        if not callable(evaluate):
            raise ControllerError(
                "--perception dom needs a controller that can run page script, and "
                f"{controller.describe()} cannot. The DOM path is browser-only; use "
                "--perception pixels for a desktop target."
            )
        raw = None
        for attempt in range(SNAPSHOT_ATTEMPTS):
            raw = evaluate(_SNAPSHOT_JS)
            if isinstance(raw, dict):
                return _snapshot_from(raw, shot)
            # ``null``, not an exception: the script answers a document with no ``body``
            # yet - a commit that landed and did not finish. The controller's retry
            # cannot see it, catching a DESTROYED context while this one is alive and
            # nearly empty. Measured on live splitkb.com: once per run, and fatal before.
            if attempt + 1 < SNAPSHOT_ATTEMPTS:
                log.info("dom.snapshot.retry", attempt=attempt + 1, url=controller.url())
                time.sleep(SNAPSHOT_RETRY_MS / 1000.0)
        raise PerceptionError(
            f"the page snapshot script returned {type(raw).__name__}, not an object, "
            f"{SNAPSHOT_ATTEMPTS} times: the document has a window but still no body"
        )


def _snapshot_from(raw: dict[str, Any], shot: Screenshot) -> DomSnapshot:
    """Build a :class:`DomSnapshot` from the script's raw result.

    Read defensively: this is data crossing out of a web page, and nonsense should give a
    perception failure naming the field rather than a ``TypeError`` up in the explorer.

    Raises:
        PerceptionError: the result cannot be read as a snapshot.
    """
    try:
        viewport = Box(0, 0, int(raw.get("w") or shot.width), int(raw.get("h") or shot.height))
        controls: list[DomControl] = []
        seen: dict[str, int] = {}
        for position, entry in enumerate(raw.get("controls") or (), start=1):
            control = _control_from(entry, position, viewport, seen)
            if control is not None:
                controls.append(control)
        return DomSnapshot(
            url=str(raw.get("url") or ""),
            title=str(raw.get("title") or ""),
            text=str(raw.get("text") or "")[:MAX_PAGE_TEXT],
            controls=tuple(controls),
            texts=_texts_from(raw.get("texts") or (), viewport, seen),
            viewport=viewport,
            scroll_y=float(raw.get("scrollY") or 0.0),
            can_go_back=bool(raw.get("canGoBack")),
            scroller=_scroller_from(raw.get("scroller")),
            loading=bool(raw.get("loading")),
            page_height=float(raw.get("pageHeight") or 0.0),
            omitted_controls=int(raw.get("omitted") or 0),
            covered_controls=int(raw.get("covered") or 0),
            by_element_id={control.element_id: control for control in controls},
        )
    except PerceptionError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise PerceptionError(f"the page snapshot could not be read: {exc}") from exc


def _scroller_from(entry: Any) -> DomScroller | None:
    """The page's scroller, or ``None`` - which means the document is what scrolls."""
    if not isinstance(entry, dict):
        return None
    return DomScroller(
        can_down=bool(entry.get("canDown")),
        can_up=bool(entry.get("canUp")),
        label=_clean(entry.get("label"))[:80],
    )


def _control_from(
    entry: Any, position: int, viewport: Box, seen: dict[str, int]
) -> DomControl | None:
    """One :class:`DomControl`, or ``None`` when the entry describes nothing clickable."""
    if not isinstance(entry, dict):
        return None
    rect = entry.get("rect") or {}
    box = _coords.clip_to_viewport(
        _coords.box_to_logical(
            rect.get("x", 0), rect.get("y", 0), rect.get("w", 0), rect.get("h", 0), 1.0
        ),
        viewport,
    )
    if box.area == 0:
        return None
    role = str(entry.get("role") or "")
    name = _clean(entry.get("name"))
    editable = bool(entry.get("editable"))
    options = tuple(
        DomOption(_clean(option.get("label")), str(option.get("value") or ""))
        for option in entry.get("options") or ()
        if isinstance(option, dict)
    )
    return DomControl(
        index=position,
        element_id=_element_id(role, name, box, seen),
        role=role,
        name=name,
        kind=_kind_of(role, editable),
        box=box,
        value=_clean(entry.get("value")),
        editable=editable,
        checked=_tri(entry.get("checked")),
        selected=_tri(entry.get("selected")),
        expanded=_tri(entry.get("expanded")),
        options=options,
        scope_text=_clean(entry.get("scope"))[:600],
    )


def _texts_from(entries: Any, viewport: Box, seen: dict[str, int]) -> tuple[Element, ...]:
    """The visible text runs as ``text`` elements, sharing the control id counter.

    Sharing ``seen`` keeps ids unique across BOTH populations, so the catalogue never
    falls back to a positional id because a text run hashed like a control.
    """
    elements: list[Element] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rect = entry.get("rect") or {}
        box = _coords.clip_to_viewport(
            _coords.box_to_logical(
                rect.get("x", 0), rect.get("y", 0), rect.get("w", 0), rect.get("h", 0), 1.0
            ),
            viewport,
        )
        text = _clean(entry.get("text"))
        if box.area == 0 or not text:
            continue
        elements.append(
            Element(
                box=box,
                kind=ElementKind.text,
                text=text,
                confidence=1.0,
                stable_id=_element_id("text", text, box, seen),
                source=ElementSource.dom,
            )
        )
    return tuple(elements)


def _kind_of(role: str, editable: bool) -> ElementKind:
    """``role`` as an ``ElementKind``; see :data:`ROLE_KINDS`.

    An editable ``combobox`` - every site search box with autocomplete - is a
    ``text_field``, because that is what ``by_kind(ElementKind.text_field)`` means by it.
    """
    if role == "combobox" and editable:
        return ElementKind.text_field
    return ROLE_KINDS.get(role, ElementKind.other)


def _elements_of(snapshot: DomSnapshot) -> tuple[Element, ...]:
    """The snapshot's controls and text as one reading-order element tuple.

    Controls keep their ``element_id``, so the id a policy chooses is the id the
    catalogue files it under; a text element cannot collide because the role is in the seed.
    """
    elements = [
        Element(
            box=control.box,
            kind=control.kind,
            text=control.name,
            confidence=1.0,
            stable_id=control.element_id,
            source=ElementSource.dom,
        )
        for control in snapshot.controls
    ]
    elements.extend(snapshot.texts)
    elements.sort(key=lambda element: (element.box.y, element.box.x))
    return tuple(elements)


def _element_id(role: str, text: str, box: Box, seen: dict[str, int]) -> str:
    """A per-screen identity for one element, unique within the snapshot.

    Seeded exactly like ``browser._stable_id`` - kind, text and a 16-pixel position grid -
    so a label that shifts a pixel keeps its name. A duplicate gets a counter suffix
    rather than colliding, because ``ElementCatalog`` falls back to a POSITIONAL id on a
    duplicated ``stable_id`` and that would break the join with the policy's choice; two
    ``Add`` buttons 16 pixels apart in a product grid is an ordinary page.
    """
    seed = f"{role}|{text}|{box.x // 16}|{box.y // 16}"
    base = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def _clean(value: Any) -> str:
    """A page-supplied string, whitespace-collapsed. Anything else becomes ``""``."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _tri(value: Any) -> bool | None:
    """An ARIA tri-state; ``None`` means "the page did not say" and is never coerced to
    ``False``: a policy told a non-checkbox is unticked will happily tick it."""
    if isinstance(value, bool):
        return value
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    return None


_SNAPSHOT_JS = (
    """
() => {
  if (!document.body) return null;
  const MAX_CONTROLS = %(max_controls)d;
  const MAX_TEXT_NODES = %(max_text_nodes)d;
  const MAX_PAGE_TEXT = %(max_page_text)d;

  const safe = (el) => !['password', 'file', 'hidden'].includes(el.type);
  const visible = (el) => !el.closest('[aria-hidden="true"],[inert]') &&
    el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});

  // The accessible name, in the order the ARIA spec resolves it. `seen` stops an
  // aria-labelledby cycle from recursing forever on a hostile or merely broken page.
  const name = (el, seen) => {
    seen = seen || new Set();
    if (!el || seen.has(el)) return '';
    seen.add(el);
    const referenced = (el.getAttribute('aria-labelledby') || '').split(/\\s+/)
      .map((id) => name(document.getElementById(id), seen)).filter(Boolean).join(' ');
    return referenced || el.getAttribute('aria-label') ||
      [...(el.labels || [])].map((l) => name(l, seen)).filter(Boolean).join(' ') ||
      (['button', 'submit', 'reset'].includes(el.type) ? el.value : '') ||
      el.getAttribute('alt') ||
      (el.tagName === 'INPUT' ? '' : [...el.childNodes].map((n) =>
        n.nodeType === 3 ? n.textContent :
        n.nodeType === 1 && n.getAttribute('aria-hidden') !== 'true' ? name(n, seen) : ''
      ).join(' ').trim()) ||
      el.getAttribute('title') || el.getAttribute('placeholder') || '';
  };

  const ROLES = ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem',
    'menuitemradio', 'option', 'gridcell', 'combobox', 'textbox', 'searchbox', 'spinbutton'];
  const SELECTOR = 'a[href],button,input,textarea,select,summary,[contenteditable="true"],' +
    ROLES.map((role) => '[role="' + role + '"]').join(',');

  const role = (el) => {
    const explicit = el.getAttribute('role');
    if (ROLES.includes(explicit)) return explicit;
    if (el.tagName === 'BUTTON' || el.tagName === 'SUMMARY') return 'button';
    if (el.tagName === 'A') return 'link';
    if (el.tagName === 'SELECT') return 'combobox';
    if (el.tagName === 'TEXTAREA' || el.isContentEditable) return 'textbox';
    if (el.tagName === 'INPUT') {
      if (['checkbox', 'radio'].includes(el.type)) return el.type;
      if (['button', 'submit', 'reset', 'image'].includes(el.type)) return 'button';
      if (el.type === 'search') return 'searchbox';
      if (el.type === 'number') return 'spinbutton';
      if (['text', 'email', 'url', 'tel'].includes(el.type)) return 'textbox';
    }
    return null;
  };

  const controls = [];
  let covered = 0;
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (!safe(el) || !visible(el) || el.matches(':disabled') ||
        el.closest('[aria-disabled="true"]')) continue;
    const r = el.getBoundingClientRect();
    const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
    const rname = role(el);
    // A centre outside the viewport is a point the mouse cannot be sent to; see the
    // module docstring on why that is dropped rather than reported.
    if (!rname || r.width <= 0 || r.height <= 0 ||
        cx < 0 || cy < 0 || cx >= innerWidth || cy >= innerHeight) continue;
    // A grid cell that contains its own button would shadow it with a bigger box.
    if (rname === 'gridcell' && el.querySelector('button,[role="button"]')) continue;

    // Is the control what is actually AT the point a click on it would be delivered
    // to? checkVisibility cannot say: it knows nothing of a scroll container clipping
    // its children, or of anything drawn on top. The point is the centre of the box
    // clipped to the viewport, which is where Python will aim (see _control_from). A
    // hit on the control's own <label> counts, because that click activates it.
    const px0 = Math.max(Math.round(r.x), 0), py0 = Math.max(Math.round(r.y), 0);
    const px1 = Math.min(Math.round(r.x + r.width), innerWidth);
    const py1 = Math.min(Math.round(r.y + r.height), innerHeight);
    const hit = document.elementFromPoint(
      px0 + Math.floor((px1 - px0) / 2), py0 + Math.floor((py1 - py0) / 2));
    if (!hit || !(el.contains(hit) || [...(el.labels || [])].some((l) => l.contains(hit)))) {
      covered += 1;
      continue;
    }

    const editable = !el.readOnly && el.getAttribute('aria-readonly') !== 'true' &&
      (['textbox', 'searchbox', 'spinbutton'].includes(rname) ||
        (rname === 'combobox' && ['INPUT', 'TEXTAREA'].includes(el.tagName)));
    const scope = el.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]');
    const control = {
      role: rname,
      name: name(el) || '',
      rect: {x: r.x, y: r.y, w: r.width, h: r.height},
      editable: editable,
      value: 'value' in el ? String(el.value) :
        (el.isContentEditable ? el.innerText.trim() : ''),
      scope: scope ? (scope.innerText || '').slice(0, 600) : '',
    };
    for (const key of ['checked', 'selected', 'expanded']) {
      const value = el.getAttribute('aria-' + key);
      if (value !== null) control[key] = value;
    }
    if (['checkbox', 'radio'].includes(el.type)) control.checked = String(el.checked);
    if (el.tagName === 'SELECT') {
      control.value = [...el.selectedOptions].map((o) => o.label).join(', ');
      control.options = [...el.options]
        .filter((o) => !o.disabled && !o.closest('optgroup[disabled]'))
        .map((o) => ({label: o.label, value: o.value}));
    }
    controls.push(control);
  }
  const omitted = Math.max(0, controls.length - MAX_CONTROLS);
  controls.splice(MAX_CONTROLS);

  // Visible text, with the rectangle of each run, so it can become an Element. A Range
  // is the only way to get the box of a bare text node.
  const texts = [];
  const words = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  let node, length = 0;
  while ((node = walker.nextNode()) && texts.length < MAX_TEXT_NODES) {
    const value = node.textContent.trim(), parent = node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') ||
        !visible(parent)) continue;
    range.selectNodeContents(node);
    const r = range.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight &&
        r.right > 0 && r.left < innerWidth) {
      texts.push({text: value.slice(0, 200), rect: {x: r.x, y: r.y, w: r.width, h: r.height}});
      if (length < MAX_PAGE_TEXT) { words.push(value); length += value.length; }
    }
  }

  // Whether there is a previous entry in this tab's session history. The Navigation
  // API answers exactly that; history.length cannot, because it counts entries in
  // BOTH directions and never says where in the stack we are - a page reached by two
  // backs still reports 3, so it would offer a back that has nothing behind it.
  // Measured on live en.wikipedia.org: false on the run's first page, true after one
  // in-site click INCLUDING a fragment link, false again once that back is taken.
  // A missing API answers false rather than guessing: never offering a move costs a
  // capability, offering one that cannot be made costs a step and a FAILED edge.
  const nav = window.navigation;
  const canGoBack = !!(nav && nav.canGoBack);

  // What a wheel at the middle of the viewport would actually scroll. Under a dialog
  // the document is locked and its option list scrolls on its own, so a document-only
  // test offers no way to the controls below that list's fold. Walk up from the centre
  // to the nearest ancestor that overflows and may scroll; null means the document.
  // The start point is where this path delivers a targetless Scroll - see DomScroller.
  let scroller = null;
  for (let el = document.elementFromPoint(innerWidth >> 1, innerHeight >> 1);
       el && el !== document.body && el !== document.documentElement;
       el = el.parentElement) {
    if (el.scrollHeight > el.clientHeight + 2 &&
        /auto|scroll/.test(getComputedStyle(el).overflowY)) {
      scroller = {
        canDown: el.scrollTop + el.clientHeight < el.scrollHeight - 2,
        canUp: el.scrollTop > 0,
        label: el.getAttribute('aria-label') || el.getAttribute('role') ||
          el.tagName.toLowerCase(),
      };
      break;
    }
  }

  return {
    url: location.href,
    title: document.title,
    w: innerWidth,
    h: innerHeight,
    scrollY: scrollY,
    canGoBack: canGoBack,
    scroller: scroller,
    loading: document.readyState !== 'complete' ||
      !!document.querySelector('[aria-busy="true"],[role="progressbar"]'),
    pageHeight: document.documentElement.scrollHeight,
    text: words.join('\\n').slice(0, MAX_PAGE_TEXT),
    controls: controls,
    texts: texts,
    omitted: omitted,
    covered: covered,
  };
}
""".replace("%(max_controls)d", str(MAX_CONTROLS))
    .replace("%(max_text_nodes)d", str(MAX_TEXT_NODES))
    .replace("%(max_page_text)d", str(MAX_PAGE_TEXT))
)
"""The one page script this perceiver runs, adapted from ``jev-ultrafast/snapshot.js``.

Differences from Jev's. No node-identity ``WeakMap`` or freshness guard: this path acts
by POINT and the explorer re-observes after every action, so a stale target shows up as
the next observation disagreeing. Its HIT-TEST is kept, though, and moved from the
executor to here: upstream reports a covered control and refuses the click
(``CoveredTarget``); this path never offers it, because a click by point lands on
whatever is on top and says nothing. Measured on an option dialog scrolled to its end:
8 of 18 checkboxes reported sat outside the list's visible box, and a click on the one
the task wanted was delivered to the backdrop - 0 ticked, no error anywhere.

Text nodes come back WITH their rectangles, because this path needs ``Element``s. And
select options are carried although no select action exists - see
:mod:`skillweaver.llm.jev_`."""
