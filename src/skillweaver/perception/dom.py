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
    "BLANK_REST_MS",
    "BLANK_SIT_OUTS",
    "CHANGED_REST_MS",
    "EFFECT_BUDGET_MS",
    "EFFECT_POLL_MS",
    "IMMEDIATE_OPERATIONS",
    "MAX_CONTEXT_CHARS",
    "MAX_CONTROLS",
    "MAX_DESCRIBED",
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
"""``(quiet, cap)`` for a move after which the page reads EXACTLY as before, when that move
is NOT one whose effect is watched for (:data:`EFFECT_BUDGET_MS`): a ``BACK``, the driver's
fresh look at a ``DONE`` claim, or a caller that did not say what its move was. One quiet
window and one more look. Upstream's numbers for this case until ``0da4053``, which took
the MUTATIONS out of it; what is left here is what a quiet window is still right for."""

EFFECT_POLL_MS = 100.0
"""How often an unchanged page is re-read while a mutation's effect is watched for.
Upstream's number (``0da4053``)."""

EFFECT_BUDGET_MS: dict[str, float] = {"CLICK": 3000.0, "ENTER": 3000.0, "TYPE_TEXT": 1000.0}
"""How long a MUTATION that shows no change is watched before "no change" is what the
policy is told, by operation. Upstream's numbers: 3.0s for a click or select, 1.0s for a
fill. ``ENTER`` is this project's operation and takes the click's budget, because what it
does is submit - its effect is a request being answered, which is the slow kind - where a
fill's effect is the field's own value, which is there at once or not at all.

This REPLACES the quiet window for these moves and is not added to it, because waiting for
quiet is wrong in BOTH directions (upstream, measured on Walmart). A page awaiting a cart
request makes no mutation and has no completed resource entry, so it reads quiet after one
window while the badge moves a second later: every add-to-cart recorded
``page_changed=False``, which tells the policy the item was not added and sends it to add
another - 40 steps and 89s against 10 and 16.8s. And a page that animates its cart update
never goes quiet, so the full cap was paid after the badge had already changed (1094ms of
load against 231ms). A poll stops the moment there is an answer, so a fast effect costs
what it costs and only a genuine no-op pays the whole budget."""

IMMEDIATE_OPERATIONS: frozenset[str] = frozenset({"SCROLL_DOWN", "SCROLL_UP", "WAIT"})
"""Moves whose effect is immediate, so an unchanged page after one is simply the answer:
nothing is polled and nothing is rested. Upstream excludes both from the watch - a dead
scroll would pay the budget for no reason - and the scroll offset is in
:attr:`DomSnapshot.digest`, so an unchanged page after a scroll is a wheel that moved
nothing. (Upstream still gives an unchanged scroll its 100/700 in-place quiet window, by
the fall-through of its ``elif``; not taken, because there is nothing for it to wait on.)"""

MAX_DESCRIBED = 30
"""Controls given a :attr:`DomControl.context` in one snapshot. Upstream's cap: the lookup
reads ``innerText``, which forces layout, and past this many the page is a list of
near-identical rows whose index carries the same information."""

MAX_CONTEXT_CHARS = 80
"""Characters of one :attr:`DomControl.context`. Upstream's cap."""

CHANGED_REST_MS: tuple[float, float] = (150.0, 1500.0)
"""``(quiet, cap)`` for a move after which the page CHANGED IN PLACE. This one is not
upstream's - its two conditions leave this case unrested, and so did this project's for a
day - and it was paid for on a live site. splitkb.com, a product page's *More info*: the
click opened a modal, the page had visibly answered, so the next frame was taken at once -
and the policy was offered ONE control, the modal's close button, because the video player
inside it had not hydrated yet. It closed the popup it had been asked to play a video in.
The same page observed after a quiesce offers two. A page answers in PHASES
(``AGENTS.md``, where Walmart's *Add* button arrives 0.61s after its title), and a frame
taken between two of them is a skeleton whatever the address bar says.

Upstream has since added this case itself (``1489129``: a dropdown's first frame had 3
elements and no text, a moment later 14 and its options) at ``(100, 700)``. These numbers
are kept: they are the pair the splitkb modal was seen to hydrate under, the shorter pair
has not been run against it, and a page that has finished answering never reaches the cap -
so what the difference costs is 50ms of quiet window on such a move."""

BLANK_REST_MS: tuple[float, float] = (200.0, 1200.0)
"""``(quiet, cap)`` for each sit-out of a BLANK page (:attr:`DomSnapshot.blank`).
Upstream's numbers."""

BLANK_SIT_OUTS = 4
"""How many times a blank page is waited on before it is handed over as it is. Upstream's
count: a bot-check or splash shell replaces itself within seconds, and at the cap this is
~5s, after which blank is the truth about the page and the policy should be told it."""

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
        section: The nearest enclosing landmark, by NAME - ``dialog``, ``search``,
            ``header``, ``nav``, ``main``, ``footer``, ``aside`` or ``form`` - or ``None``
            outside all of them. Named and never scored; see :data:`_SNAPSHOT_JS`.
        opens: The control's ``aria-haspopup`` value (``menu``, ``listbox``, ``dialog``,
            ``true`` ...): what a click on it is declared to open. ``None`` when the page
            did not say.
        context: The text of the row or card around a control whose LABEL IS SHARED with
            another control on this screen - seven *Add item to cart* buttons, thirty
            *hide* links - with the label itself taken out. ``None`` for a control whose
            label is its own, and for one past :data:`MAX_DESCRIBED`. Upstream's
            ``context`` (``cbf517a``); narrower than :attr:`scope_text`, which every
            control carries and which is never shown as part of a name. It is NOT part of
            :attr:`element_id`; see :func:`_element_id` for why.
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
    section: str | None = None
    opens: str | None = None
    context: str | None = None

    @property
    def label(self) -> str:
        """The name, or the role - never empty, so a target table can quote an icon-only
        control."""
        return self.name or self.role

    @property
    def action_name(self) -> str:
        """What DISTINGUISHES this control: upstream's ``action_name``, the label and then
        the context when there is one. What loop guards and a policy's history key on, so
        that withholding one *Add item to cart* does not withhold all seven."""
        return f"{self.label} \u2014 {self.context}" if self.context else self.label


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
        can_press_enter: Whether a NON-EMPTY typeable field holds focus, so that an Enter
            key press has a defined destination. Upstream's ``enter`` action: some
            searches have no button and submit only on Enter.
        enter_label: That field's accessible name, for the offer's wording; ``""`` when
            it has none or nothing is offered.
        by_element_id: The controls keyed by ``element_id``, which is how a policy turns
            the id it chose back into what it knows about that control.
        context_ms: What the shared-label pass cost inside the page script, by the page's
            own clock. A measurement and part of no identity: ``innerText`` forces layout,
            so this is the number to read before raising :data:`MAX_DESCRIBED`.
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
    can_press_enter: bool = False
    enter_label: str = ""
    by_element_id: dict[str, DomControl] = field(default_factory=dict, compare=False, repr=False)
    context_ms: float = field(default=0.0, compare=False)

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
            # ``context`` is IN, as upstream has it (its fingerprint hashes the action
            # list, which carries it). It is the text of the card around a shared-label
            # control, so it is where a card answers its own button - "1 in cart" beside
            # the seventh *Add* - and on a long page that text can sit past the
            # ``MAX_PAGE_TEXT`` cut of ``text`` above, where nothing else here would see
            # it. It cannot make the digest unstable: it is a pure function of the DOM,
            # capped in document order, so an unchanged page re-reads to the same value
            # (checked on Hacker News and on local cards, ``scripts/check_k_dom.py``).
            [
                (c.element_id, c.value, c.checked, c.selected, c.expanded, c.context)
                for c in self.controls
            ],
            self.scroll_y,
            None if scroller is None else (scroller.can_down, scroller.can_up),
            # The ENTER offer is IN, as upstream has it (its fingerprint hashes the
            # action list, and ``enter`` is an action). Deliberate, because of what
            # ``JevDriver`` keys on this: a click that only FOCUSES a field already
            # holding a query changes no text and no value, so without this it is
            # recorded ``page_changed=False`` - "that did nothing" - on the one move that
            # made the submit available, and the spent-move memory files the new offer
            # under the screen that did not have it. What a screen OFFERS is part of
            # which screen it is. ``section`` and ``opens`` are NOT in: they describe a
            # control the ``element_id`` already names, and do not move when it is used.
            self.enter_label if self.can_press_enter else None,
        ]
        return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()[:16]

    @property
    def blank(self) -> bool:
        """An interstitial with nothing to read and nothing to act on: a bot-check or
        challenge shell, or a splash, that replaces itself a moment later.

        Upstream's ``Agent.blank``. Deciding here wastes the decision - the only moves
        left to offer are ``WAIT`` and ``BLOCKED``, and a ``BLOCKED`` ends the run on a
        page that was about to become the real site. This is NOT a way past a
        verification page: one that asks a person for something has text and a control,
        is not blank, and fails the run as ``AGENTS.md`` requires.
        """
        return not self.text.strip() and not self.controls

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
        "_blank_waits",
        "_counters",
        "_fingerprinter",
        "_last",
        "_observed_ms",
        "_polls",
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
        self._armed: tuple[int, DomSnapshot, bool, str | None] | None = None
        self._rests = 0
        self._polls = 0
        self._rested_ms = 0.0
        self._blank_waits = 0
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

    @property
    def polls(self) -> int:
        """How many times an unchanged page was re-read while a mutation's effect was
        watched for (:data:`EFFECT_BUDGET_MS`). Page-script reads with no capture, so
        they are in ``counters.detections`` and not in ``counters.captures``."""
        return self._polls

    @property
    def blank_waits(self) -> int:
        """How many times a blank page was sat out; see :meth:`_sit_out`. Counted apart
        from :attr:`rests`, which are waits a policy's move armed."""
        return self._blank_waits

    def rest_after(
        self,
        actions: int,
        basis: DomSnapshot,
        *,
        waited: bool = False,
        operation: str | None = None,
    ) -> None:
        """Arm ONE coming observation to be taken once the move has been ANSWERED.

        An acting policy calls this as it answers: ``basis`` is the screen it decided
        on, ``actions`` how many controller actions its move performs - the explorer
        observes after each, and the one worth waiting for is the LAST, which is the
        frame the next decision is made on - and ``operation`` what the move was. What
        happens to that frame is upstream's three-way split (``0da4053``), by what the
        move did to the page:

        * it reads EXACTLY as ``basis`` did, after a mutation (:data:`EFFECT_BUDGET_MS`):
          the page is WATCHED - re-read every :data:`EFFECT_POLL_MS` until it differs or
          the operation's budget is spent. No quiet window; that constant says why.
        * the ADDRESS changed (:data:`ARRIVAL_REST_MS`): one quiet window, because a
          navigation may still be materializing its content. Also applied to a frame the
          watch landed on, when what landed was a redirect.
        * it changed IN PLACE (:data:`CHANGED_REST_MS`): one quiet window, because an
          overlay's first frame is half-built.

        An unchanged page after a scroll or a wait (:data:`IMMEDIATE_OPERATIONS`, or
        ``waited``) is handed over as it is. ``operation=None`` is a caller that did not
        say - every call site written before the watch existed - and gets what it got
        then: one :data:`NO_CHANGE_REST_MS` quiet window on an unchanged page. That is
        deliberately also what the driver's fresh look at a ``DONE`` claim wants, which is
        a wait armed as not one: an unchanged page there is the ORDINARY case, and
        watching it would cost every finished run three seconds.

        Armed per move rather than switched on, because every other reader of this
        perceiver must stay untaxed: a warm replay, the admission gate's rest loop and a
        skill's ``wait_for_text`` all observe in a loop, and a quiet window inside each
        read is what ``AGENTS.md`` forbids ``_settle`` for. Re-arming replaces whatever
        was armed, so a move that stopped early cannot leave a wait behind for long.
        """
        self._armed = (max(int(actions), 1), basis, waited, operation)

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
        if self._rested(controller, snapshot, shot):
            # Capture AGAIN, then read: the frame and the controls must be one moment,
            # and the frame taken before the wait - or the watch - is the moment being
            # replaced.
            shot = controller.capture()
            self._counters.captures += 1
            snapshot = self._read(controller, shot)
            self._counters.detections += 1
        for sat_out in range(BLANK_SIT_OUTS):
            if not snapshot.blank or not self._sit_out(controller, snapshot, sat_out + 1):
                break
            shot = controller.capture()
            self._counters.captures += 1
            snapshot = self._read(controller, shot)
            self._counters.detections += 1
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

    def _rested(self, controller: Controller, snapshot: DomSnapshot, shot: Screenshot) -> bool:
        """Whether this frame was the armed one AND time was then spent on the page, so
        that the caller must capture and read again."""
        if self._armed is None:
            return False
        remaining, basis, waited, operation = self._armed
        if remaining > 1:
            self._armed = (remaining - 1, basis, waited, operation)
            return False
        self._armed = None
        started = time.monotonic()
        watched = False
        if snapshot.url == basis.url and snapshot.digest == basis.digest:
            budget = None if waited else EFFECT_BUDGET_MS.get(operation or "")
            if budget is not None:
                snapshot = self._watch(controller, basis, shot, budget, started)
                watched = True
                if snapshot.url == basis.url:
                    # Upstream's shape: what the watch found is handed over at once. A
                    # quiet window here is the wait the watch replaced - on a page that
                    # animates its answer it is the whole cap, after the answer.
                    return True
            elif waited or operation in IMMEDIATE_OPERATIONS:
                return False
        if snapshot.url != basis.url:
            why, (quiet, cap) = "arrived", ARRIVAL_REST_MS
        elif snapshot.digest != basis.digest:
            why, (quiet, cap) = "changed", CHANGED_REST_MS
        else:
            why, (quiet, cap) = "unchanged", NO_CHANGE_REST_MS
        quiesce = getattr(controller, "quiesce", None)
        if not callable(quiesce):
            return watched
        began = time.monotonic()
        quiesce(quiet, cap)
        spent = (time.monotonic() - began) * 1000.0
        self._rests += 1
        self._rested_ms += spent
        log.info("dom.rest", why=why, waited_ms=round(spent), controls=len(snapshot.controls))
        return True

    def _watch(
        self,
        controller: Controller,
        basis: DomSnapshot,
        shot: Screenshot,
        budget_ms: float,
        started: float,
    ) -> DomSnapshot:
        """Re-read an unchanged page until it differs from ``basis`` or ``budget_ms`` is
        spent; the last snapshot read. Upstream's effect poll.

        Each look is the page script ALONE - no capture, as upstream polls with
        ``screenshot=False``: the digest is what is being asked about, and the one frame
        that matters is the one :meth:`observe` captures after this returns, together with
        a read of its own, so the frame and the controls handed over are still one moment.
        ``shot`` only supplies a viewport size if the page omits its own. A read that
        FAILS mid-watch is a document being replaced, which is an answer: the watch stops
        and the read after it reports whatever is there.
        """
        deadline = started + budget_ms / 1000.0
        snapshot, polls, landed = basis, 0, False
        while time.monotonic() < deadline:
            time.sleep(EFFECT_POLL_MS / 1000.0)
            polls += 1
            try:
                snapshot = self._read(controller, shot)
            except (ControllerError, PerceptionError):
                landed = True
                break
            self._counters.detections += 1
            if snapshot.url != basis.url or snapshot.digest != basis.digest:
                landed = True
                break
        spent = (time.monotonic() - started) * 1000.0
        self._polls += polls
        self._rests += 1
        self._rested_ms += spent
        log.info("dom.rest", why="effect", waited_ms=round(spent), polls=polls, landed=landed)
        return snapshot

    def _sit_out(self, controller: Controller, snapshot: DomSnapshot, attempt: int) -> bool:
        """Wait once on a BLANK page (:attr:`DomSnapshot.blank`); ``False`` if this
        controller cannot wait, which hands the blank frame over as it is.

        Unlike :meth:`rest_after` this is NOT armed per move, and that is deliberate: the
        run's FIRST observation follows a navigation no policy made, and it is the one
        upstream measured this on - the first complete document was the shell, and the
        run's first and only decision was taken on it. It taxes no other reader, because
        it costs nothing on a page that has anything on it, and there is no warm replay,
        gate rest loop or ``wait_for_text`` for which a page with no text and no control
        is the frame it wanted. Inside :meth:`observe`, so the wait lands in
        :attr:`site_ms` with the other rests - the site made us wait, not a model.
        """
        quiesce = getattr(controller, "quiesce", None)
        if not callable(quiesce):
            return False
        started = time.monotonic()
        quiesce(*BLANK_REST_MS)
        spent = (time.monotonic() - started) * 1000.0
        self._blank_waits += 1
        log.info("dom.rest", why="blank", waited_ms=round(spent), attempt=attempt, url=snapshot.url)
        return True

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
        enter = raw.get("enter")
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
            can_press_enter=isinstance(enter, dict),
            enter_label=_clean(enter.get("label"))[:200] if isinstance(enter, dict) else "",
            page_height=float(raw.get("pageHeight") or 0.0),
            omitted_controls=int(raw.get("omitted") or 0),
            covered_controls=int(raw.get("covered") or 0),
            by_element_id={control.element_id: control for control in controls},
            context_ms=float(raw.get("contextMs") or 0.0),
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
        section=_clean(entry.get("section"))[:40] or None,
        opens=_clean(entry.get("opens"))[:40] or None,
        context=_clean(entry.get("context"))[:MAX_CONTEXT_CHARS] or None,
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

    :attr:`DomControl.context` is deliberately NOT in the seed. Seven *Add item to cart*
    buttons never collided here: each sits in its own card, so the position grid already
    tells them apart, and the counter catches two in one cell. What they lacked was a NAME
    a policy could read, and that is what context is. As an identity it would be worse
    than position on the one axis that matters: it is the text of the card, and a card
    ANSWERS its button - "Add" becomes a stepper, "1 in cart" appears - so an id built on
    it would rename the control for having been used, and every memory keyed on the id
    (the explorer's dead ends, ``JevDriver``'s spent moves) would file the same button
    twice. It is also capped per snapshot (:data:`MAX_DESCRIBED`), so which controls have
    one moves with the scroll offset. The id stays where the control IS.
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
  const MAX_DESCRIBED = %(max_described)d;
  const MAX_CONTEXT_CHARS = %(max_context_chars)d;

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

  // Where a control sits, NAMED and never scored, as upstream has it: a control in the
  // open dialog, the header or a form means something different from the same label in
  // the footer, and the policy can weigh that itself - a positional prior computed here
  // would be a guess baked into the observation. The NEAREST landmark wins, so a search
  // form inside a header is 'form', not 'header'.
  const LANDMARKS = '[role="dialog"],[role="search"],[role="banner"],[role="navigation"],' +
    '[role="main"],[role="contentinfo"],[role="complementary"],' +
    'dialog,header,nav,main,aside,footer,form';
  const ROLE_NAMES = {banner: 'header', navigation: 'nav', contentinfo: 'footer',
    complementary: 'aside'};
  const section = (el) => {
    const landmark = el.closest(LANDMARKS);
    if (!landmark) return null;
    const explicit = landmark.getAttribute('role');
    return explicit ? (ROLE_NAMES[explicit] || explicit) : landmark.tagName.toLowerCase();
  };

  // Whether text can be typed into it. One definition, because the ENTER offer below
  // has to mean by "a field" exactly what the control list means by it.
  const typeable = (el, rname) => !el.readOnly && el.getAttribute('aria-readonly') !== 'true' &&
    (['textbox', 'searchbox', 'spinbutton'].includes(rname) ||
      (rname === 'combobox' && ['INPUT', 'TEXTAREA'].includes(el.tagName)));

  // What surrounds a control, for when its own name does not identify it: upstream's
  // describe(). A store listing can show seven buttons all named "Add item to cart"; the
  // product name lives in the card around each one. Reads innerText, which forces
  // layout, so it runs only for the ambiguous ones and is capped.
  const describe = (el, label) => {
    for (let p = el.parentElement, i = 0; p && i < 4; p = p.parentElement, i++) {
      // Stop at the row or card holding this control. A container full of other controls
      // is a header or a list, and its text describes all of them equally - no help.
      if (p.querySelectorAll('a,button,input,select,textarea,[role="button"]').length > 3) break;
      const whole = (p.innerText || '').replace(/\\s+/g, ' ').trim();
      if (whole.length <= label.length + 3 || whole.length > 300) continue;
      const rest = whole.split(label).join(' ').replace(/\\s+/g, ' ').trim();
      if (rest.length > 2) return rest.slice(0, MAX_CONTEXT_CHARS);
    }
    return null;
  };

  const controls = [];
  const nodes = [];
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

    const editable = typeable(el, rname);
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
    const where = section(el);
    if (where) control.section = where;
    if (el.getAttribute('aria-haspopup')) control.opens = el.getAttribute('aria-haspopup');
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
    nodes.push(el);
  }
  const omitted = Math.max(0, controls.length - MAX_CONTROLS);
  controls.splice(MAX_CONTROLS);

  // Only labels shared by several controls need disambiguating, and only the first few:
  // past that the page is a list of near-identical rows and the index carries the same
  // information. Upstream counts its click and fill actions and not its selects; here
  // that is every control but a <select>. The label is the one Python will show
  // (DomControl.label: the name, else the role), whitespace-collapsed as _clean does it,
  // because it is compared against innerText collapsed the same way.
  const contextStart = performance.now();
  const labelOf = (c) => (c.name || c.role).replace(/\\s+/g, ' ').trim();
  const shared = {};
  controls.forEach((c, i) => {
    if (nodes[i].tagName !== 'SELECT') shared[labelOf(c)] = (shared[labelOf(c)] || 0) + 1;
  });
  let described = 0;
  for (let i = 0; i < controls.length && described < MAX_DESCRIBED; i++) {
    const label = labelOf(controls[i]);
    if (nodes[i].tagName === 'SELECT' || (shared[label] || 0) < 2) continue;
    const context = describe(nodes[i], label);
    if (context) { controls[i].context = context; described++; }
  }
  const contextMs = performance.now() - contextStart;

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

  // Some searches have no button at all and submit only on Enter, so a run can type a
  // query and then have no way to run it (upstream: GitHub, 120 steps and an exhausted
  // budget, 5 with this). Offered only when a NON-EMPTY field holds focus, so the key
  // press has a defined destination, and only past the same filters a field has to pass
  // to be listed at all: safe (never a password), enabled, and TYPEABLE. That last one
  // is narrower than upstream's bare INPUT/TEXTAREA test on purpose: a focused checkbox
  // is an INPUT whose value is "on", and Enter there is an implicit form submission
  // offered under the name of a field nobody typed in.
  const focused = document.activeElement;
  const enter = focused && ['INPUT', 'TEXTAREA'].includes(focused.tagName) &&
    focused.value && safe(focused) && !focused.matches(':disabled') &&
    !focused.closest('[aria-disabled="true"]') && typeable(focused, role(focused))
    ? {label: name(focused) || ''} : null;

  return {
    url: location.href,
    title: document.title,
    w: innerWidth,
    h: innerHeight,
    scrollY: scrollY,
    canGoBack: canGoBack,
    scroller: scroller,
    enter: enter,
    loading: document.readyState !== 'complete' ||
      !!document.querySelector('[aria-busy="true"],[role="progressbar"]'),
    pageHeight: document.documentElement.scrollHeight,
    text: words.join('\\n').slice(0, MAX_PAGE_TEXT),
    controls: controls,
    texts: texts,
    omitted: omitted,
    covered: covered,
    contextMs: contextMs,
  };
}
""".replace("%(max_controls)d", str(MAX_CONTROLS))
    .replace("%(max_text_nodes)d", str(MAX_TEXT_NODES))
    .replace("%(max_page_text)d", str(MAX_PAGE_TEXT))
    .replace("%(max_described)d", str(MAX_DESCRIBED))
    .replace("%(max_context_chars)d", str(MAX_CONTEXT_CHARS))
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

Upstream's ``enter`` and ``back`` ACTIONS are facts here (``enter``, ``canGoBack``), not
entries in a list: what is offered is the policy's business (:mod:`skillweaver.llm.jev_`),
and this script says only what is true of the page.

``describe()`` and the shared-label pass are upstream's (``cbf517a``) unchanged in every
number; the one adaptation is that upstream's pass reads nodes back out of its identity
cache and this one keeps them in an array beside the controls, having no such cache.

Text nodes come back WITH their rectangles, because this path needs ``Element``s. And
select options are carried although no select action exists - see
:mod:`skillweaver.llm.jev_`."""
