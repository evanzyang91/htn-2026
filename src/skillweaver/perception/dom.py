"""The DOM perceiver: ask the page what is on it, instead of looking at a picture of it.

A :class:`~skillweaver.contracts.Perceiver` that reads the browser's own accessibility
information - roles, accessible names, values, checked/expanded state and
``getBoundingClientRect()`` - and emits ordinary :class:`~skillweaver.contracts.Element`
values with ``source=dom``. It runs NO detector and NO OCR.

Read ``AGENTS.md`` first. This project's acting loop was pixels-only by design, and
:class:`~skillweaver.controllers.browser.BrowserGroundTruth` is documented as an offline
teacher the agent may not reach. This module is a DELIBERATE, authorized relaxation of
that rule for BROWSER USE ONLY, and it is a separate class from ``BrowserGroundTruth``
precisely so the relaxation stays visible: nothing here is reachable unless a run was
started with ``--perception dom``. The pixel path remains the default.

What this buys, and what it costs
---------------------------------

**It buys the whole OCR bill.** Reading is 84-97% of a pixel observation on every live
page tried (see :mod:`skillweaver.perception.ocr`), and this path does not read. It
still CAPTURES - :attr:`~skillweaver.contracts.Observation.screenshot` is part of the
contract, the fingerprinter hashes it, and the capture is the cheap part - but a frame
costs one screenshot and one ``page.evaluate`` instead of a screenshot, a YOLO forward
pass and an ONNX recognition batch per text line. Do not take that as a speed claim
without a number beside it; measure the two paths on the same page and quote the
counts, which is what :class:`~skillweaver.orchestrator.PerceptionCounts` is for.

**It costs everything the page declines to say.** Text baked into an image, a canvas, a
cross-origin iframe and a closed shadow root are all invisible here and all perfectly
visible to OCR. The two paths therefore see genuinely different screens, which is why
they keep separate skill libraries - see :mod:`skillweaver.perception_mode` for how, and
for the measurement that says a fingerprint will NOT catch the crossing on its own.

Two element populations, one list
---------------------------------

:meth:`DomPerceiver.observe` emits controls first and then visible text, both in reading
order, because both are needed by different consumers:

* **Controls** - anything actionable, from the same selector Jev uses. These carry a
  :class:`DomControl` in :attr:`DomPerceiver.last`, which is what a policy needs to
  build an indexed target table.
* **Text** - visible text nodes, as :attr:`~skillweaver.contracts.ElementKind.text`
  elements. The policy ignores these; ``ctx.see.find_text`` in stored skill code does
  not, and neither does
  :func:`~skillweaver.perception.fingerprint.structural_hash`, so dropping them would
  make a DOM screen an unusable thing to write a skill against.

Geometry is LOGICAL pixels throughout, as the contract requires: ``getBoundingClientRect``
already reports CSS pixels, so the conversion is at scale 1.0 and only the rounding has
to agree with everything else - which is why it goes through
:mod:`skillweaver.controllers._coords` like every other producer.

Off-screen is dropped on purpose
--------------------------------

A control whose centre is outside the viewport is not reported, exactly as in Jev's
``snapshot.js``. This path's actions are POINT-based - a click is delivered at the
element's box centre through the unchanged
:class:`~skillweaver.controllers.browser.BrowserController` - so an element the camera
could not see is an element the mouse cannot reach. Reporting it would offer a policy a
target that silently misses. Scrolling is how the rest of the page is reached, which is
why ``SCROLL_UP``/``SCROLL_DOWN`` are in the action space at all.
"""

from __future__ import annotations

import hashlib
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
    "MAX_CONTROLS",
    "MAX_PAGE_TEXT",
    "MAX_TEXT_NODES",
    "SNAPSHOT_ATTEMPTS",
    "SNAPSHOT_RETRY_MS",
    "DomControl",
    "DomOption",
    "DomPerceiver",
    "DomSnapshot",
    "ROLE_KINDS",
]

MAX_CONTROLS = 250
"""Controls reported from one frame, matching Jev's own cap.

A page with more actionable things than this in ONE viewport is a page whose extra
targets are decoration; the cap bounds the policy's target table, which is the thing
that grows the request. :attr:`DomSnapshot.omitted_controls` says when it bit, so a cap
that ever truncates a real page is findable rather than silent.
"""

MAX_TEXT_NODES = 400
"""Visible text elements reported from one frame.

Higher than the control cap because text is what a stored skill reads, and a dense
article legitimately has hundreds of lines. It is still a cap: the element list feeds
:class:`~skillweaver.contracts.ElementIndex`, whose ``best`` and ``find_text`` scan it
per query.
"""

SNAPSHOT_ATTEMPTS = 4
"""How many times :meth:`DomPerceiver._read` asks a document that answers ``null``.

See the loop for what that answer means and why the controller's own retry does not
cover it. Four, with :data:`SNAPSHOT_RETRY_MS` between, because a page that has
committed a navigation gets a body within a frame or two and one that never does is a
broken page worth reporting rather than waiting on.
"""

SNAPSHOT_RETRY_MS = 150.0
"""How long to wait between those attempts."""

MAX_PAGE_TEXT = 6000
"""Characters of visible page text carried on the snapshot, as in Jev's ``snapshot.js``.

This is the blob a policy is shown as page context, NOT the element list. It is capped
because it goes into a request body on every step.
"""

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
"""ARIA role to :class:`~skillweaver.contracts.ElementKind`.

The mapping is deliberately lossy in one direction: several roles collapse onto
``menu`` and ``text_field`` because :class:`~skillweaver.contracts.ElementKind` is the
vocabulary a PIXEL detector can support, and a kind the detector cannot emit would be a
kind stored skill code could only ever find on one path. An editable ``combobox`` is
re-mapped to ``text_field`` in :func:`_kind_of`, where its editability is known.
"""


@dataclass(frozen=True, slots=True)
class DomOption:
    """One selectable value of a ``<select>``. ``label`` is what a person reads."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class DomControl:
    """One actionable thing on the page, as the page describes it.

    This is the richness :class:`~skillweaver.contracts.Element` deliberately does not
    carry - ``Element`` is the source-agnostic type every consumer in this project is
    written against, and widening it would be a change to the shared surface. A policy
    that needs to know a checkbox is already ticked reads it from here instead, via
    :attr:`DomPerceiver.last`.

    Attributes:
        index: 1-based position in :attr:`DomSnapshot.controls`, which is the number an
            indexed target table quotes.
        element_id: The id this control's :class:`~skillweaver.contracts.Element` will
            carry as ``stable_id``, and therefore the id an
            :class:`~skillweaver.agent.explorer.ElementCatalog` files it under. This is
            the ONE join between a policy's choice and the grounding the explorer does,
            so it is computed once, here.
        role: The ARIA role, as ``snapshot.js`` resolves it.
        name: The accessible name - ``aria-labelledby``, then ``aria-label``, then a
            ``<label>``, then own text, then ``title`` or ``placeholder``.
        kind: ``role`` mapped onto :data:`ROLE_KINDS`.
        box: Bounding rectangle in LOGICAL pixels, clipped to the viewport.
        value: The field's current value, ``""`` when it has none.
        editable: Whether text can be typed into it.
        checked / selected / expanded: The matching ARIA state, or ``None`` when the
            page does not say. ``None`` means unknown and must not be read as ``False``.
        options: The selectable values, for a ``<select>`` only.
        scope_text: Text of the nearest enclosing row, list item, card or form, capped.
            What tells a policy that this ``Add`` button belongs to THAT product.
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
        """The name, falling back to the role - never empty, so a target table always
        has something to quote for an icon-only control."""
        return self.name or self.role


@dataclass(frozen=True, slots=True)
class DomSnapshot:
    """Everything one ``page.evaluate`` returned, beside the observation it produced.

    Attributes:
        url / title: The document's own.
        text: Visible page text, capped at :data:`MAX_PAGE_TEXT`.
        controls: Actionable elements in reading order, 1-based via
            :attr:`DomControl.index`.
        texts: Visible text runs as ready-made ``text`` elements. Not part of the
            policy's target table; see the module docstring on the two populations.
        viewport: The visible area in logical pixels.
        scroll_y / page_height: Where the page is scrolled to and how tall it is,
            which is what decides whether scrolling up or down is even offered.
        omitted_controls: How many controls :data:`MAX_CONTROLS` cut. See that constant.
        can_go_back: Whether this tab has a previous session-history entry, which is
            what decides whether going back is offered at all. The page's own answer,
            from the Navigation API; see the script for why ``history.length`` is not
            that answer. It counts only entries contiguous and SAME-ORIGIN with this
            one, so the ``about:blank`` a run starts from does not make it true and a
            back is never offered as a way off the site being explored.
        by_element_id: The controls again, keyed by
            :attr:`DomControl.element_id`. This is the lookup a policy does to turn the
            id it chose back into what it knows about that control.
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
    can_go_back: bool = False
    by_element_id: dict[str, DomControl] = field(default_factory=dict, compare=False, repr=False)

    @property
    def can_scroll_down(self) -> bool:
        """Whether there is page below the fold."""
        return self.scroll_y + self.viewport.h < self.page_height - 2

    @property
    def can_scroll_up(self) -> bool:
        """Whether there is page above the fold."""
        return self.scroll_y > 0


class DomPerceiver:
    """A :class:`~skillweaver.contracts.Perceiver` over the page's own control list.

    Fills :class:`~skillweaver.contracts.Observation` exactly as
    :class:`~skillweaver.orchestrator.ComposedPerceiver` does - same screenshot, same
    reading-order elements, same :class:`~skillweaver.contracts.ElementIndex`, same
    fingerprinter - so everything above ``Element`` is unaffected by which path
    produced it.

    Args:
        fingerprinter: Identifies the screen. Defaults to
            :class:`~skillweaver.perception.fingerprint.StateFingerprinter`, which is
            what every stored precondition was hashed with.
        counters: The tally to charge work to, shared with whatever else counts. A
            fresh one is made when not given. ``ocr_reads`` stays at zero on this path
            and that is the measurement, not an omission.

    Not thread-safe: :attr:`last` is one slot, so one perceiver drives one browser.
    """

    __slots__ = ("_counters", "_fingerprinter", "_last")

    def __init__(
        self,
        fingerprinter: Fingerprinter | None = None,
        *,
        counters: PerceptionCounters | None = None,
    ) -> None:
        self._fingerprinter = fingerprinter if fingerprinter is not None else StateFingerprinter()
        self._counters = counters if counters is not None else PerceptionCounters()
        self._last: DomSnapshot | None = None

    def __repr__(self) -> str:
        return f"DomPerceiver(counts={self._counters.snapshot()})"

    @property
    def counters(self) -> PerceptionCounters:
        """The running tally of everything this perceiver has done."""
        return self._counters

    @property
    def last(self) -> DomSnapshot | None:
        """The snapshot behind the most recent :meth:`observe`, or ``None`` before one.

        A policy reads this to get the roles, values and states that
        :class:`~skillweaver.contracts.Element` does not carry. It belongs to the LAST
        observation only; anything holding an older ``Observation`` must not assume
        this still describes it.
        """
        return self._last

    def observe(self, controller: Controller) -> Observation:
        """One frame: capture, ask the page, index, fingerprint.

        Raises:
            ControllerError: if the capture or the page script fails.
            PerceptionError: if the page answers with something unusable, or if
                fingerprinting fails.
        """
        shot: Screenshot = controller.capture()
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
        return observation

    def _read(self, controller: Controller, shot: Screenshot) -> DomSnapshot:
        """Run :data:`_SNAPSHOT_JS` on the controller's page and parse the result.

        The page script is handed to the controller through a narrow duck-typed hook
        (``evaluate``) rather than by importing
        :class:`~skillweaver.controllers.browser.BrowserController`, so a controller
        that can run page script simply has one and this module stays out of the
        controllers package.

        Raises:
            ControllerError: if the controller cannot run page script at all.
            PerceptionError: if the script returns something unusable.
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
            # ``null``, not an exception: the script's own first line answers a document
            # with no ``body`` yet, which is a commit that has landed and not finished.
            # The controller's retry cannot see this - it catches a DESTROYED execution
            # context, and this context is alive and nearly empty - so the wait is here.
            # Measured on live splitkb.com: a click that navigates hit it once in a run
            # and, before this loop existed, ended the run with a PerceptionError.
            if attempt + 1 < SNAPSHOT_ATTEMPTS:
                log.info("dom.snapshot.retry", attempt=attempt + 1, url=controller.url())
                time.sleep(SNAPSHOT_RETRY_MS / 1000.0)
        raise PerceptionError(
            f"the page snapshot script returned {type(raw).__name__}, not an object, "
            f"{SNAPSHOT_ATTEMPTS} times: the document has a window but still no body"
        )


# --------------------------------------------------------------------------------------
# Parsing the page's answer
# --------------------------------------------------------------------------------------


def _snapshot_from(raw: dict[str, Any], shot: Screenshot) -> DomSnapshot:
    """Build a :class:`DomSnapshot` from the script's raw result.

    Every field is read defensively: this is data crossing back out of a web page, and a
    page that returns nonsense should give a perception failure naming the field rather
    than a ``TypeError`` halfway up the explorer.

    Raises:
        PerceptionError: if the result cannot be read as a snapshot.
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
            page_height=float(raw.get("pageHeight") or 0.0),
            omitted_controls=int(raw.get("omitted") or 0),
            by_element_id={control.element_id: control for control in controls},
        )
    except PerceptionError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise PerceptionError(f"the page snapshot could not be read: {exc}") from exc


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

    Sharing ``seen`` is what guarantees an id is unique across BOTH populations, so the
    catalogue never falls back to a positional id for a control because a text run
    happened to hash the same way.
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
    """``role`` as an :class:`~skillweaver.contracts.ElementKind`. See :data:`ROLE_KINDS`.

    An editable ``combobox`` - the shape every site search box with autocomplete takes -
    is a ``text_field`` rather than a ``menu``, because that is what stored skill code
    asking ``ctx.see.by_kind(ElementKind.text_field)`` means by it.
    """
    if role == "combobox" and editable:
        return ElementKind.text_field
    return ROLE_KINDS.get(role, ElementKind.other)


def _elements_of(snapshot: DomSnapshot) -> tuple[Element, ...]:
    """The snapshot's controls and text as one reading-order element tuple.

    Controls keep the ``element_id`` they were given, so the id a policy chooses is the
    id the catalogue files it under. Text elements get their own hashed id from the same
    function, which cannot collide with a control's because the role is part of the seed.
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

    Seeded exactly like :func:`skillweaver.controllers.browser._stable_id` - kind, text
    and a 16-pixel position grid - so the same control keeps its name across observations
    of one screen while a label that shifts a pixel does not get a new one.

    Uniqueness matters more here than on the pixel path, because
    :class:`~skillweaver.agent.explorer.ElementCatalog` falls back to a POSITIONAL id
    whenever a ``stable_id`` is duplicated, and a positional id would break the join
    between the policy's choice and the catalogue. A duplicate therefore gets a counter
    suffix rather than being allowed to collide - two ``Add`` buttons 16 pixels apart in
    a product grid is an ordinary page, not a corner case.
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
    """An ARIA tri-state: ``True``, ``False``, or ``None`` for "the page did not say".

    ``None`` is never coerced to ``False``. "This checkbox is not ticked" and "this
    element has no checked state" are different facts, and a policy told the first when
    the second is true will happily tick something that was never a checkbox.
    """
    if isinstance(value, bool):
        return value
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    return None


# --------------------------------------------------------------------------------------
# The page script
# --------------------------------------------------------------------------------------

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

  return {
    url: location.href,
    title: document.title,
    w: innerWidth,
    h: innerHeight,
    scrollY: scrollY,
    canGoBack: canGoBack,
    pageHeight: document.documentElement.scrollHeight,
    text: words.join('\\n').slice(0, MAX_PAGE_TEXT),
    controls: controls,
    texts: texts,
    omitted: omitted,
  };
}
""".replace("%(max_controls)d", str(MAX_CONTROLS))
    .replace("%(max_text_nodes)d", str(MAX_TEXT_NODES))
    .replace("%(max_page_text)d", str(MAX_PAGE_TEXT))
)
"""The one page script this perceiver runs, adapted from ``jev-ultrafast/snapshot.js``.

Differences from Jev's, and why:

* **No node-identity cache and no freshness guards.** Jev keeps a ``WeakMap`` of DOM
  nodes so it can dispatch input at a node id and verify the node did not move between
  the decision and the click. This path acts by POINT through the unchanged
  :class:`~skillweaver.controllers.browser.BrowserController`, and the explorer
  re-observes after every single action
  (:class:`~skillweaver.agent.explorer._TapedController`), so the guard has nothing to
  protect: a stale target shows up as the next observation disagreeing, which the
  critic already judges.
* **Text nodes are returned with their rectangles.** Jev only needs a text blob for the
  model. This path needs ``Element``s, per the module docstring.
* **Select options are carried but no select action exists.** See
  :mod:`skillweaver.llm.jev_` on why ``SELECT`` is not offered on this branch.
"""
