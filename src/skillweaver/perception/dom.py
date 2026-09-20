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
class DomSnapshot:
    """Everything one ``page.evaluate`` returned, beside the observation it produced.

    Attributes:
        url / title: The document's own.
        text: Visible page text, capped at :data:`MAX_PAGE_TEXT`.
        controls: Actionable elements in reading order, 1-based via ``DomControl.index``.
        texts: Visible text runs as ready-made ``text`` elements, not in the policy's
            target table.
        viewport: The visible area in logical pixels.
        scroll_y / page_height: What decides whether scrolling is offered.
        omitted_controls: How many controls :data:`MAX_CONTROLS` cut.
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
        """The snapshot behind the most recent :meth:`observe`, or ``None``.

        It belongs to the LAST observation only; anything holding an older
        ``Observation`` must not assume this still describes it.
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

Three differences from Jev's. No node-identity ``WeakMap`` or freshness guard: this path
acts by POINT and the explorer re-observes after every action, so a stale target shows up
as the next observation disagreeing. Text nodes come back WITH their rectangles, because
this path needs ``Element``s. And select options are carried although no select action
exists - see :mod:`skillweaver.llm.jev_`."""
