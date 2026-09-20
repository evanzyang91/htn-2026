"""The hardening pass: what turns a model's first draft into a skill worth keeping.

A model that has just watched a successful run writes that run back literally - the
pixel it clicked, the URL it happened to start from, the company name it was given
this once. Code like that works exactly once: the next release moves the button four
pixels and the skill is a liability. This module rewrites the draft against the
trajectory it was written from, BEFORE the admission gate in
:mod:`skillweaver.skills.synthesize` ever runs it::

    hardened = harden(draft_code, trajectory, params={"company": {"type": "string"}})
    hardened.code             # the rewritten source
    hardened.changes          # one readable line per rewrite, for the run log
    hardened.added_params     # params to merge into the skill's schema

Seven rewrites, applied in this order:

*Fixed sleeps are removed.* ``ctx.ctl.press("Enter")`` followed by
``ctx.ctl.wait(2000)`` does not wait for the page: every action is SETTLED by the
controller before it returns - ``BrowserController._settle`` pauses and then waits for
the page to finish loading - so the wait sleeps on top of a wait that has already
happened. Measured on live Wikipedia, three such sleeps were 38% of a stored skill's
whole run time, and removing them halved warm replay while changing nothing else (the
measurement is in ``AGENTS.md``). The capability is not removed, only the reflex: a
wait for something no load event covers - an animation, a debounce, a spinner - is KEPT
when an adjacent ``ctx.log`` NAMES that thing, which is the same bargain a positional
lookup gets below. See :func:`reflex_waits`, which is the detector on its own.

*A read the skill INSISTS on is allowed to look twice.* The pass above removes a sleep
because the controller already settled the action; this one adds the wait that settle
cannot give. ``BrowserController._settle`` waits for the page's LOAD EVENT, and a
control the page answers without navigating fired that long ago - measured on live
splitkb.com, the click on "Add to cart" returned in 130ms with the document complete
and the old url still showing, while the cart it redirects to did not commit until
1170ms. A recording made at model speed never meets that gap; the skill replayed at code
speed walks straight into it and reads the page it is still standing on.

    So ``NAME = ctx.see.find_text(...)`` immediately followed by a ``ctx.expect`` on
    ``NAME`` - and only that shape, only after an action, and only once per action -
    becomes ``ctx.wait_for_text(...)``. The ``expect`` is the whole warrant: it is the
    skill declaring that the text MUST be there, which is exactly when looking again is
    right and never when the skill is merely asking what is on screen. A read that
    branches (``if ctx.see.find_text("Error"): ...``) expects nothing and is left alone,
    because waiting four seconds for something you hope is absent is a tax on every run.
    On the happy path the rewrite costs NOTHING: the first look is the observation the
    skill was about to make. See :func:`awaited_reads`, which is the detector on its own,
    and :data:`~skillweaver.skills.api.AWAIT_BUDGET_MS` for the measurement.

*Hardcoded navigation is lifted out.* A ``run`` that begins by typing a URL, or by
performing a ``Navigate``, is a skill that insists on arriving its own way. The
prologue is REMOVED from the body and reported; the screen it was reaching for is
recorded as the skill's precondition instead.

    Deliberately, this pass does NOT make the skill route itself through the site
    graph. Routing to a skill's precondition belongs to the planner, which already
    does it before invoking one (``Plan.steps`` interleaves route actions between
    ``SkillCall``s). A skill that re-routed inside its own body would duplicate that
    work and break composition - a skill called as a step must act from where it was
    put, not go somewhere first. Lifting navigation into the precondition is the
    architecture, not a shortcut; do not add ``ctx.graph`` routing here.

*Literal coordinates become perception lookups.* ``ctx.ctl.click(Point(400, 140))``
becomes a ``ctx.see`` query for the element the RECORDING shows at that point, a
``ctx.expect`` that it is on screen, and a click on what was found. This is the
rewrite that matters most: a skill full of raw pixels is a screenshot, not a skill.

*Values that clearly vary become parameters.* A string literal typed into a field,
which the trajectory shows was this run's data, becomes a parameter with the recorded
value as its default - so the skill generalizes without breaking its own replay.

*Names become readable.* A local bound to a perception lookup and called ``e`` or
``tmp`` is renamed after what it holds (``confirm_payment_button``).

*Positions become meanings.* ``ctx.see.by_kind("text")[1]`` is a skill that reaches
its search box by counting, and the count changes the moment an advert loads or OCR
reads one extra caption - this is the single commonest reason a stored skill fails to
replay on a real site. Every bare positional subscript into a perception result is
found (including through an intermediate variable) and re-anchored on something
nameable: the element's own text where the recording shows some, otherwise the
nearest labelled thing it sits in or beside. Where the recording genuinely offers no
anchor - an unlabelled checkbox on a page of unlabelled checkboxes - the lookup is
KEPT and a ``ctx.log`` line is injected saying so, because a skill that works
positionally and announces it is worth more than no skill. See
:func:`positional_lookups`, which is the detector on its own.

Everything here is AST-to-AST and the result is re-emitted with ``ast.unparse``,
which normalizes formatting and DROPS COMMENTS. That is the deliberate trade: a
uniform, re-parseable library beats a model's commentary. Nothing here validates or
executes the result - :func:`~skillweaver.skills.synthesize.Synthesizer.admit` does
that, and code this pass could not improve is passed through unchanged rather than
raised on, because the gate is what decides whether it is any good.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from skillweaver.contracts import (
    Box,
    Click,
    Drag,
    Element,
    ElementKind,
    Move,
    Observation,
    Point,
    Scroll,
    Trajectory,
    TypeText,
)
from skillweaver.logging_ import get_logger

__all__ = [
    "COORDINATE_METHODS",
    "AwaitedRead",
    "Hardening",
    "PositionalLookup",
    "ReflexWait",
    "awaited_reads",
    "element_at",
    "harden",
    "positional_lookups",
    "reflex_waits",
    "typed_texts",
]

log = get_logger(__name__)

COORDINATE_METHODS: Mapping[str, tuple[int, ...]] = {
    "click": (0,),
    "move": (0,),
    "scroll": (0,),
    "drag": (0, 1),
    "perform": (),
}
"""``ctx.ctl`` methods that take targets, and which positional arguments they are.
``perform`` takes an action object instead, which is handled separately."""

_INTERACTIVE = frozenset(
    {
        ElementKind.button,
        ElementKind.link,
        ElementKind.row,
        ElementKind.text_field,
        ElementKind.checkbox,
        ElementKind.radio,
        ElementKind.tab,
        ElementKind.menu,
    }
)
"""Kinds specific enough to be worth passing to ``find_text`` as a filter. For a
bare label or an unclassified blob the kind is as likely to be wrong at replay time
as right, and a wrong filter finds nothing where the text alone would have found it."""

_URL = re.compile(r"^(?:https?://|www\.|file://)\S+$", re.IGNORECASE)

_OPAQUE_NAME = re.compile(
    r"^(?:_\w*|[a-z]\d*|el\d*|els|elem\w*|tmp\d*|temp\d*|res\d*|ret\d*|val\d*|"
    r"obj\d*|btn\d*|txt\d*|item\d*|thing\d*|foo|bar)$"
)
"""Local names that say nothing about what they hold, and so are worth replacing."""

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")

_POSITIONAL_QUERIES = frozenset({"by_kind", "all"})
"""``ctx.see`` queries that name NOTHING about what they return. Subscripting one
picks an element by where it happens to sit in the perception order, and that order
moves with the page: one extra advert, one more caption read by OCR, a wider window."""

_NAMED_QUERIES = frozenset({"find_text", "best", "by_id", "nearest", "containing"})
"""``ctx.see`` queries that already say what they are looking for. ``result[0]`` on
one of these is "the best match for what I asked", not "the second thing on screen",
so it is left alone - this is what a re-anchored lookup is rewritten INTO."""

_ANNOUNCES_POSITION = re.compile(
    r"position|reading order|no readable|not readable|by index|ordinal", re.IGNORECASE
)
"""What a ``ctx.log`` line has to mention for a positional lookup to count as already
announced. The archive skill's own "sender name not readable" line matches, so a model
that owned up to its fallback is not made to own up twice."""

_NOT_AN_ACTION = frozenset({"wait", "supports"})
"""``ctx.ctl`` members that settle nothing. ``supports`` asks a question and touches no
page; ``wait`` is its own settle - ``BrowserController._deliver`` returns early on one
rather than settling after it - so a wait does not cover the wait that follows it."""

_ANNOUNCES_WAIT = re.compile(
    r"animat|transition|debounce|throttl|spinner|fade|slide|carousel|countdown|"
    r"typeahead|autocomplete|suggestion|toast|re-?render|poll",
    re.IGNORECASE,
)
"""What an adjacent ``ctx.log`` has to NAME for a fixed wait to survive this pass.

Every one of these is something the load event does not cover, which is the whole
question: the page arriving is already waited for, a menu finishing its slide is not.
Deliberately absent are "load", "navigate" and "page" - a wait explained by the thing
that has demonstrably already happened is the reflex this pass exists to remove, not an
exception to it."""


_ANCHOR_REACH = 240
"""How far, in LOGICAL pixels, a textless element may sit from the labelled thing used
to find it. Past that they are not the same row and the anchor would be a guess."""


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hardening:
    """What the pass produced, and what it did.

    Attributes:
        code: The rewritten source. Always parseable when the input was; the
            original string when the input did not parse or defined no ``run``.
        changes: One human-readable line per rewrite, in the order they were made.
            Empty when the draft was already hard.
        added_params: Parameters lifted out of literals, as ``name -> JSON schema``
            with the recorded value as ``default``. Merge into ``Skill.params``.
        coordinates_replaced: Literal coordinates that became perception lookups.
        lookups_added: ``ctx.see`` queries introduced.
        navigation_lifted: The navigation targets removed from the body.
        renamed: Opaque local names mapped to what they became.
        positions_anchored: Positional subscripts rewritten onto a named element.
        positions_announced: Positional subscripts the recording gave no anchor for,
            which were KEPT and made to announce themselves with a ``ctx.log`` line.
            A non-empty tuple is not a failure; it is the skill saying where it is
            thin, and :attr:`positions_unanchored` counts it.
        waits_removed: Fixed sleeps dropped because the action before them had
            already been settled. A caller that reports a rejection to the model must
            report these too: the model cannot see the code that was run, and a wait
            it needs and never learns was deleted is a repair loop with no exit.
        awaits_added: Reads that were allowed to look again while the page answers.
            The counterpart of :attr:`waits_removed`, and the opposite trade: a
            duration was taken away, a condition was put in.
    """

    code: str
    changes: tuple[str, ...] = ()
    added_params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    coordinates_replaced: int = 0
    lookups_added: int = 0
    navigation_lifted: tuple[str, ...] = ()
    renamed: Mapping[str, str] = field(default_factory=dict, hash=False)
    positions_anchored: int = 0
    positions_announced: tuple[str, ...] = ()
    waits_removed: tuple[ReflexWait, ...] = ()
    awaits_added: tuple[AwaitedRead, ...] = ()

    @property
    def positions_unanchored(self) -> int:
        """How many lookups still navigate by position, having said so."""
        return len(self.positions_announced)

    @property
    def changed(self) -> bool:
        """Whether the pass rewrote anything at all."""
        return bool(self.changes)

    def summary(self) -> str:
        """One line for a log or a repair prompt."""
        if not self.changes:
            return "hardening: nothing to change"
        return "hardening: " + "; ".join(self.changes)


@dataclass(frozen=True, slots=True)
class PositionalLookup:
    """One place a skill reaches for an element by counting instead of by naming.

    Attributes:
        query: The ``ctx.see`` method that produced the list - ``"by_kind"`` or
            ``"all"``. Both say nothing about WHICH element is wanted.
        kind: The kind ``by_kind`` was given, when it was a literal; ``None`` for
            ``all()`` and for a kind computed at runtime.
        index: The constant subscript. Negative counts from the end.
        via: The local the list was held in, when the subscript went through one
            (``rows = ctx.see.by_kind("row")`` ... ``rows[2]``); ``None`` when the
            query was subscripted directly.
        line: Line number in the source it was found in, 1-based.
    """

    query: str
    kind: str | None
    index: int
    via: str | None = None
    line: int = 0

    @property
    def call(self) -> str:
        """The query as it reads in source."""
        if self.query != "by_kind":
            return "ctx.see.all()"
        return f"ctx.see.by_kind({self.kind!r})" if self.kind else "ctx.see.by_kind(...)"

    @property
    def position(self) -> str:
        """The index in words: ``"number 2"``, or ``"1 from the end"``."""
        return f"number {self.index + 1}" if self.index >= 0 else f"{abs(self.index)} from the end"

    def __str__(self) -> str:
        if self.via:
            return f"{self.via}[{self.index}], from {self.call}"
        return f"{self.call}[{self.index}]"


@dataclass(frozen=True, slots=True)
class ReflexWait:
    """One fixed sleep a skill takes for something it has already been given.

    Attributes:
        ms: The literal duration, in milliseconds.
        after: The ``ctx.ctl`` action the wait follows, whose settle already waited
            for the page. This is what makes the wait redundant rather than merely
            long, and it is what the model is told when a repair is asked for.
        line: Line number in the source it was found at, 1-based.
    """

    ms: int
    after: str
    line: int = 0

    def __str__(self) -> str:
        return f"ctx.ctl.wait({self.ms}) after {self.after}()"


@dataclass(frozen=True, slots=True)
class AwaitedRead:
    """One read a skill required, which was allowed to look again while the page answers.

    Attributes:
        query: The text the read is looking for, when it was a literal; ``None`` when
            it was computed - a parameter, or a string the skill built.
        after: The ``ctx.ctl`` action the read follows. This is the control whose
            answer is being waited for, and the reason the read races at all.
        line: Line number in the source it was found at, 1-based.
    """

    query: str | None
    after: str
    line: int = 0

    def __str__(self) -> str:
        what = repr(self.query) if self.query is not None else "a computed string"
        return f"the read for {what} after {self.after}()"


# --------------------------------------------------------------------------------------
# Reading the trajectory
# --------------------------------------------------------------------------------------


def _action_points(action: Any) -> tuple[Point, ...]:
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


def _locate(trajectory: Trajectory, x: int, y: int) -> tuple[Element, Observation] | None:
    """:func:`element_at`, and the screen the element was found on.

    The screen matters to the anchoring pass: an element with no readable text can
    only be described by what sits around it, and "around it" is a fact about one
    observation, not about the run.
    """
    point = Point(x, y)

    def smallest(elements: Sequence[Element]) -> Element | None:
        hits = [e for e in elements if e.box.contains(point)]
        return min(hits, key=lambda e: e.box.area) if hits else None

    for step in trajectory.steps:
        if any(p == point for p in _action_points(step.action)):
            found = smallest(step.before.elements)
            if found is not None:
                return found, step.before
    for step in trajectory.steps:
        for observation in (step.before, step.after):
            found = smallest(observation.elements)
            if found is not None:
                return found, observation
    return None


def element_at(trajectory: Trajectory, x: int, y: int) -> Element | None:
    """The element the RECORDING shows at logical point ``(x, y)``, or ``None``.

    A step whose own action targeted exactly this point is consulted first, using
    the screen as it was BEFORE that action - that is the element the model meant.
    Failing that, every observation in the run is searched. Among candidates the
    smallest box wins, so a button inside a row beats the row.

    Coordinates are LOGICAL pixels, like everything else in this codebase.
    """
    found = _locate(trajectory, x, y)
    return found[0] if found is not None else None


def _reach(box: Box, point: Point) -> float:
    """Distance in LOGICAL pixels from ``point`` to ``box``; ``0.0`` when inside."""
    dx = max(box.x - point.x, 0, point.x - (box.x + box.w))
    dy = max(box.y - point.y, 0, point.y - (box.y + box.h))
    return float((dx * dx + dy * dy) ** 0.5)


def _labelled_neighbour(observation: Observation, element: Element) -> Element | None:
    """The labelled thing a textless ``element`` can be found FROM, or ``None``.

    A checkbox with no text is still findable when it sits in a row that reads
    something: "the checkbox nearest that row" survives the row moving, which its
    position in the checkbox list does not. Containment beats proximity - the row
    the control is IN is a stronger claim than the caption beside it - and past
    :data:`_ANCHOR_REACH` nothing is claimed at all.
    """
    centre = element.box.center
    best: tuple[int, float, int, Element] | None = None
    for other in observation.elements:
        if other is element or not other.text.strip():
            continue
        if other.box == element.box:
            continue
        inside = other.box.contains(centre)
        distance = 0.0 if inside else _reach(other.box, centre)
        if not inside and distance > _ANCHOR_REACH:
            continue
        key = (0 if inside else 1, distance, other.box.area, other)
        if best is None or key[:3] < best[:3]:
            best = key
    return best[3] if best is not None else None


def _at_position(observation: Observation, kind: str | None, index: int) -> Element | None:
    """What ``by_kind(kind)[index]`` - or ``all()[index]`` when ``kind`` is ``None`` -
    would have picked on this screen. ``Observation.elements`` is already in reading
    order, which is the order ``ctx.see`` returns these two queries in."""
    matching = [e for e in observation.elements if kind is None or e.kind.value == kind]
    if -len(matching) <= index < len(matching):
        return matching[index]
    return None


def typed_texts(trajectory: Trajectory) -> tuple[str, ...]:
    """Every distinct string the run typed, in order - this run's data, and so the
    first thing that should become a parameter rather than a literal."""
    seen: list[str] = []
    for step in trajectory.steps:
        if isinstance(step.action, TypeText) and step.action.text not in seen:
            seen.append(step.action.text)
    return tuple(seen)


def _field_near(trajectory: Trajectory, text: str) -> Element | None:
    """The text field that was on screen when ``text`` was typed, if there was one."""
    for step in trajectory.steps:
        if isinstance(step.action, TypeText) and step.action.text == text:
            fields = [e for e in step.before.elements if e.kind == ElementKind.text_field]
            if fields:
                return fields[0]
    return None


# --------------------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------------------


def _slug(text: str, words: int = 3) -> str:
    """A readable snake_case identifier fragment from element text."""
    found = [w.lower() for w in _WORD.findall(text)][:words]
    return "_".join(found)


def _variable_name(element: Element, taken: set[str]) -> str:
    """A name for the local holding a lookup of ``element``, unique within ``taken``."""
    base = _slug(element.text) or element.kind.value
    if element.kind in _INTERACTIVE and not base.endswith(element.kind.value):
        base = f"{base}_{element.kind.value}"
    base = base if base.isidentifier() else f"target_{element.kind.value}"
    name, n = base, 1
    while name in taken or not name.isidentifier():
        n += 1
        name = f"{base}_{n}"
    taken.add(name)
    return name


def _param_name(text: str, field_element: Element | None, taken: set[str]) -> str:
    """A name for the parameter a typed literal becomes."""
    base = ""
    if field_element is not None:
        base = _slug(field_element.text, words=2)
    base = f"{base}_text" if base else "typed_text"
    name, n = base, 1
    while name in taken or not name.isidentifier():
        n += 1
        name = f"{base}_{n}"
    taken.add(name)
    return name


# --------------------------------------------------------------------------------------
# Reading the draft
# --------------------------------------------------------------------------------------


def _ctl_method(node: ast.expr) -> str | None:
    """``"click"`` for ``ctx.ctl.click``, else ``None``."""
    match node:
        case ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="ctx"), attr="ctl"), attr=str() as method
        ):
            return method
    return None


def _ctx_method(node: ast.expr, member: str) -> str | None:
    """``"find_text"`` for ``ctx.see.find_text`` when ``member`` is ``"see"``."""
    match node:
        case ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="ctx"), attr=str() as got), attr=str() as method
        ) if got == member:
            return method
    return None


def _number(node: ast.expr) -> int | None:
    """The integer value of a numeric literal, negation included; else ``None``."""
    match node:
        case ast.Constant(value=bool()):
            return None
        case ast.Constant(value=int() | float() as value):
            return int(value)
        case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() | float() as value)):
            return -int(value)
    return None


def _literal_point(node: ast.expr) -> tuple[int, int] | None:
    """The logical point a coordinate literal denotes, in any shape a model writes it.

    ``Point(400, 140)``, ``(400, 140)``, ``[400, 140]`` and ``Box(20, 120, 760, 40)``
    (its center, as ``ActionSurface`` would take it) are all recognized. Anything
    else - a name, an expression, a lookup - is left alone: it is not a literal.
    """
    match node:
        case ast.Call(func=ast.Name(id="Point"), args=args, keywords=[]) if len(args) == 2:
            x, y = _number(args[0]), _number(args[1])
            return (x, y) if x is not None and y is not None else None
        case ast.Call(func=ast.Name(id="Point"), args=[], keywords=keywords):
            values = {k.arg: _number(k.value) for k in keywords if k.arg in ("x", "y")}
            if values.keys() == {"x", "y"} and None not in values.values():
                return int(values["x"]), int(values["y"])  # type: ignore[arg-type]
        case ast.Tuple(elts=elts) | ast.List(elts=elts) if len(elts) == 2:
            x, y = _number(elts[0]), _number(elts[1])
            return (x, y) if x is not None and y is not None else None
        case ast.Call(func=ast.Name(id="Box"), args=args, keywords=[]) if len(args) == 4:
            numbers = [_number(a) for a in args]
            if all(n is not None for n in numbers):
                x, y, w, h = numbers  # type: ignore[misc]
                return x + w // 2, y + h // 2
    return None


def _navigation_target(node: ast.stmt) -> str | None:
    """The URL a statement navigates to, when it is hardcoded navigation.

    Two shapes: typing a URL into whatever has focus (an address bar), and
    performing a ``Navigate`` - which skill code cannot even construct, but models
    write it anyway because it is what the trajectory shows.
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    method = _ctl_method(call.func)
    if method == "type_text" and call.args:
        match call.args[0]:
            case ast.Constant(value=str() as text) if _URL.match(text.strip()):
                return text.strip()
    if method == "perform" and call.args:
        match call.args[0]:
            case ast.Call(func=ast.Name(id="Navigate"), args=[ast.Constant(value=str() as url)]):
                return url
    return None


def _is_enter_press(node: ast.stmt) -> bool:
    """``ctx.ctl.press("Enter")`` - the other half of typing a URL."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    if _ctl_method(call.func) != "press":
        return False
    return (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and str(call.args[0].value).lower() == "enter"
    )


def _run_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    return None


def _bound_names(fn: ast.FunctionDef) -> set[str]:
    names = {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef) and node is not fn:
            names.add(node.name)
    return names


_BLOCK_FIELDS = ("body", "orelse", "finalbody")
"""Statement lists a compound statement owns. A prefix belongs in the block the
statement that needs it lives in, not hoisted to the top of ``run``: hoisting a
lookup out of an ``if`` makes it run when it should not, and out of a ``for`` makes
it run once when the screen changes every pass."""


def _see_query(node: ast.expr) -> tuple[str, str | None] | None:
    """``("by_kind", "row")`` for ``ctx.see.by_kind("row")``; ``None`` for anything
    that is not a ``ctx.see`` call. The kind is ``None`` when it is not a literal."""
    if not isinstance(node, ast.Call):
        return None
    method = _ctx_method(node.func, "see")
    if method is None:
        return None
    kind: str | None = None
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        kind = node.args[0].value
    return method, kind


def _blocks(statement: ast.stmt) -> Iterator[tuple[object, str, list[ast.stmt]]]:
    """Every statement list ``statement`` owns, as ``(owner, field name, block)``.
    Assigning back through ``setattr(owner, name, ...)`` replaces it in place."""
    for name in _BLOCK_FIELDS:
        block = getattr(statement, name, None)
        if isinstance(block, list) and block and all(isinstance(s, ast.stmt) for s in block):
            yield statement, name, block
    for handler in getattr(statement, "handlers", ()):
        yield handler, "body", handler.body


def _own_nodes(statement: ast.stmt) -> list[ast.AST]:
    """Every node ``statement`` owns, stopping at the blocks it merely contains."""
    found: list[ast.AST] = []
    for name, value in ast.iter_fields(statement):
        if name in _BLOCK_FIELDS or name == "handlers":
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ast.AST):
                found.extend(ast.walk(item))
    return found


def _track_sources(statement: ast.stmt, sources: dict[str, tuple[str, str | None]]) -> None:
    """Record what each local was last bound to, so ``rows[2]`` can be read as the
    query that filled ``rows``. A name bound to anything else stops being tracked -
    a stale entry would make this pass rewrite the wrong expression."""
    if not isinstance(statement, ast.Assign):
        for node in _own_nodes(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                sources.pop(node.id, None)
        return
    query = _see_query(statement.value)
    for target in statement.targets:
        if not isinstance(target, ast.Name):
            for node in ast.walk(target):
                if isinstance(node, ast.Name):
                    sources.pop(node.id, None)
            continue
        if query is not None:
            sources[target.id] = query
        else:
            sources.pop(target.id, None)


def _positional_subscript(
    node: ast.AST, sources: Mapping[str, tuple[str, str | None]]
) -> PositionalLookup | None:
    """``node`` read as a bare positional pick out of a perception result, or ``None``.

    Both shapes the model writes are the same defect: ``ctx.see.by_kind("text")[1]``
    and ``rows = ctx.see.all()`` followed by ``rows[1]``. A subscript of a NAMED query
    (``find_text(...)[0]``) is not one - that index means "the best match" - and
    neither is a slice or a computed index, which are not fixed positions at all.
    """
    if not isinstance(node, ast.Subscript):
        return None
    index = _number(node.slice)
    if index is None:
        return None
    query = _see_query(node.value)
    if query is not None:
        method, kind = query
        if method in _POSITIONAL_QUERIES:
            return PositionalLookup(method, kind, index, line=getattr(node, "lineno", 0))
        return None
    if isinstance(node.value, ast.Name):
        source = sources.get(node.value.id)
        if source is not None and source[0] in _POSITIONAL_QUERIES:
            return PositionalLookup(
                source[0], source[1], index, via=node.value.id, line=getattr(node, "lineno", 0)
            )
    return None


def _scan_block(
    body: Sequence[ast.stmt],
    sources: dict[str, tuple[str, str | None]],
    found: list[PositionalLookup],
) -> None:
    """Collect positional lookups in source order, descending into nested blocks."""
    for statement in body:
        for node in _own_nodes(statement):
            lookup = _positional_subscript(node, sources)
            if lookup is not None:
                found.append(lookup)
        for _, _, block in _blocks(statement):
            _scan_block(block, dict(sources), found)
        _track_sources(statement, sources)


def positional_lookups(code: str) -> tuple[PositionalLookup, ...]:
    """Every place ``code``'s ``run`` reaches for an element by position, in order.

    This is the detector on its own, with no trajectory and no rewriting, so a test -
    or a reviewer - can ask one question of a skill: does it navigate by meaning?
    An empty tuple is the answer that matters. Source that does not parse, or that
    defines no ``run``, has no lookups to report rather than being an error; judging
    it is the admission gate's job.

    See :data:`_POSITIONAL_QUERIES` for what counts and :data:`_NAMED_QUERIES` for
    what deliberately does not.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    fn = _run_function(tree)
    if fn is None:
        return ()
    found: list[PositionalLookup] = []
    _scan_block(fn.body, {}, found)
    return tuple(found)


# --------------------------------------------------------------------------------------
# Reading the draft: sleeps
# --------------------------------------------------------------------------------------


def _settling_action(statement: ast.stmt) -> str | None:
    """The ``ctx.ctl`` action ``statement`` performs, or ``None``.

    An action is anything the controller delivers and then SETTLES: a click, a type,
    a press, a scroll, a navigate. The settle is what makes the wait after it
    redundant, so ``wait`` and ``supports`` do not count (:data:`_NOT_AN_ACTION`),
    and neither does an action buried in a larger statement - a result that was
    assigned or tested is a shape this pass declines to reason about.
    """
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    method = _ctl_method(statement.value.func)
    if method is None or method in _NOT_AN_ACTION:
        return None
    return method


def _wait_ms(statement: ast.stmt) -> int | None:
    """The duration of ``ctx.ctl.wait(<literal>)`` in milliseconds, or ``None``.

    A wait whose duration is COMPUTED is not a literal and is left alone: a number
    the skill worked out is a decision, and this pass only removes reflexes.
    """
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if _ctl_method(call.func) != "wait":
        return None
    if len(call.args) == 1 and not call.keywords:
        duration = _number(call.args[0])
    elif not call.args and len(call.keywords) == 1 and call.keywords[0].arg == "ms":
        duration = _number(call.keywords[0].value)
    else:
        return None
    return duration if duration is not None and duration >= 0 else None


def _announces_wait(statement: ast.stmt) -> bool:
    """Whether ``statement`` is a ``ctx.log`` that names what a wait is FOR.

    The message is searched whole, so a line built from pieces
    (``ctx.log("waiting for the " + name + " animation")``) counts. What has to be
    in it is :data:`_ANNOUNCES_WAIT`.
    """
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    match statement.value.func:
        case ast.Attribute(value=ast.Name(id="ctx"), attr="log"):
            pass
        case _:
            return False
    for node in ast.walk(statement.value):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _ANNOUNCES_WAIT.search(node.value):
                return True
    return False


def _page_neutral(statement: ast.stmt) -> bool:
    """Whether the settle of the action before ``statement`` still stands after it.

    Reading the screen, checking a condition and writing the trace all leave the page
    exactly where the last action left it, so a wait further down is still a wait for
    something that has already been waited for. Anything that reaches the page -
    ``ctx.ctl`` in any form, ``ctx.call`` running another skill - and any compound
    statement, whose branches this pass does not follow, ends that.
    """
    if not isinstance(statement, ast.Expr | ast.Assign | ast.AnnAssign | ast.AugAssign | ast.Pass):
        return False
    for node in ast.walk(statement):
        match node:
            case ast.Attribute(value=ast.Name(id="ctx"), attr="ctl" | "call"):
                return False
    return True


def _reflex_waits_in(body: Sequence[ast.stmt]) -> dict[int, ReflexWait]:
    """Which statements of ONE block are reflex waits, keyed by index in that block.

    A block is walked in order carrying the last action the controller settled.
    :func:`_page_neutral` statements are crossed without losing it; anything else
    clears it, and a block starts with nothing settled - a wait first inside an
    ``if`` is judged on its own branch, not on what ran before the branch was taken.
    A wait a neighbouring ``ctx.log`` explains is never a reflex, wherever it sits.
    """
    found: dict[int, ReflexWait] = {}
    settled: str | None = None
    for position, statement in enumerate(body):
        duration = _wait_ms(statement)
        if duration is not None:
            announced = (position > 0 and _announces_wait(body[position - 1])) or (
                position + 1 < len(body) and _announces_wait(body[position + 1])
            )
            if settled is not None and not announced:
                found[position] = ReflexWait(duration, settled, getattr(statement, "lineno", 0))
            continue
        action = _settling_action(statement)
        if action is not None:
            settled = action
        elif not _page_neutral(statement):
            settled = None
    return found


def _is_expect(call: ast.Call) -> bool:
    """Whether ``call`` is ``ctx.expect(...)``."""
    match call.func:
        case ast.Attribute(value=ast.Name(id="ctx"), attr="expect"):
            return True
    return False


def _required_read(statement: ast.stmt, following: ast.stmt | None) -> ast.Call | None:
    """The ``ctx.see.find_text`` call ``statement`` makes and then INSISTS on, or ``None``.

    Two shapes count here, and :func:`_acted_on_read` holds the third::

        found = ctx.see.find_text("Subtotal")        # bound, then
        ctx.expect(bool(found), "no cart")           # required by the very next line

        ctx.expect(bool(ctx.see.find_text("Subtotal")), "no cart")   # required inline

    The ``ctx.expect`` is what makes the read a requirement rather than a question,
    and a requirement is the only read worth looking twice for. A read whose result is
    branched on, logged, counted or returned is asking what is on screen right now, and
    the honest answer to that is what is on screen right now.

    A call that passes ``fuzzy`` is declined: ``ctx.wait_for_text`` does not take it,
    on purpose - a near match answers on the screen the wait was supposed to outlast.
    """

    def usable(call: ast.Call) -> ast.Call | None:
        if _ctx_method(call.func, "see") != "find_text":
            return None
        if any(keyword.arg == "fuzzy" for keyword in call.keywords):
            return None
        return call

    match statement:
        case ast.Expr(value=ast.Call() as call) if _is_expect(call):
            if not call.args:
                return None
            for node in ast.walk(call.args[0]):
                if isinstance(node, ast.Call) and (found := usable(node)) is not None:
                    return found
            return None
        case ast.Assign(targets=[ast.Name(id=name)], value=ast.Call() as call):
            if usable(call) is None or following is None:
                return None
            if not isinstance(following, ast.Expr) or not isinstance(following.value, ast.Call):
                return None
            expectation = following.value
            if not _is_expect(expectation) or not expectation.args:
                return None
            for node in ast.walk(expectation.args[0]):
                if isinstance(node, ast.Name) and node.id == name:
                    return call
            return None
        case _:
            return None


def _acts(statement: ast.stmt) -> str | None:
    """The last ``ctx.ctl`` action anywhere inside ``statement``, or ``None``.

    :func:`_settling_action` asks whether a statement IS an action; this asks whether
    one happens anywhere within it, a loop's body included, which is the question "has
    this skill touched the page yet?" needs answered.
    """
    last: str | None = None
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            method = _ctl_method(node.func)
            if method is not None and method not in _NOT_AN_ACTION:
                last = method
    return last


def _acted_on_read(statement: ast.stmt, rest: Sequence[ast.stmt]) -> ast.Call | None:
    """The ``ctx.see.find_text`` call whose result ``rest`` goes on to ACT on, or ``None``.

    The third shape, beside the two :func:`_required_read` accepts::

        adds = ctx.see.find_text("Add to cart - " + product, "button")   # bound, then
        if not adds:                                                     # maybe replaced,
            adds = ctx.see.best("Add to cart button for " + product)
        ctx.ctl.click(adds[0])                                           # and PRESSED

    Handing a read's result to ``ctx.ctl`` proves it is required as surely as a
    ``ctx.expect`` does - nothing can be pressed that was not found - and it is the shape
    that needs the second look most, because what follows an empty first look is a
    FALLBACK. Measured on live walmart.com, traced lookup by lookup: a result's title was
    on screen at +2.32s, this read for its button found nothing at +2.33s because the
    button had not hydrated, ``ctx.see.best`` - a ranking with a winner when nothing fits -
    answered with the header's "Cart contains 0 items" at +2.42s, and the skill clicked
    that. It passed one gate attempt and then 0 of 3 replays, the cart empty every time.

    The ``ctx.expect`` rule missed it twice over. Its expect was four fallbacks further
    down, not on the next line; and the read followed a wait that had already been
    converted, after which "every later read is a read of a screen that has arrived" -
    which is false of a page that answers in PHASES, titles and then buttons.
    """
    match statement:
        case ast.Assign(targets=[ast.Name(id=name)], value=ast.Call() as call):
            if _ctx_method(call.func, "see") != "find_text":
                return None
            if any(keyword.arg == "fuzzy" for keyword in call.keywords):
                return None
        case _:
            return None
    for later in rest:
        for node in ast.walk(later):
            if not isinstance(node, ast.Call) or _ctl_method(node.func) in (None, *_NOT_AN_ACTION):
                continue
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(argument)):
                    return call
    return None


def _awaited_reads_in(
    body: Sequence[ast.stmt], acted: str | None = None
) -> dict[int, tuple[AwaitedRead, ast.Call]]:
    """Which reads of ONE block race the control before them, keyed by index.

    ``acted`` is the last action the skill performed BEFORE this block, anywhere above
    it. Only :func:`_acted_on_read` consults it: a read about to be pressed is looked
    for again whenever the page has been touched at all, because a page that answers in
    phases is still answering after the first thing it said.

    The block is walked in order carrying the last action the controller settled,
    exactly as :func:`_reflex_waits_in` does, and for the same reason: a read is only
    racing if something just acted. Crossing a :func:`_page_neutral` statement keeps
    that action; anything else clears it, and so does converting a read - one action
    is answered once, and every later read on that screen is a read of a screen that
    has already arrived.
    """
    found: dict[int, tuple[AwaitedRead, ast.Call]] = {}
    settled: str | None = None
    for position, statement in enumerate(body):
        following = body[position + 1] if position + 1 < len(body) else None
        read = _required_read(statement, following) if settled is not None else None
        if read is not None:
            query = _string(read.args[0]) if read.args else None
            found[position] = (
                AwaitedRead(query, settled, getattr(statement, "lineno", 0)),
                read,
            )
            settled = None
            continue
        touched = settled or acted
        pressed = _acted_on_read(statement, body[position + 1 :]) if touched else None
        if pressed is not None:
            query = _string(pressed.args[0]) if pressed.args else None
            found[position] = (
                AwaitedRead(query, touched, getattr(statement, "lineno", 0)),
                pressed,
            )
            continue
        action = _settling_action(statement)
        if action is not None:
            settled = action
        elif not _page_neutral(statement):
            settled = None
        acted = _acts(statement) or acted
    return found


def _string(node: ast.expr) -> str | None:
    """``node`` as a string literal, or ``None`` when it is anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def awaited_reads(code: str) -> tuple[AwaitedRead, ...]:
    """Every read in ``code``'s ``run`` that would race the control before it.

    The detector on its own, with no trajectory and no rewriting, so one question can
    be asked of a skill: does it read the screen straight after acting on a control
    whose answer has not arrived? Source that does not parse, or that defines no
    ``run``, has no reads to report rather than being an error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    fn = _run_function(tree)
    if fn is None:
        return ()
    found: list[AwaitedRead] = []

    def descend(body: Sequence[ast.stmt], acted: str | None) -> None:
        found.extend(read for read, _ in _awaited_reads_in(body, acted).values())
        for statement in body:
            for _, _, block in _blocks(statement):
                descend(block, acted)
            acted = _acts(statement) or acted

    descend(fn.body, None)
    return tuple(sorted(found, key=lambda read: read.line))


def reflex_waits(code: str) -> tuple[ReflexWait, ...]:
    """Every fixed sleep in ``code``'s ``run`` that an action's settle already covers.

    This is the detector on its own, with no trajectory and no rewriting, so a test -
    or a reviewer - can ask one question of a skill: does it sleep for time the
    browser has already spent? An empty tuple is the answer that matters, and it is
    also the answer for a wait that was explained, which is a wait this project wants
    skills to keep. Source that does not parse, or that defines no ``run``, has no
    waits to report rather than being an error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    fn = _run_function(tree)
    if fn is None:
        return ()
    found: list[ReflexWait] = []

    def descend(body: Sequence[ast.stmt]) -> None:
        found.extend(_reflex_waits_in(body).values())
        for statement in body:
            for _, _, block in _blocks(statement):
                descend(block)

    descend(fn.body)
    return tuple(sorted(found, key=lambda wait: wait.line))


# --------------------------------------------------------------------------------------
# Building the replacement code
# --------------------------------------------------------------------------------------


def _attr(root: str, *path: str) -> ast.expr:
    node: ast.expr = ast.Name(id=root, ctx=ast.Load())
    for part in path:
        node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
    return node


def _lookup_call(element: Element) -> tuple[ast.expr, str]:
    """A ``ctx.see`` query for ``element``, and how to describe it in a failure.

    Text is the query when there is any, with the element's kind as a filter only
    when that kind is specific enough to help (see :data:`_INTERACTIVE`). With no
    text at all the kind is all there is to go on.
    """
    text = element.text.strip()
    if text:
        args: list[ast.expr] = [ast.Constant(value=text)]
        if element.kind in _INTERACTIVE:
            args.append(ast.Constant(value=element.kind.value))
        return ast.Call(func=_attr("ctx", "see", "find_text"), args=args, keywords=[]), repr(text)
    call = ast.Call(
        func=_attr("ctx", "see", "by_kind"),
        args=[ast.Constant(value=element.kind.value)],
        keywords=[],
    )
    return call, f"a {element.kind.value}"


def _expect_found(name: str, why: str) -> ast.stmt:
    """``ctx.expect(bool(name), why)`` - what makes an empty lookup honest instead of
    an ``IndexError`` two lines later."""
    return ast.Expr(
        value=ast.Call(
            func=_attr("ctx", "expect"),
            args=[
                ast.Call(
                    func=ast.Name(id="bool", ctx=ast.Load()),
                    args=[ast.Name(id=name, ctx=ast.Load())],
                    keywords=[],
                ),
                ast.Constant(value=why),
            ],
            keywords=[],
        )
    )


def _lookup_statements(name: str, element: Element, verb: str) -> list[ast.stmt]:
    """``name = ctx.see...`` followed by the ``ctx.expect`` that makes a miss honest."""
    call, described = _lookup_call(element)
    return [
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=call),
        _expect_found(name, f"no {described} on screen to {verb}"),
    ]


def _first(name: str) -> ast.expr:
    """``name[0]`` - the best match of a query that said what it was looking for."""
    return ast.Subscript(
        value=ast.Name(id=name, ctx=ast.Load()), slice=ast.Constant(value=0), ctx=ast.Load()
    )


def _nearest_statements(target: str, anchor: str, kind: ElementKind) -> list[ast.stmt]:
    """``target = ctx.see.nearest(anchor[0].box.center, kind)``, checked.

    This is how an element with no text of its own is still addressed by meaning: by
    the labelled thing it sits in. ``anchor`` must already be bound and checked.
    """
    centre = ast.Attribute(
        value=ast.Attribute(value=_first(anchor), attr="box", ctx=ast.Load()),
        attr="center",
        ctx=ast.Load(),
    )
    call = ast.Call(
        func=_attr("ctx", "see", "nearest"),
        args=[centre, ast.Constant(value=kind.value)],
        keywords=[],
    )
    return [
        ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=call),
        _expect_found(target, f"no {kind.value} beside the element it belongs to"),
    ]


def _log_statement(message: str) -> ast.stmt:
    """``ctx.log(message)`` - one line into the run trace."""
    return ast.Expr(
        value=ast.Call(func=_attr("ctx", "log"), args=[ast.Constant(value=message)], keywords=[])
    )


def _announces_position(statement: ast.stmt) -> bool:
    """Whether ``statement`` is a ``ctx.log`` that owns up to a positional lookup."""
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    match statement.value.func:
        case ast.Attribute(value=ast.Name(id="ctx"), attr="log"):
            pass
        case _:
            return False
    for argument in statement.value.args:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            if _ANNOUNCES_POSITION.search(argument.value):
                return True
    return False


def _replace_child(root: ast.AST, old: ast.AST, new: ast.expr) -> bool:
    """Swap ``old`` for ``new`` wherever it hangs off ``root``, by identity."""
    for parent in ast.walk(root):
        for name, value in ast.iter_fields(parent):
            if value is old:
                setattr(parent, name, new)
                return True
            if isinstance(value, list):
                for position, item in enumerate(value):
                    if item is old:
                        value[position] = new
                        return True
    return False


def _verb_of(statement: ast.stmt) -> str:
    """The ``ctx.ctl`` method a statement is for, to name it in a failure message."""
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            method = _ctl_method(node.func)
            if method is not None:
                return method
    return "use"


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------


class _Hardener:
    """One hardening of one draft. Not reusable; :func:`harden` builds one per call."""

    def __init__(self, trajectory: Trajectory, params: Mapping[str, Any]) -> None:
        self.trajectory = trajectory
        self.params = dict(params)
        self.changes: list[str] = []
        self.added_params: dict[str, Any] = {}
        self.renamed: dict[str, str] = {}
        self.navigation: list[str] = []
        self.coordinates = 0
        self.lookups = 0
        self.taken: set[str] = set()
        self.grounded: dict[str, tuple[Element, Observation]] = {}
        self.anchored = 0
        self.announced: list[str] = []
        self.owned_up = False
        self.waits: list[ReflexWait] = []
        self.awaits: list[AwaitedRead] = []

    # -- sleeps -------------------------------------------------------------------------

    def strip_reflex_waits(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Drop every fixed sleep the action before it had already been settled for.

        Runs FIRST, before navigation is lifted, for two reasons. A wait between a
        typed URL and the Enter that submitted it would otherwise hide that pair from
        :meth:`strip_navigation`, which reads them as adjacent. And a wait after a
        lifted navigation would, once the navigation is gone, look like the first
        statement of the block and be kept - a sleep for a page the skill no longer
        loads is the emptiest one there is.
        """
        for statement in body:
            for owner, name, block in _blocks(statement):
                setattr(owner, name, self.strip_reflex_waits(block))
        reflexes = _reflex_waits_in(body)
        for wait in reflexes.values():
            self.waits.append(wait)
            self.changes.append(
                f"dropped ctx.ctl.wait({wait.ms}) after {wait.after}(): the controller "
                f"settles every action and waits for the page to load, so those {wait.ms}ms "
                "were spent on top of a wait that had already happened"
            )
        return [statement for position, statement in enumerate(body) if position not in reflexes]

    # -- the wait that settling cannot give ----------------------------------------------

    def await_expected_reads(
        self, body: list[ast.stmt], acted: str | None = None
    ) -> list[ast.stmt]:
        """Let a read the skill REQUIRES look again while the control answers.

        Runs straight after the sleeps are stripped, and the pairing is the point: the
        sleep went because the controller had already waited for the page's load event,
        and this goes in because that event is exactly what a control answering in the
        background does not fire. A duration out, a condition in.

        Only the shapes :func:`_required_read` and :func:`_acted_on_read` accept, and
        only after an action - see :func:`_awaited_reads_in`. Statements are rewritten in
        place: nothing is added, nothing is removed, and the read's arguments are
        carried across untouched, so a parameterized query stays parameterized.
        """
        before = acted
        for statement in body:
            for owner, name, block in _blocks(statement):
                setattr(owner, name, self.await_expected_reads(block, before))
            before = _acts(statement) or before
        for read, call in _awaited_reads_in(body, acted).values():
            call.func = _attr("ctx", "wait_for_text")
            self.awaits.append(read)
            self.changes.append(
                f"let {read} wait for the page to answer (ctx.wait_for_text): the "
                f"controller settles {read.after}() by waiting for the load event, which "
                "a control that answers in the background never fires - and a skill runs "
                "far faster than the recording it was written from"
            )
        return body

    # -- navigation ---------------------------------------------------------------------

    def strip_navigation(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Drop hardcoded navigation, and the Enter that submitted a typed URL."""
        kept: list[ast.stmt] = []
        just_lifted = False
        for statement in body:
            url = _navigation_target(statement)
            if url is not None:
                self.navigation.append(url)
                self.changes.append(
                    f"lifted navigation to {url!r} out of the body: where the skill "
                    "starts is its precondition, not its first action"
                )
                just_lifted = True
                continue
            if just_lifted and _is_enter_press(statement):
                # The Enter belonged to the URL that has just gone; on its own it
                # would submit whatever the caller left in the field.
                self.changes.append("dropped the Enter that submitted the lifted URL")
                just_lifted = False
                continue
            just_lifted = False
            kept.append(statement)
        return kept

    # -- coordinates --------------------------------------------------------------------

    def replace_coordinates(self, statement: ast.stmt) -> list[ast.stmt]:
        """``statement`` with every literal coordinate replaced by a lookup, preceded
        by the lookups themselves."""
        prefix: list[ast.stmt] = []

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                visit(child)
            if not isinstance(node, ast.Call):
                return
            method = _ctl_method(node.func)
            if method is None or method not in COORDINATE_METHODS:
                return
            for position in COORDINATE_METHODS[method]:
                if position >= len(node.args):
                    continue
                point = _literal_point(node.args[position])
                if point is None:
                    continue
                found = _locate(self.trajectory, *point)
                if found is None:
                    log.debug("harden.unknown_point", x=point[0], y=point[1])
                    continue
                element = found[0]
                name = _variable_name(element, self.taken)
                prefix.extend(_lookup_statements(name, element, method))
                # Remembered for the anchoring pass: when the element had no text,
                # the lookup just written is a `by_kind` and so is itself positional.
                self.grounded[name] = (element, found[1])
                node.args[position] = _first(name)
                self.coordinates += 1
                self.lookups += 1
                self.changes.append(
                    f"replaced the literal coordinate ({point[0]}, {point[1]}) in "
                    f"{method}() with a ctx.see lookup for {element.text or element.kind.value!r} "
                    f"bound to {name!r}"
                )

        visit(statement)
        return [*prefix, statement]

    # -- parameters ---------------------------------------------------------------------

    def lift_literals(self, fn: ast.FunctionDef) -> None:
        """Turn each typed string literal the run supplied into a parameter with the
        recorded value as its default, so the skill generalizes and still replays."""
        recorded = typed_texts(self.trajectory)
        existing = set(self.params) | _bound_names(fn)
        lifted: dict[str, str] = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or _ctl_method(node.func) != "type_text":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            text = node.args[0].value
            if not isinstance(text, str) or text not in recorded or _URL.match(text.strip()):
                continue
            name = lifted.get(text)
            if name is None:
                name = _param_name(text, _field_near(self.trajectory, text), existing)
                lifted[text] = name
                self.added_params[name] = {
                    "type": "string",
                    "description": f"Text typed during the recorded run (was {text!r}).",
                    "default": text,
                }
                fn.args.kwonlyargs.append(ast.arg(arg=name))
                fn.args.kw_defaults.append(ast.Constant(value=text))
                self.changes.append(
                    f"lifted the typed literal {text!r} into the parameter {name!r}, "
                    f"defaulting to the recorded value"
                )
            node.args[0] = ast.Name(id=name, ctx=ast.Load())

    # -- names --------------------------------------------------------------------------

    def rename_opaque(self, fn: ast.FunctionDef) -> None:
        """Rename locals bound to a perception lookup that say nothing about what
        they hold. Only those: renaming anything else would be guessing."""
        renames: dict[str, str] = {}
        taken = _bound_names(fn) | self.taken
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not _OPAQUE_NAME.match(target.id):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            if _ctx_method(node.value.func, "see") is None:
                continue
            query = ""
            for argument in node.value.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    query = argument.value
                    break
            base = _slug(query) or "found"
            new = base if base.isidentifier() else "found"
            n = 1
            while new in taken:
                n += 1
                new = f"{base}_{n}"
            taken.add(new)
            renames[target.id] = new
        if not renames:
            return
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]
        self.renamed.update(renames)
        self.grounded = {renames.get(k, k): v for k, v in self.grounded.items()}
        for old, new in renames.items():
            self.changes.append(f"renamed the local {old!r} to {new!r}")

    # -- positions ----------------------------------------------------------------------

    def anchor_positions(self, fn: ast.FunctionDef) -> None:
        """Re-anchor every positional pick onto something nameable, or make it say so.

        Runs LAST, after renaming, so the locals it reasons about are the ones that
        survive, and after :meth:`replace_coordinates`, so the ``by_kind`` fallback
        that pass writes for a textless element is judged by the same rule as the
        model's own code.
        """
        self.taken |= _bound_names(fn)
        fn.body = self._anchor_block(fn.body, {})

    def _anchor_block(
        self, body: list[ast.stmt], sources: dict[str, tuple[str, str | None]]
    ) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        for statement in body:
            rewritten.extend(self._anchor_statement(statement, sources))
            for owner, name, block in _blocks(statement):
                setattr(owner, name, self._anchor_block(block, dict(sources)))
            rewritten.append(statement)
            if _announces_position(statement):
                self.owned_up = True
            _track_sources(statement, sources)
        return rewritten

    def _anchor_statement(
        self, statement: ast.stmt, sources: Mapping[str, tuple[str, str | None]]
    ) -> list[ast.stmt]:
        """What has to run before ``statement`` for its positional picks to be honest."""
        prefix: list[ast.stmt] = []
        for node in _own_nodes(statement):
            lookup = _positional_subscript(node, sources)
            if lookup is None:
                continue
            assert isinstance(node, ast.Subscript)
            resolved = self._resolve(lookup)
            replacement = (
                self._anchor_statements(*resolved, _verb_of(statement))
                if resolved is not None
                else None
            )
            if replacement is None:
                if not self.owned_up:
                    prefix.append(_log_statement(self._confession(lookup)))
                    self.owned_up = True
                self.announced.append(str(lookup))
                self.changes.append(
                    f"kept {lookup} - the recording names nothing it could be anchored "
                    "on - and made the skill log that it navigates by position"
                )
                continue
            statements, name, described = replacement
            prefix.extend(statements)
            _replace_child(statement, node, _first(name))
            self.anchored += 1
            self.lookups += 1
            self.changes.append(f"anchored {lookup} on {described} instead of its position")
        return prefix

    def _resolve(self, lookup: PositionalLookup) -> tuple[Element, Observation] | None:
        """Which element the recording says that position denotes, and on what screen.

        Two kinds of evidence, and nothing else. The run ACTED on the element that
        index picks out - that is what the model meant, and it is how a click the
        hardener itself grounded on a coordinate is recognized again. Or every screen
        in the run that has an element at that index shows the SAME labelled element,
        so the index is not doing any work. Anything less is a guess, and a guess
        rewritten into a skill is worse than the index it replaced.
        """
        if lookup.via is not None and lookup.via in self.grounded:
            return self.grounded[lookup.via]
        if lookup.query == "by_kind" and lookup.kind is None:
            return None
        kind = lookup.kind if lookup.query == "by_kind" else None
        seen: list[tuple[Element, Observation]] = []
        for step in self.trajectory.steps:
            points = _action_points(step.action)
            for observation, acted_on in ((step.before, points), (step.after, ())):
                element = _at_position(observation, kind, lookup.index)
                if element is None:
                    continue
                if any(element.box.contains(p) for p in acted_on):
                    return element, observation
                seen.append((element, observation))
        if not seen:
            return None
        first, where = seen[0]
        text = first.text.strip().casefold()
        if not text:
            return None
        same = all(e.kind is first.kind and e.text.strip().casefold() == text for e, _ in seen)
        return (first, where) if same else None

    def _anchor_statements(
        self, element: Element, observation: Observation, verb: str
    ) -> tuple[list[ast.stmt], str, str] | None:
        """The lookup that finds ``element`` by meaning, the local it lands in, and
        how to describe it in the change log. ``None`` when nothing names it."""
        text = element.text.strip()
        if text:
            name = _variable_name(element, self.taken)
            return _lookup_statements(name, element, verb), name, f"the text {text!r}"
        label = _labelled_neighbour(observation, element)
        if label is None or not label.text.strip():
            return None
        anchor = _variable_name(label, self.taken)
        target = _variable_name(element, self.taken)
        statements = _lookup_statements(anchor, label, f"find the {element.kind.value} beside")
        statements.extend(_nearest_statements(target, anchor, element.kind))
        described = f"the {element.kind.value} nearest {label.text.strip()!r}"
        return statements, target, described

    def _confession(self, lookup: PositionalLookup) -> str:
        """The ``ctx.log`` line a skill with no alternative has to carry."""
        what = lookup.kind or "element"
        return (
            f"no readable text to anchor on: taking {what} {lookup.position} in reading "
            "order, which moves if the page layout changes"
        )


def harden(
    code: str,
    trajectory: Trajectory,
    *,
    params: Mapping[str, Any] | None = None,
) -> Hardening:
    """Rewrite a generated skill's source against the run it was written from.

    Args:
        code: The model's draft. Source that does not parse, or that defines no
            module-level ``run``, is returned unchanged with no changes recorded -
            the admission gate is what rejects it, not this pass.
        trajectory: The recorded run. Every rewrite is grounded in it: the element
            at a coordinate, the text that was typed, the URL that was navigated to.
        params: The parameters the model declared, so lifted ones do not collide.

    Returns:
        A :class:`Hardening`. ``code`` is always a string of Python source, and
        ``changed`` says whether anything happened.

    Never raises: a draft this pass cannot improve is a draft the gate will judge.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        log.debug("harden.unparseable")
        return Hardening(code=code)
    fn = _run_function(tree)
    if fn is None:
        return Hardening(code=code)

    pass_ = _Hardener(trajectory, params or {})
    pass_.taken |= _bound_names(fn)
    body = pass_.strip_reflex_waits(fn.body)
    body = pass_.await_expected_reads(body)
    body = pass_.strip_navigation(body)
    rewritten: list[ast.stmt] = []
    for statement in body:
        rewritten.extend(pass_.replace_coordinates(statement))
    fn.body = rewritten or [ast.Pass()]
    pass_.lift_literals(fn)
    pass_.rename_opaque(fn)
    pass_.anchor_positions(fn)

    ast.fix_missing_locations(tree)
    hardened = Hardening(
        code=ast.unparse(tree) + "\n",
        changes=tuple(pass_.changes),
        added_params=dict(pass_.added_params),
        coordinates_replaced=pass_.coordinates,
        lookups_added=pass_.lookups,
        navigation_lifted=tuple(pass_.navigation),
        renamed=dict(pass_.renamed),
        positions_anchored=pass_.anchored,
        positions_announced=tuple(pass_.announced),
        waits_removed=tuple(pass_.waits),
        awaits_added=tuple(pass_.awaits),
    )
    log.info(
        "skill.harden",
        run_id=trajectory.run_id,
        changes=len(hardened.changes),
        coordinates=hardened.coordinates_replaced,
        lookups=hardened.lookups_added,
        anchored=hardened.positions_anchored,
        positional=hardened.positions_unanchored,
        waits=len(hardened.waits_removed),
        slept_ms=sum(wait.ms for wait in hardened.waits_removed),
        awaits=len(hardened.awaits_added),
    )
    return hardened
