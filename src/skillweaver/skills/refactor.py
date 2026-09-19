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

Four rewrites, applied in this order:

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from skillweaver.contracts import (
    Click,
    Drag,
    Element,
    ElementKind,
    Move,
    Point,
    Scroll,
    Trajectory,
    TypeText,
)
from skillweaver.logging_ import get_logger

__all__ = [
    "COORDINATE_METHODS",
    "Hardening",
    "element_at",
    "harden",
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
    """

    code: str
    changes: tuple[str, ...] = ()
    added_params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    coordinates_replaced: int = 0
    lookups_added: int = 0
    navigation_lifted: tuple[str, ...] = ()
    renamed: Mapping[str, str] = field(default_factory=dict, hash=False)

    @property
    def changed(self) -> bool:
        """Whether the pass rewrote anything at all."""
        return bool(self.changes)

    def summary(self) -> str:
        """One line for a log or a repair prompt."""
        if not self.changes:
            return "hardening: nothing to change"
        return "hardening: " + "; ".join(self.changes)


# --------------------------------------------------------------------------------------
# Reading the trajectory
# --------------------------------------------------------------------------------------


def _action_points(action: Any) -> tuple[Point, ...]:
    if isinstance(action, Click | Move | Scroll):
        return (action.point,)
    if isinstance(action, Drag):
        return (action.start, action.end)
    return ()


def element_at(trajectory: Trajectory, x: int, y: int) -> Element | None:
    """The element the RECORDING shows at logical point ``(x, y)``, or ``None``.

    A step whose own action targeted exactly this point is consulted first, using
    the screen as it was BEFORE that action - that is the element the model meant.
    Failing that, every observation in the run is searched. Among candidates the
    smallest box wins, so a button inside a row beats the row.

    Coordinates are LOGICAL pixels, like everything else in this codebase.
    """
    point = Point(x, y)

    def smallest(elements: Sequence[Element]) -> Element | None:
        hits = [e for e in elements if e.box.contains(point)]
        return min(hits, key=lambda e: e.box.area) if hits else None

    for step in trajectory.steps:
        if any(p == point for p in _action_points(step.action)):
            found = smallest(step.before.elements)
            if found is not None:
                return found
    for step in trajectory.steps:
        for observation in (step.before, step.after):
            found = smallest(observation.elements)
            if found is not None:
                return found
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


def _lookup_statements(name: str, element: Element, verb: str) -> list[ast.stmt]:
    """``name = ctx.see...`` followed by the ``ctx.expect`` that makes a miss honest."""
    call, described = _lookup_call(element)
    why = f"no {described} on screen to {verb}"
    return [
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=call),
        ast.Expr(
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
        ),
    ]


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
                element = element_at(self.trajectory, *point)
                if element is None:
                    log.debug("harden.unknown_point", x=point[0], y=point[1])
                    continue
                name = _variable_name(element, self.taken)
                prefix.extend(_lookup_statements(name, element, method))
                node.args[position] = ast.Subscript(
                    value=ast.Name(id=name, ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                )
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
        for old, new in renames.items():
            self.changes.append(f"renamed the local {old!r} to {new!r}")


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
    body = pass_.strip_navigation(fn.body)
    rewritten: list[ast.stmt] = []
    for statement in body:
        rewritten.extend(pass_.replace_coordinates(statement))
    fn.body = rewritten or [ast.Pass()]
    pass_.lift_literals(fn)
    pass_.rename_opaque(fn)

    ast.fix_missing_locations(tree)
    hardened = Hardening(
        code=ast.unparse(tree) + "\n",
        changes=tuple(pass_.changes),
        added_params=dict(pass_.added_params),
        coordinates_replaced=pass_.coordinates,
        lookups_added=pass_.lookups,
        navigation_lifted=tuple(pass_.navigation),
        renamed=dict(pass_.renamed),
    )
    log.info(
        "skill.harden",
        run_id=trajectory.run_id,
        changes=len(hardened.changes),
        coordinates=hardened.coordinates_replaced,
        lookups=hardened.lookups_added,
    )
    return hardened
