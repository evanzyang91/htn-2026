"""Families: skills grouped by what they DO, not by how they were asked for.

Every other reuse decision here is made in words and they all fail together on a request
worded differently. The ACTION SIGNATURE is the key that is not: an ordered tuple of
``OPERATION(role)`` tokens with every label, value and parameter abstracted away, so
``TYPE_TEXT(text_field) -> CLICK(button) -> CLICK(link)`` reads the same whether the run
bought coffee on one site or a keyboard on another.

Operations come from the skill's own CODE, because the code is what will run: a
trajectory records a model finding its way, and one Walmart errand whose skill is four
actions took seventeen to record. ROLES come from the code where it names them and from
the recording only where the two line up action for action (:func:`derive_signature`); a
role neither can show stays ``?``, which matches anything, because an unknown is not a
difference. Three normalisations keep the stream about the WORKFLOW: focus-select-type
is ONE ``TYPE_TEXT``; scrolls, waits and pointer moves are dropped; a modifier chord is
dropped and a bare key kept, so ``PRESS(enter)`` survives and Cmd+A does not.

**Earned, never guessed**: :func:`derive_signature` runs only for a candidate that
already passed its own verifier and the critic, so a skill without a verifier belongs to
no family (:func:`earned`). **The same shape can be the opposite intent** - adding to a
cart and removing from one - which a signature cannot tell apart and is not asked to;
:func:`same_intent` is the second half of every family decision.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from skillweaver.contracts import (
    Action,
    Click,
    ElementKind,
    Observation,
    PressKey,
    Skill,
    Trajectory,
    TypeText,
)
from skillweaver.trajectory.render import moves_of

__all__ = [
    "INTENTS",
    "MAX_FAMILY_DISTANCE",
    "MIN_FAMILY_STEPS",
    "UNKNOWN_ROLE",
    "Subgoal",
    "derive_signature",
    "distance",
    "earned",
    "families",
    "head_verb",
    "intent_of",
    "leads_alike",
    "nearest_workflow",
    "relatives",
    "render",
    "same_family",
    "same_intent",
    "signature_from_code",
    "signature_from_trajectory",
    "skeleton",
    "tokens_of",
]

UNKNOWN_ROLE = "?"
"""The role of an element neither the code nor the recording could name."""

MAX_FAMILY_DISTANCE = 0.34
"""How far apart two signatures may be and still be one family, as a fraction of the
longer one (:func:`distance`). ``0.34`` admits one edit in three tokens.

Measured 2026-09-20 over the twelve signatures this project then had - five Wikipedia,
three walmart.com, three splitkb.com, one sandbox shop - all 66 pairs;
``scripts/measure_families.py`` prints the table and re-runs in a second. Fourteen pairs
are the same errand: eleven score 0.12 to 0.33, and the outliers are Walmart's
three-step "add X" against its five-step "add X, then open the cart" at 0.40, the
six-step sandbox shop against splitkb pixels at 0.42, and the sandbox shop against
Walmart's three-step skill at 0.50. Every pair that is a DIFFERENT errand scores >= 0.50.

So the two populations TOUCH at 0.50 and no cut separates them. ``0.34`` admits eleven
true pairs and no false one with 0.16 of room; ``0.45`` would admit thirteen with 0.05.
The lower is kept because a relative left out costs a slower run while a stranger let in
lends its sentence to an errand it does not perform. RELATIVE so length matters - one
edit is a sixth of a six-step workflow and the whole of a one-step one.

Two things the cut cannot fix, both handled elsewhere: **shape is not intent**
(:func:`same_intent`), and **one action is not a shape** - two different single-``CLICK``
errands are 0.00 apart, which is what :data:`MIN_FAMILY_STEPS` keeps out.
"""

MIN_FAMILY_STEPS = 2
"""The shortest signature that may belong to a family. See the second note on
:data:`MAX_FAMILY_DISTANCE` for the measured pair that set it."""

_ROLE_COST = 0.5
"""What it costs when two tokens agree on the operation and disagree on the role.
Half an edit: a site that renders its search control as a ``link`` where another uses
a ``button`` has built the same step differently, not a different step."""

_SUBMIT_COST = 0.5
"""What it costs to submit a field with Enter on one site and with its button on the
other. The same half, for the same reason."""

_KINDS = tuple(kind.value for kind in ElementKind)

_MODIFIERS = frozenset({"meta", "control", "ctrl", "alt", "shift", "cmd", "command"})

_LOOKUPS = frozenset({"find_text", "wait_for_text", "nearest", "by_kind", "best"})
"""The :class:`~skillweaver.contracts.ElementIndex` and ``ctx`` reads whose result a
skill goes on to act on."""

_TOKEN = re.compile(r"^([A-Z_]+)\((.*)\)$")


# --------------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------------


def _token(operation: str, role: str) -> str:
    return f"{operation}({role})"


def _parts(token: str) -> tuple[str, str]:
    match = _TOKEN.match(token)
    return (match.group(1), match.group(2)) if match else (token, UNKNOWN_ROLE)


def render(signature: Sequence[str]) -> str:
    """``signature`` as one line, for a log or a report."""
    return " -> ".join(signature) if signature else "(no signature)"


def _append(out: list[str], operation: str, role: str) -> None:
    """Add one token, folding focus-then-type into a single ``TYPE_TEXT``."""
    if operation == "TYPE_TEXT":
        if out and _parts(out[-1])[0] == "CLICK":
            role = _parts(out.pop())[1] if role == UNKNOWN_ROLE else role
        if role == UNKNOWN_ROLE:
            # Typing lands in a field whatever the detector called the box around it.
            role = ElementKind.text_field.value
    out.append(_token(operation, role))


def _press(out: list[str], keys: Sequence[str]) -> None:
    """A bare key is part of the workflow; a modifier chord is housekeeping."""
    names = [str(key).casefold() for key in keys]
    if not names or any(name in _MODIFIERS for name in names):
        return
    out.append(_token("PRESS", "+".join(names)))


# --------------------------------------------------------------------------------------
# From the code
# --------------------------------------------------------------------------------------


def signature_from_code(code: str) -> tuple[str, ...]:
    """The action signature of a skill's ``run``, read out of its source.

    Statements are walked in source order, branches included, so a fallback lookup
    contributes nothing and a conditional action contributes its token once. An action
    inside a loop also counts once: the signature says WHAT is done.

    ``()`` for source that does not parse or defines no ``run`` - "no signature", never
    an error, because a signature is an index over the library, not a condition of
    admission.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
        ),
        None,
    )
    if run is None:
        return ()
    reader = _CodeReader()
    reader.walk(run.body)
    return tuple(reader.tokens)


class _CodeReader:
    """Walks ``run`` in order, remembering which kind each looked-up variable holds."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self._kinds: dict[str, str] = {}

    def walk(self, body: Iterable[ast.stmt]) -> None:
        for statement in body:
            self._statement(statement)

    def _statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None:
                self._calls(value)
                kind = self._lookup_kind(value)
                for target in targets:
                    # The FIRST lookup names the role; what follows under an ``if not``
                    # is the same control reached another way.
                    if isinstance(target, ast.Name) and kind and target.id not in self._kinds:
                        self._kinds[target.id] = kind
            return
        if isinstance(node, ast.Expr):
            self._calls(node.value)
            return
        if isinstance(node, ast.Return) and node.value is not None:
            self._calls(node.value)
            return
        for name in ("test", "iter"):
            inner = getattr(node, name, None)
            if isinstance(inner, ast.expr):
                self._calls(inner)
        for name in ("body", "orelse", "finalbody"):
            inner_body = getattr(node, name, None)
            if isinstance(inner_body, list):
                self.walk(s for s in inner_body if isinstance(s, ast.stmt))
        for handler in getattr(node, "handlers", ()):
            self.walk(handler.body)

    def _calls(self, expression: ast.expr) -> None:
        """Every ``ctx.ctl.*`` call inside ``expression``, in evaluation order."""
        found = [n for n in ast.walk(expression) if isinstance(n, ast.Call)]
        found.sort(key=lambda n: (n.lineno, n.col_offset))
        for call in found:
            self._action(call)

    def _action(self, call: ast.Call) -> None:
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "ctl"
        ):
            return
        if func.attr in ("click", "double_click") and call.args:
            _append(self.tokens, "CLICK", self._role_of(call.args[0]))
        elif func.attr == "type_text":
            _append(self.tokens, "TYPE_TEXT", UNKNOWN_ROLE)
        elif func.attr == "press":
            keys = [a.value for a in call.args if isinstance(a, ast.Constant)]
            _press(self.tokens, [k for k in keys if isinstance(k, str)])

    def _role_of(self, target: ast.expr) -> str:
        """The kind of the variable a click is aimed at: ``add[0]`` -> ``add``."""
        node: ast.expr = target
        while isinstance(node, ast.Subscript | ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            return self._kinds.get(node.id, UNKNOWN_ROLE)
        if isinstance(node, ast.Call):
            return self._lookup_kind(node) or UNKNOWN_ROLE
        return UNKNOWN_ROLE

    def _lookup_kind(self, value: ast.expr) -> str | None:
        """The element kind a lookup asked for, or ``None`` when it named none."""
        node: ast.expr = value
        while isinstance(node, ast.Subscript):
            node = node.value
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            return None
        method = node.func.attr
        if method not in _LOOKUPS:
            return None
        explicit = [k.value for k in node.keywords if k.arg == "kind"]
        position = 0 if method == "by_kind" else 1
        if not explicit and len(node.args) > position:
            explicit = [node.args[position]]
        for argument in explicit:
            kind = _kind_named(argument)
            if kind:
                return kind
        if method == "best" and node.args:
            # ``best('Add to cart button for ' + product)``: the description is prose,
            # and the only kind it can vouch for is one it spells out.
            words = " ".join(
                c.value
                for c in ast.walk(node.args[0])
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ).casefold()
            spelled = [k for k in _KINDS if re.search(rf"\b{k.replace('_', '[ _]')}\b", words)]
            if len(spelled) == 1:
                return spelled[0]
        return None


def _kind_named(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if node.value in _KINDS else None
    if isinstance(node, ast.Attribute) and node.attr in _KINDS:
        return node.attr
    return None


# --------------------------------------------------------------------------------------
# From the recording
# --------------------------------------------------------------------------------------


def signature_from_trajectory(trajectory: Trajectory) -> tuple[str, ...]:
    """The action signature of a recorded run, from the moves that WORKED.

    A move the critic rejected fell short of what it claimed and a step the controller
    refused did nothing. The role of a click is the kind of the smallest element under
    its point on the screen it was aimed at.
    """
    worked = [
        (step.action, step.before)
        for move in moves_of(trajectory)
        if not move.rejected
        for step in move.steps
        if step.result.ok
    ]
    return tokens_of(worked)


def tokens_of(performed: Iterable[tuple[Action, Observation]]) -> tuple[str, ...]:
    """Signature tokens for actions that reached a screen, each paired with the
    observation it was aimed at. One definition, so what a recording is reduced to and
    what a live move is matched against a skeleton with cannot drift apart."""
    out: list[str] = []
    for action, before in performed:
        if isinstance(action, Click):
            hits = [e for e in before.elements if e.box.contains(action.point)]
            role = min(hits, key=lambda e: e.box.area).kind.value if hits else UNKNOWN_ROLE
            _append(out, "CLICK", role)
        elif isinstance(action, TypeText):
            _append(out, "TYPE_TEXT", UNKNOWN_ROLE)
        elif isinstance(action, PressKey):
            _press(out, action.keys)
    return tuple(out)


def derive_signature(code: str, trajectory: Trajectory | None = None) -> tuple[str, ...]:
    """The signature to store: the code's operations, with the recording's roles filled in
    where the code named none - and ONLY where that is not a guess.

    A role is borrowed when the recording's accepted moves have exactly the code's
    sequence of operations. Anything looser was measured wrong: on live splitkb.com a
    skill ending in ``click(add[0])`` was aligned against a recording whose accepted
    clicks were *Search*, the product link and *More info* - both *Add to cart* clicks
    had WORKED and been judged failed by a critic that cannot see an AJAX add - and the
    skill's last action took the role of a button it never presses. An unknown role costs
    nothing in :func:`distance`; a wrong one costs half an edit against every relative.
    """
    coded = list(signature_from_code(code))
    if trajectory is None or not any(_parts(t)[1] == UNKNOWN_ROLE for t in coded):
        return tuple(coded)
    recorded = signature_from_trajectory(trajectory)
    if [_parts(t)[0] for t in coded] != [_parts(t)[0] for t in recorded]:
        return tuple(coded)
    for index, theirs in enumerate(recorded):
        operation, role = _parts(coded[index])
        if role == UNKNOWN_ROLE:
            coded[index] = _token(operation, _parts(theirs)[1])
    return tuple(coded)


# --------------------------------------------------------------------------------------
# Distance and families
# --------------------------------------------------------------------------------------


def _substitution(left: str, right: str) -> float:
    if left == right:
        return 0.0
    (l_op, l_role), (r_op, r_role) = _parts(left), _parts(right)
    if l_op == r_op:
        return 0.0 if UNKNOWN_ROLE in (l_role, r_role) else _ROLE_COST
    submits = {("PRESS", "enter"), ("CLICK", ElementKind.button.value)}
    if {(l_op, l_role), (r_op, r_role)} == submits:
        return _SUBMIT_COST
    return 1.0


def distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Edit distance between two signatures as a fraction of the longer, ``0.0..1.0``.
    Two empty signatures are ``1.0`` apart, not ``0.0``: having earned nothing is not a
    resemblance."""
    if not left or not right:
        return 1.0
    previous = [float(j) for j in range(len(right) + 1)]
    for i, mine in enumerate(left, start=1):
        current = [float(i)]
        for j, theirs in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1.0,
                    current[j - 1] + 1.0,
                    previous[j - 1] + _substitution(mine, theirs),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def earned(skill: Skill) -> bool:
    """Whether ``skill`` may take part in a family at all.

    It needs a signature - which only a verifier-passed admission writes - AND it still
    needs the verifier, because that is what makes reuse self-correcting: a relative
    chosen wrongly fails its own check instead of reporting a fast, free success.
    """
    return bool(skill.action_signature) and bool(skill.verifier_code) and not skill.demoted_reason


def same_family(left: Skill, right: Skill) -> bool:
    """Whether two skills do the same thing, by shape. Intent is NOT judged here -
    that is :func:`same_intent`, and a caller that reuses across a family needs both."""
    if not (earned(left) and earned(right)):
        return False
    if min(len(left.action_signature), len(right.action_signature)) < MIN_FAMILY_STEPS:
        return False
    return distance(left.action_signature, right.action_signature) <= MAX_FAMILY_DISTANCE


def relatives(skill: Skill, library: Iterable[Skill]) -> list[Skill]:
    """Every OTHER skill in ``library`` that is in ``skill``'s family, nearest first,
    on any site. Ties break on ``(domain, name)`` so the order is reproducible."""
    found = [
        (distance(skill.action_signature, other.action_signature), other)
        for other in library
        if (other.name, other.domain) != (skill.name, skill.domain) and same_family(skill, other)
    ]
    found.sort(key=lambda pair: (pair[0], pair[1].domain, pair[1].name))
    return [other for _, other in found]


def families(library: Iterable[Skill]) -> list[list[Skill]]:
    """``library`` partitioned into families by single linkage, largest first.

    For a listing and for the measurement script. Decisions use :func:`relatives`, which
    asks the pairwise question directly and cannot chain two strangers together.
    """
    members = [skill for skill in library if earned(skill)]
    groups: list[list[Skill]] = []
    for skill in members:
        joined = [g for g in groups if any(same_family(skill, other) for other in g)]
        merged = [skill] + [other for g in joined for other in g]
        groups = [g for g in groups if g not in joined] + [merged]
    groups.sort(key=lambda g: (-len(g), g[0].domain, g[0].name))
    return groups


# --------------------------------------------------------------------------------------
# Intent
# --------------------------------------------------------------------------------------


INTENTS: Mapping[str, frozenset[str]] = {
    "acquire": frozenset("add put buy get purchase order grab pick want need".split()),
    "remove": frozenset(
        "remove delete take empty clear drop discard cancel erase undo subtract".split()
    ),
    "find": frozenset("search find look lookup query".split()),
    "open": frozenset("open go visit navigate show view read jump follow".split()),
    "change": frozenset("change set update edit rename switch toggle".split()),
    "send": frozenset("send submit post share reply".split()),
}
"""Verbs that mean the same ERRAND, by class. A closed list, and what is not on it is
not guessed at: two verbs in one class may stand in for each other - *buy me a box of*
for *add a box of* - and a verb in a different class, or on no list, is a different
errand as far as this module can tell."""

_OPENERS = frozenset(
    "please kindly can could would will you i we id like want need to help me us lets let "
    "just now then also hey ok okay".split()
)
"""Words a request may start with before it gets to its verb."""

_WORD = re.compile(r"[a-z]+")


def intent_of(word: str) -> str | None:
    """The intent class of one verb, or ``None`` when it is on no list."""
    word = word.casefold()
    return next((name for name, verbs in INTENTS.items() if word in verbs), None)


def head_verb(text: str) -> str | None:
    """The word a request leads with once politeness is skipped: its verb, in the
    imperative sentences tasks are written in. ``None`` for a sentence with no such
    word."""
    words = _WORD.findall(text.casefold())
    for index, word in enumerate(words):
        if word in _OPENERS:
            # "I want a box of tea" leads with its verb; "I want to add" does not.
            rest = words[index + 1 :]
            if word in ("want", "need") and (not rest or rest[0] != "to"):
                return word
            continue
        return word
    return None


def leads_alike(request: str, learned: str) -> bool:
    """Whether two sentences lead with the same verb, or with two of one
    :data:`INTENTS` class. A verb on no list matches only itself."""
    asked, known = head_verb(request), head_verb(learned)
    if asked is None or known is None:
        return False
    return asked == known or (intent_of(asked) is not None and intent_of(asked) == intent_of(known))


def same_intent(
    request: str, skill: Skill, *, outside: str | None = None, strict: bool = True
) -> bool:
    """Whether ``request`` asks for the errand ``skill`` performs. Declines when unsure.

    Two conditions: the request LEADS with the verb the skill was learned under or one of
    its :data:`INTENTS` class (a verb on no list matches only itself), and no verb of a
    DIFFERENT class appears elsewhere unless the learned sentence carries that same word
    - which refuses *add it, then remove the old one* while still matching *add X to the
    cart, then open the cart*.

    ``strict=False`` refuses only a KNOWN conflict. That is for a caller who supplied the
    values, and it exists because the strict setting broke a warm path that had always
    worked: *Confirm payment of the Acme Corp invoice* against a skill learned as *Pay
    the ... invoice*, where neither verb is on a list and "I do not know these verbs" is
    not "these are opposites".

    ``outside`` is the request with the bound argument's text cut out, so a product
    called *Clear Glass Set* is a value and not the verb *clear*.
    """
    learned = skill.provenance.task_text
    mine = intent_of(head_verb(learned) or "")
    if strict:
        if not leads_alike(request, learned):
            return False
    else:
        theirs = intent_of(head_verb(request) or "")
        if mine is not None and theirs is not None and mine != theirs:
            return False
    if mine is None:
        return True  # with no class of its own, nothing elsewhere can be its opposite
    spoken = set(_WORD.findall(learned.casefold()))
    for word in _WORD.findall((outside if outside is not None else request).casefold()):
        theirs = intent_of(word)
        if theirs is not None and theirs != mine and word not in spoken:
            return False
    return True


# --------------------------------------------------------------------------------------
# The skeleton a cold run is handed
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Subgoal:
    """One step of a relative's workflow, as something to aim at.

    ``token`` is the signature token; ``text`` is what an acting policy is shown. The
    text carries no label, value or URL from the site the relative was learned on, so it
    is as true on the new site as on the old one.
    """

    token: str
    text: str

    def matches(self, token: str) -> bool:
        """Whether a performed move is this step, by the family's own measure."""
        return _substitution(self.token, token) < 1.0


_PHRASES: Mapping[tuple[str, str], str] = {
    ("TYPE_TEXT", "*"): "type the value the errand names into the {role} that takes it",
    ("PRESS", "enter"): "submit what was just typed",
    ("PRESS", "*"): "press the {role} key",
    ("CLICK", "button"): "press the button that performs the next part of the errand",
    ("CLICK", "link"): "open the link that matches what the errand names",
    ("CLICK", "text_field"): "focus the field the errand needs",
    ("CLICK", "*"): "click the {role} that moves the errand forward",
}


def skeleton(signature: Sequence[str]) -> tuple[Subgoal, ...]:
    """``signature`` as ordered subgoals, for a cold run to aim at one at a time.

    A prior and never a rail: the Explorer follows a skeleton only while the moves that
    work keep matching it, and drops it the moment they stop.
    """
    out: list[Subgoal] = []
    for token in signature:
        operation, role = _parts(token)
        phrase = _PHRASES.get((operation, role)) or _PHRASES.get((operation, "*"))
        if phrase is None:
            continue
        named = "control" if role == UNKNOWN_ROLE else role.replace("_", " ")
        out.append(Subgoal(token, phrase.format(role=named)))
    return tuple(out)


MIN_SKELETON_STEPS = MIN_FAMILY_STEPS
"""A workflow shorter than this is not worth handing to a cold run: one step is the
errand itself, and saying it again in vaguer words helps nobody."""


def nearest_workflow(request: str, domain: str, library: Iterable[Skill]) -> Skill | None:
    """The stored skill whose workflow a COLD run for ``request`` should be shown.

    Nothing is bound, or the run would not be cold, so this cannot ask for much: the
    skill must have EARNED its signature, be at least :data:`MIN_SKELETON_STEPS` long
    and lead with the same kind of verb - a search workflow is no prior for a removal.
    Among those, same site wins, then most relatives (a workflow seen on three shops is
    a better guess about a fourth), then most successes.

    Being wrong here is cheap by construction - the explorer drops a skeleton the screen
    disagrees with - which is why this may be loose where :func:`same_intent` may not.
    """
    pool = [
        skill
        for skill in library
        if earned(skill)
        and len(skill.action_signature) >= MIN_SKELETON_STEPS
        and leads_alike(request, skill.provenance.task_text)
    ]
    if not pool:
        return None
    return min(
        pool,
        key=lambda skill: (
            skill.domain != domain,
            -len(relatives(skill, pool)),
            -skill.stats.successes,
            skill.domain,
            skill.name,
        ),
    )
