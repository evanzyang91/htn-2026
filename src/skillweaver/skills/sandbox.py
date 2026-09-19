"""``SkillRunner``: compiling and executing the code a model wrote for us.

**This is not a security boundary.** It is a guard rail against a careless
generation - a model that reaches for ``import os`` because that is how it has seen
a thousand scripts start, or writes a ``while`` loop with no exit. A determined
attacker who can choose the source text can get out of a restricted namespace in
CPython; that is a known property of the language, not a bug to be fixed here. If
skill code ever comes from somewhere we do not trust, this module is the wrong tool
and a real sandbox - a separate process with an OS-level jail - is the right one.
Against the failure mode we actually have, three cheap mechanisms are enough:

*A static scan before anything runs* (:func:`scan_code`). Imports, ``open``,
``eval``, ``exec``, ``getattr``, private and dunder attributes, ``global``, and any
free name that is not an allowlisted builtin are rejected with a
:class:`~skillweaver.errors.SandboxViolation` that NAMES what was attempted and the
line it was on, because that message goes straight back to the model that will
rewrite the skill.

*A namespace with nothing in it.* Execution globals hold the executing skill's own
definitions and :data:`SAFE_BUILTINS` - arithmetic, strings, sequences, comparison,
a handful of exceptions. No module globals of this file, no ``__import__``, no file
access. The static scan makes this belt-and-braces: nothing should ever reach it.

*Hard limits* (:class:`~skillweaver.skills.api.SkillLimits`). A step budget charged
by every action, a wall-clock timeout enforced by a line tracer so a ``while True``
is interrupted rather than hung, and a composition-depth cap so two skills calling
each other stop at depth three instead of exhausting the interpreter's stack.

Everything else about a run is reporting. A failure of any kind comes back as
``SkillResult(ok=False, error=..., trace=...)`` with the failing line and its source
text, and every execution - nested calls included - is folded into the store's
statistics through ``record_run``, so the library learns which skills are worth
keeping simply by being used::

    runner = SkillRunner(store)
    ctx = runner.context(controller, perceiver, graph=graph, domain="example.com")
    result = runner.run(store.get("pay_invoice", "example.com"), {"company": "Acme"}, ctx)
    result.ok, result.steps, result.ms, result.trace

Note on the tracer: installing it calls ``sys.settrace`` for the duration of a
top-level run, so a debugger or ``coverage`` attached to the same thread does not see
skill code. The previous trace function is restored afterwards.
"""

from __future__ import annotations

import ast
import builtins
import linecache
import sys
import time
import traceback
from collections.abc import Mapping
from types import CodeType
from typing import Any

from skillweaver.contracts import (
    Controller,
    GraphView,
    Perceiver,
    Skill,
    SkillContext,
    SkillResult,
    SkillStore,
)
from skillweaver.errors import (
    BudgetExceeded,
    ExpectationFailed,
    SandboxViolation,
    SkillNotFound,
)
from skillweaver.logging_ import get_logger
from skillweaver.skills.api import (
    RunLedger,
    SkillAPI,
    SkillLimits,
)

__all__ = [
    "BANNED_BUILTINS",
    "SAFE_BUILTINS",
    "SkillRunner",
    "scan_code",
]

log = get_logger(__name__)

_SAFE_BUILTIN_NAMES = (
    # arithmetic and numbers
    "abs", "bool", "divmod", "float", "int", "max", "min", "pow", "round", "sum",
    # strings and characters
    "chr", "format", "ord", "repr", "str",
    # sequences, sets and mappings
    "dict", "enumerate", "filter", "frozenset", "len", "list", "map", "range",
    "reversed", "set", "slice", "sorted", "tuple", "zip",
    # predicates
    "all", "any", "isinstance", "issubclass",
    # exceptions a skill may reasonably raise or catch
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError",
    "KeyError", "LookupError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
)  # fmt: skip

SAFE_BUILTINS: Mapping[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
"""The only builtins skill code can see, plus the ``True``/``False``/``None``
keywords the compiler handles itself.

Chosen by asking what a short UI procedure genuinely needs: compute a number,
compare it, slice a list of elements, format a string, raise or catch an ordinary
error. Conspicuously absent are ``print`` (use ``ctx.log``, which lands in the
trace), ``type``, ``object`` and ``super`` (each a step away from the class
hierarchy), ``getattr``/``setattr``/``hasattr`` (attribute access by computed name
would defeat the static scan) and everything touching files, imports or code."""

BANNED_BUILTINS: Mapping[str, str] = {
    "__import__": "import modules",
    "breakpoint": "enter a debugger",
    "compile": "compile code",
    "delattr": "reach attributes by name",
    "dir": "enumerate attributes",
    "eval": "evaluate code",
    "exec": "execute code",
    "exit": "stop the process",
    "getattr": "reach attributes by name",
    "globals": "reach module globals",
    "hasattr": "reach attributes by name",
    "help": "enter an interactive helper",
    "id": "read raw object addresses",
    "input": "read from stdin",
    "locals": "reach the local namespace",
    "memoryview": "reach raw memory",
    "object": "reach the class hierarchy",
    "open": "open files",
    "quit": "stop the process",
    "setattr": "reach attributes by name",
    "super": "reach the class hierarchy",
    "type": "reach the class hierarchy",
    "vars": "reach a namespace",
}
"""Names a model is likely to reach for, mapped to what it would be trying to do.
Rejected with that phrasing so the repair prompt says *why*, not just *no*."""


# --------------------------------------------------------------------------------------
# The static scan
# --------------------------------------------------------------------------------------


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the source itself binds anywhere: module-level and nested defs,
    classes, parameters, assignments, loop and ``with`` targets, comprehension
    variables, ``except ... as``, and match patterns.

    A deliberate over-approximation - it ignores scope, so a name bound in one
    function counts as bound in another. Being too permissive here can only produce a
    plain ``NameError`` at runtime, while being too strict would reject working code.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                bound.add(name)
            case ast.arg(arg=name):
                bound.add(name)
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                bound.add(name)
            case ast.ExceptHandler(name=str() as name):
                bound.add(name)
            case ast.MatchAs(name=str() as name) | ast.MatchStar(name=str() as name):
                bound.add(name)
            case ast.MatchMapping(rest=str() as name):
                bound.add(name)
    return bound


def _reject(what: str, node: ast.AST) -> SandboxViolation:
    line = getattr(node, "lineno", 0)
    return SandboxViolation(f"{what} (line {line})")


def scan_code(code: str, *, what: str = "skill code") -> ast.Module:
    """Parse ``code`` and reject everything skill code may not do, returning the tree.

    Useful on its own: skill synthesis can scan a generated skill before storing it,
    so a skill that could never run never enters the library.

    Raises:
        SandboxViolation: naming the attempt and its line number.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxViolation(f"{what} does not parse: {exc.msg} (line {exc.lineno})") from exc

    bound = _bound_names(tree)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                first = names[0].name if names else "?"
                raise _reject(f"skill code may not import modules: 'import {first}'", node)
            case ast.ImportFrom(module=module):
                raise _reject(
                    f"skill code may not import modules: 'from {module or '.'} import ...'", node
                )
            case ast.Global(names=names) | ast.Nonlocal(names=names):
                keyword = "global" if isinstance(node, ast.Global) else "nonlocal"
                raise _reject(
                    f"skill code may not use '{keyword} {', '.join(names)}': "
                    "module globals are not reachable",
                    node,
                )
            case ast.AsyncFunctionDef() | ast.Await() | ast.AsyncFor() | ast.AsyncWith():
                raise _reject("skill code may not be asynchronous; write it straight-line", node)
            case ast.Attribute(attr=attr) if attr.startswith("_"):
                raise _reject(
                    f"skill code may not reach the private attribute {attr!r}; "
                    "everything a skill needs is a public member of ctx",
                    node,
                )
            case ast.Name(id=name, ctx=ast.Load()) if name not in bound:
                if name in BANNED_BUILTINS:
                    raise _reject(
                        f"skill code may not use {name!r} to {BANNED_BUILTINS[name]}", node
                    )
                if name not in SAFE_BUILTINS:
                    raise _reject(
                        f"name {name!r} is not defined in the skill sandbox: "
                        "skill code reaches the world only through ctx",
                        node,
                    )
    return tree


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------


def _skill_filename(skill: Skill) -> str:
    """The pseudo-filename the skill's code object is compiled under, so its frames
    are recognizable in a traceback and its source is quotable in a trace."""
    return f"<skill {skill.domain}/{skill.name} v{skill.version}>"


def _remember_source(filename: str, code: str) -> None:
    """Teach ``linecache`` the skill's source so tracebacks carry real source lines.
    The ``None`` mtime means ``checkcache`` will not evict it."""
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)


def _skill_frames(exc: BaseException) -> list[traceback.FrameSummary]:
    """The traceback frames that belong to skill code, outermost first. Frames from
    this module and the API are dropped: a model repairing its skill should not be
    shown our internals."""
    return [
        frame
        for frame in traceback.extract_tb(exc.__traceback__)
        if frame.filename.startswith("<skill ")
    ]


class SkillRunner:
    """A :class:`~skillweaver.contracts.SkillRunner`: compile a skill, execute it
    against a :class:`~skillweaver.skills.api.SkillAPI`, report what happened.

    Args:
        store: Where skills are read from for ``ctx.call`` and where each run's
            outcome is recorded. ``None`` runs skills that are not in a library yet -
            what synthesis does while it is still repairing one - in which case
            ``ctx.call`` raises ``SkillNotFound`` and nothing is recorded.
        limits: Defaults for every context this runner builds.

    One runner is reusable across runs and caches compiled code objects. Each
    execution still gets a fresh globals dict, so a skill cannot leave state behind
    for the next run of itself.
    """

    __slots__ = ("_cache", "_limits", "_store")

    def __init__(self, store: SkillStore | None = None, *, limits: SkillLimits | None = None):
        self._store = store
        self._limits = limits if limits is not None else SkillLimits()
        self._cache: dict[tuple[str, str], CodeType] = {}

    def __repr__(self) -> str:
        return f"SkillRunner(store={type(self._store).__name__}, limits={self._limits})"

    @property
    def limits(self) -> SkillLimits:
        """The limits every context from :meth:`context` is built with."""
        return self._limits

    # -- building a context ----------------------------------------------------------------

    def context(
        self,
        controller: Controller,
        perceiver: Perceiver,
        *,
        graph: GraphView | None = None,
        domain: str = "",
        limits: SkillLimits | None = None,
    ) -> SkillAPI:
        """A fresh :class:`~skillweaver.skills.api.SkillAPI` wired to this runner.

        Build the context this way rather than directly: it is what connects
        ``ctx.call`` back to the runner, and it gives the context the ledger whose
        limits then cover the whole composition. One context per top-level run.
        """
        return SkillAPI(
            controller,
            perceiver,
            graph=graph,
            domain=domain,
            ledger=RunLedger(limits if limits is not None else self._limits),
            invoke=self._invoke,
        )

    # -- SkillRunner protocol --------------------------------------------------------------

    def run(self, skill: Skill, args: Mapping[str, Any], ctx: SkillContext) -> SkillResult:
        """Execute ``skill.code`` with ``args``, and its verifier when it has one.

        Every skill-level failure - a sandbox violation, a failed ``ctx.expect``, a
        limit, an exception, a verifier saying no - is REPORTED in the result rather
        than raised, because the caller is usually a planner that wants to try
        something else and a synthesizer that wants the trace.

        ``steps`` and ``ms`` cover nested calls too, and ``trace`` is this skill's
        slice of the run: its log lines, its actions and, on failure, the skill
        frames with their source.

        Raises:
            BudgetExceeded: the one exception allowed out, so an exhausted agent run
                stops instead of starting the next skill.
        """
        ledger = getattr(ctx, "ledger", None)
        if not isinstance(ledger, RunLedger):
            ledger = RunLedger(self._limits)
        top_level = ledger.depth == 0
        mark, steps_before = len(ledger.trace), ledger.steps
        started = time.perf_counter()

        try:
            value = self._timed(skill, args, ctx, ledger, top_level)
        except BudgetExceeded:
            raise
        except BaseException as exc:  # noqa: BLE001 - every skill failure is reported
            ms = (time.perf_counter() - started) * 1000
            frames = _skill_frames(exc)
            lineno = frames[-1].lineno if frames else None
            error = f"{type(exc).__name__}: {exc}"
            if lineno is not None:
                error += f" (line {lineno})"
            lines = [
                f"{frame.name}() line {frame.lineno}: {(frame.line or '').strip()}"
                for frame in frames
            ]
            log.info("skill.run", name=skill.name, domain=skill.domain, ok=False, error=error)
            return SkillResult(
                ok=False,
                value=None,
                steps=ledger.steps - steps_before,
                ms=ms,
                error=error,
                trace=tuple(ledger.trace[mark:]) + tuple(lines) + (error,),
            )
        else:
            ms = (time.perf_counter() - started) * 1000
            log.info("skill.run", name=skill.name, domain=skill.domain, ok=True, ms=round(ms, 1))
            return SkillResult(
                ok=True,
                value=value,
                steps=ledger.steps - steps_before,
                ms=ms,
                error=None,
                trace=tuple(ledger.trace[mark:]),
            )

    # -- execution -------------------------------------------------------------------------

    def _timed(
        self,
        skill: Skill,
        args: Mapping[str, Any],
        ctx: SkillContext,
        ledger: RunLedger,
        top_level: bool,
    ) -> Any:
        """:meth:`_execute` under the wall-clock tracer, which only the top-level run
        installs and which is always removed before the caller inspects the failure -
        so extracting a traceback cannot itself trip the timeout."""
        if not (top_level and ledger.limits.max_seconds > 0):
            return self._execute(skill, args, ctx, ledger)
        previous = sys.gettrace()
        sys.settrace(_deadline_tracer(ledger))
        try:
            return self._execute(skill, args, ctx, ledger)
        finally:
            sys.settrace(previous)

    def _invoke(self, ctx: SkillAPI, name: str, kwargs: Mapping[str, Any]) -> Any:
        """Back end of ``ctx.call``: resolve ``name`` in the calling skill's domain
        and execute it against the SAME context, so it shares the budget and trace.

        Failures propagate here rather than becoming a ``SkillResult``: the caller is
        skill code, which may want to catch them.
        """
        domain = ctx.domain
        if self._store is None:
            raise SkillNotFound(f"cannot call {name!r}: this runner has no skill store")
        return self._execute(self._store.get(name, domain), kwargs, ctx, ctx.ledger)

    def _execute(self, skill: Skill, args: Mapping[str, Any], ctx: Any, ledger: RunLedger) -> Any:
        """Compile and call one skill, recording the outcome whatever it is.

        The ``finally`` is the point: statistics accumulate as a side effect of
        running, including for a skill that failed, so ``SkillStats.successes`` over
        ``runs`` means what it says.
        """
        ledger.check_time()
        with ledger.descend(skill.name, skill.domain):
            ledger.note(f"call {skill.name}({_render_args(args)})")
            ok = False
            started = time.perf_counter()
            try:
                value = self._call(skill, args, ctx)
                self._verify(skill, value, ctx)
                ok = True
                return value
            finally:
                self._record(skill, ok, (time.perf_counter() - started) * 1000)

    def _call(self, skill: Skill, args: Mapping[str, Any], ctx: SkillContext) -> Any:
        namespace = self._namespace(skill)
        run = namespace.get("run")
        if not callable(run):
            raise SandboxViolation(f"skill {skill.name!r} does not define a callable run(ctx, ...)")
        return run(ctx, **dict(args))

    def _verify(self, skill: Skill, value: Any, ctx: SkillContext) -> None:
        """Run ``skill.verifier_code`` when it has one.

        A verifier saying no is an ``ExpectationFailed``, the same clean failure as
        ``ctx.expect``: the code ran, the world did not end up as promised.
        """
        if not skill.verifier_code:
            return
        namespace = self._namespace(skill, skill.verifier_code, suffix=" verify")
        verify = namespace.get("verify")
        if not callable(verify):
            raise SandboxViolation(
                f"verifier for {skill.name!r} does not define a callable verify(ctx, result)"
            )
        if not verify(ctx, value):
            raise ExpectationFailed(f"the verifier for {skill.name!r} rejected the result")

    def _namespace(self, skill: Skill, code: str | None = None, suffix: str = "") -> dict[str, Any]:
        """Execute one piece of the skill's source in a namespace holding only its own
        definitions and :data:`SAFE_BUILTINS`, and return that namespace."""
        source = skill.code if code is None else code
        filename = _skill_filename(skill) + suffix
        key = (filename, source)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = compile(scan_code(source, what=f"skill {skill.name!r}"), filename, "exec")
            self._cache[key] = compiled
        _remember_source(filename, source)
        namespace: dict[str, Any] = {
            "__builtins__": dict(SAFE_BUILTINS),
            "__name__": filename,
            "__doc__": None,
        }
        exec(compiled, namespace)  # noqa: S102 - the whole point; see the module docstring
        return namespace

    def _record(self, skill: Skill, ok: bool, ms: float) -> None:
        """Fold this execution into the store's statistics. A skill that is not in
        the library yet (synthesis is still repairing it) simply has nowhere to
        record, which is not a failure of the run."""
        if self._store is None:
            return
        try:
            self._store.record_run(skill.name, skill.domain, ok, ms)
        except SkillNotFound:
            log.debug("skill.record.skipped", name=skill.name, domain=skill.domain)


def _render_args(args: Mapping[str, Any]) -> str:
    """``args`` as they would be written at a call site, values truncated."""
    parts = []
    for name, value in args.items():
        text = repr(value)
        parts.append(f"{name}={text if len(text) <= 40 else text[:37] + '...'}")
    return ", ".join(parts)


_TRACE_STRIDE = 32
"""Trace events between clock reads. Small enough that a tight loop is noticed
immediately, large enough that the read is not the cost of running a skill."""


def _deadline_tracer(ledger: RunLedger) -> Any:
    """A ``sys.settrace`` function that interrupts skill code once the ledger's
    wall-clock limit is spent.

    This is what makes the timeout real: a budget checked only when a skill acts
    cannot stop ``while True: pass``, and a watchdog thread cannot interrupt CPython
    bytecode. The clock is read every ``_TRACE_STRIDE`` events rather than every one,
    which keeps a tight loop cheap while still noticing within microseconds of work.
    """
    counter = 0

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter % _TRACE_STRIDE == 0:
            ledger.check_time()
        return tracer

    return tracer
