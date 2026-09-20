"""``SkillRunner``: compiling and executing the code a model wrote for us.

**Not a security boundary** - a guard rail against a careless generation. A determined
attacker who chooses the source text can get out of a restricted namespace in CPython;
untrusted skill code needs a separate process with an OS-level jail instead.

Three mechanisms. A *static scan* before anything runs (:func:`scan_code`): imports,
``open``, ``eval``, ``exec``, ``getattr``, private and dunder attributes, ``global`` and
any free name that is not an allowlisted builtin are rejected with a SandboxViolation
NAMING the attempt and its line, because that message goes back to the model that will
rewrite it. A *namespace* holding only the skill's own definitions and
:data:`SAFE_BUILTINS`. And *hard limits* (:class:`~skillweaver.skills.api.SkillLimits`).

The wall-clock tracer's reach ends at Python: a call wedged inside a native library
executes no frames, so the clock is never read again. Structural, not a hole to patch
here - each native thing the sandbox reaches through bounds its own work and reports
failing to. Installing the tracer calls ``sys.settrace`` for a top-level run, so a
debugger on the same thread does not see skill code; the previous function is restored.
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
    "NOT_THE_SKILL",
    "SAFE_BUILTINS",
    "SkillRunner",
    "scan_code",
]

log = get_logger(__name__)

NOT_THE_SKILL = "perception failed, not the skill: "
"""Opens the ``error`` of a run that failed while an observation was failing.

A fixed prefix rather than a new field on ``SkillResult``, which is shared surface.
Everything that reads an error gets the cause in the first six words, and a reader that
only pattern-matches can test for this string."""

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
"""The only builtins skill code can see, plus the ``True``/``False``/``None`` keywords
the compiler handles itself.

Chosen by asking what a short UI procedure genuinely needs. Conspicuously absent are
``print`` (use ``ctx.log``), ``type``, ``object`` and ``super`` (each a step from the
class hierarchy), ``getattr``/``setattr``/``hasattr`` (attribute access by computed name
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
    """Every name the source itself binds anywhere.

    A deliberate over-approximation that ignores scope, so a name bound in one function
    counts as bound in another: being too permissive here can only produce a plain
    ``NameError`` at runtime, while being too strict would reject working code.
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

    Useful on its own: synthesis can scan a generated skill before storing it.

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
    """A SkillRunner: compile a skill, execute it against a SkillAPI, report what happened.

    Args:
        store: Where skills are read from for ``ctx.call`` and where each run's outcome
            is recorded. ``None`` runs skills not in a library yet - what synthesis does
            while repairing one - so ``ctx.call`` raises and nothing is recorded.
        limits: Defaults for every context this runner builds.

    Reusable across runs and caches compiled code objects. Each execution still gets a
    fresh globals dict, so a skill cannot leave state behind for the next run of itself.
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

        Never raises for a failing skill: every outcome, including a SandboxViolation or a
        blown limit, comes back as a :class:`~skillweaver.contracts.SkillResult` carrying
        the error, the failing line and the trace. The exception is an agent-level
        BudgetExceeded, which is re-raised because it is about the RUN, not the skill.

        A run whose observation failed is reported with the :data:`NOT_THE_SKILL` prefix
        and kept out of the skill's statistics: a skill whose eyes broke has not failed
        its task.
        """
        ledger = getattr(ctx, "ledger", None)
        if not isinstance(ledger, RunLedger):
            ledger = RunLedger(self._limits)
        top_level = ledger.depth == 0
        mark, steps_before = len(ledger.trace), ledger.steps
        blind_before = len(ledger.perception_failures)
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
            if len(ledger.perception_failures) > blind_before:
                error = NOT_THE_SKILL + error
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

        With one exclusion, and it is deliberate. A failure that happened while the
        EYES were broken - an OCR read abandoned, a screenshot that would not decode -
        says nothing about the skill, so it is not written down at all. Recording it
        as a failure would demote a skill for something it did not do, and recording
        it as a success would be a lie; the honest entry is no entry. A run that
        SUCCEEDED despite a failed observation is still recorded, because succeeding
        is evidence either way.
        """
        ledger.check_time()
        with ledger.descend(skill.name, skill.domain):
            ledger.note(f"call {skill.name}({_render_args(args)})")
            ok = False
            blind_before = len(ledger.perception_failures)
            started = time.perf_counter()
            try:
                value = self._call(skill, args, ctx)
                self._verify(skill, value, ctx)
                ok = True
                return value
            finally:
                if ok or len(ledger.perception_failures) == blind_before:
                    self._record(skill, ok, (time.perf_counter() - started) * 1000)
                else:
                    log.info(
                        "skill.run.blameless",
                        name=skill.name,
                        domain=skill.domain,
                        why=ledger.perception_failures[-1],
                    )

    def _call(self, skill: Skill, args: Mapping[str, Any], ctx: SkillContext) -> Any:
        namespace = self._namespace(skill)
        run = namespace.get("run")
        if not callable(run):
            raise SandboxViolation(f"skill {skill.name!r} does not define a callable run(ctx, ...)")
        return run(ctx, **dict(args))

    def _verify(self, skill: Skill, value: Any, ctx: SkillContext) -> None:
        """Run ``skill.verifier_code`` when it has one. A verifier saying no is an
        ``ExpectationFailed``, the same clean failure as ``ctx.expect``: the code ran, the
        world did not end up as promised."""
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

    What makes the timeout real FOR SKILL CODE: a budget checked only when a skill acts
    cannot stop ``while True: pass``, and a watchdog thread cannot interrupt CPython
    bytecode. The clock is read every ``_TRACE_STRIDE`` events, which keeps a tight loop
    cheap while still noticing within microseconds of work.

    It is also the whole of what this can do. Tracing is driven by the interpreter's own
    frame events, so a thread inside a native call produces none: a 12-task evaluation
    once sat at 98.8% CPU for 36 minutes inside ONNX Runtime's thread pool with this
    installed and never fired once. The bound on such a call lives where the call is made.
    """
    counter = 0

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal counter
        counter += 1
        if counter % _TRACE_STRIDE == 0:
            ledger.check_time()
        return tracer

    return tracer
