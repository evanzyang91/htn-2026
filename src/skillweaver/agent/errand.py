"""An errand: one request naming SEVERAL items, done as one learned skill per item.

Nothing else in this project loops a skill over a list. A request naming three products is,
to the planner, one sentence that no stored skill binds: ``_from_template`` and
``_through_slot`` both refuse a sentence with more than one quotation ("which one is the
value is a guess"), the diff reader refuses a span holding ``and``, and the request falls to
the composer - one model call, a chain of at most ``compose.MAX_STEPS``, and no known route
from where the first call ENDS (the cart) back to where the second STARTS (the home page) -
or, on an empty library, to ONE cold exploration of the whole sentence, whose skill would
then be "add these three things", parameterised by nothing anybody asks twice.

So the list is split BEFORE the planner sees it, into single-item tasks in the SAME sentence
shape, because ``bind_args`` reads an argument only from a quoted slot whose anchoring words
still line up (``planner._anchored``)::

    Add "A", "B" and "C" to the cart
      -> Add "A" to the cart | Add "B" to the cart | Add "C" to the cart

and every item then goes through ``Agent.run`` - the same entry ``cli run`` uses - in ONE
browser session and ONE process, so the start-up and the first page load are paid once. The
first item may be cold and is learned through the ordinary admission gate; the rest are
meant to be warm, zero model calls in the action loop. Nothing here weakens the gate, the
warm critic or the report: a warm miss is printed as a miss.

Four things this module decides, each for a reason a run would otherwise teach.

* **One agent PER ITEM, one browser for all of them.** ``build_agent`` recalls the warm
  critic's end screen when it is BUILT, from the library as it stood then. An agent built
  before item 1 was learned holds a critic with nothing recalled, which escalates every
  later replay to a paid model call. Rebuilding is milliseconds; the browser, the model
  client and the policy are opened once.
* **Learning is allowed only while the errand has put nothing in the world.** The gate
  resets the world between its re-runs - on a shop, it EMPTIES THE CART - so a skill
  learned at item 3 would delete items 1 and 2. Once any item has succeeded, later cold
  items run with ``learn=False`` and the report says so.
* **The learned item is re-checked against the site.** The gate's last act may be a reset
  or a failed re-run, so after a learning item the site is asked whether the item is still
  there, and it is re-added ONCE through the warm path if it is not. That re-add is its
  own row in the report, never folded into the first.
* **No efficiency number without ground truth beside it.** ``ErrandReport.explain`` prints
  seconds per item only when a ``GroundTruth`` reader ran AND found every requested item
  exactly once; otherwise it prints why there is no number. Timings pass through
  ``eval.metrics._measured``, the project's one door for them, on the SITE's verdict per
  item and not the agent's.

The session is assembled from the orchestrator's own openers (``_open_world`` and friends),
as ``inspector/session.py`` already does, because ``Workbench.session`` opens a new browser
per call and ties one agent to one task. ``_check_orchestrator_seam`` names a rename at
start-up. A reset recipe reaches ``BrowserGroundTruth`` ONLY through ``_reset_for``, which
is the one call site allowed to hand it out.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from skillweaver.contracts import Budget, Controller, Navigate, TaskSpec
from skillweaver.errors import SkillWeaverError
from skillweaver.logging_ import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------------------

_QUOTED = re.compile(r"[\"“]([^\"“”]{1,120})[\"”]")
"""A double-quoted span, straight or curly - the planner's own definition of "the thing
the person means" (``planner._DOUBLE_QUOTED``). Single quotes are left alone here: an
apostrophe inside a product name would open a list that is not one."""

_SEPARATOR = re.compile(r"\s*(?:,|;|&|\band\b|\bplus\b|\bthen\b|,\s*and\b|,\s*then\b)\s*", re.I)
"""What may stand between two items of a list and nothing else. Anything more between two
quotations - *add "A" to the cart and remove "B"* - is two errands of different kinds, and
is refused rather than split."""

How = Literal["single", "quoted-list", "named-list", "undecomposable"]


@dataclass(frozen=True, slots=True)
class Decomposition:
    """An errand as single-item tasks.

    Attributes:
        errand: The person's own sentence, untouched.
        items: What each task is about, in the order asked, duplicates removed.
        tasks: One sentence per item - the errand's own head and tail around ONE quoted
            item, so it is the person's wording that reaches the recorder and becomes the
            stored ``Precedent``, never a model's rewrite.
        how: Which reader answered.
        why: For ``undecomposable``, what stopped it.
    """

    errand: str
    items: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    how: How = "undecomposable"
    why: str = ""

    def __bool__(self) -> bool:
        return bool(self.tasks)


def decompose(errand: str, *, name_items: Callable[[str], str] | None = None) -> Decomposition:
    """Split ``errand`` into single-item tasks, deterministically when it quotes its items.

    Args:
        name_items: The OPTIONAL fallback for an errand that quotes nothing: a callable
            returning the errand rewritten with each item in double quotes (the goal
            refinement writer, see :func:`refinement_namer`). Its WORDS are not used -
            only the items it names, and only when each one occurs verbatim in the
            person's own sentence, which is then split around them. A rewrite that
            invents a product name is refused rather than shopped for.
    """
    text = errand.strip()
    spans = list(_QUOTED.finditer(text))
    if len(spans) == 1:
        return Decomposition(text, (spans[0].group(1).strip(),), (text,), "single")
    if len(spans) > 1:
        return _split_quoted(text, spans)
    if name_items is None:
        return Decomposition(
            text, why="the errand quotes no item and no fallback reader was configured"
        )
    try:
        named = name_items(text)
    except SkillWeaverError as exc:
        return Decomposition(text, why=f"the fallback reader failed: {exc}")
    return _split_named(text, [m.group(1).strip() for m in _QUOTED.finditer(named or "")])


def _split_quoted(text: str, spans: Sequence[re.Match[str]]) -> Decomposition:
    for left, right in zip(spans, spans[1:], strict=False):
        between = text[left.end() : right.start()]
        if not _SEPARATOR.fullmatch(between):
            return Decomposition(
                text,
                why=f"{between.strip()!r} between two quoted items is more than a list "
                "separator, so this is several different errands and not one list",
            )
    head, tail = text[: spans[0].start()], text[spans[-1].end() :]
    items = _unique(m.group(1).strip() for m in spans)
    return Decomposition(
        text, items, tuple(f'{head}"{item}"{tail}' for item in items), "quoted-list"
    )


def _split_named(text: str, named: Sequence[str]) -> Decomposition:
    items = _unique(n for n in named if n)
    if len(items) < 2:
        return Decomposition(text, why="the fallback reader named fewer than two items")
    folded = text.casefold()
    found: list[tuple[int, int]] = []
    for item in items:
        at = folded.find(item.casefold())
        if at < 0 or folded.find(item.casefold(), at + 1) >= 0:
            return Decomposition(
                text,
                why=f"the fallback reader named {item!r}, which the errand itself does not "
                "say exactly once; a product the person never typed is not shopped for",
            )
        found.append((at, at + len(item)))
    order = sorted(range(len(items)), key=lambda i: found[i][0])
    for a, b in zip(order, order[1:], strict=False):
        if not _SEPARATOR.fullmatch(text[found[a][1] : found[b][0]]):
            return Decomposition(text, why="the named items do not sit in one plain list")
    head, tail = text[: found[order[0]][0]], text[found[order[-1]][1] :]
    own = tuple(text[found[i][0] : found[i][1]] for i in order)
    return Decomposition(text, own, tuple(f'{head}"{item}"{tail}' for item in own), "named-list")


def _unique(values: Any) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for value in values:
        seen.setdefault(value.casefold(), value)
    return tuple(seen.values())


_NAME_THE_ITEMS = (
    "This goal is a shopping errand naming several items. Keep the person's wording and put "
    "EACH item to obtain in double quotes, exactly as the person wrote it. Quote nothing else."
)


def refinement_namer(config: Any, url: str) -> Callable[[str], str]:
    """The goal-refinement writer as a ``name_items`` reader: the same
    ``OpenAITextWriter.rewrite_goal`` that ``SKILLWEAVER_REFINE_GOAL`` uses, given guidance
    that asks for quotation marks instead of a rewrite. One text-model call, made only for
    an errand that quotes nothing."""

    def name(errand: str) -> str:
        from skillweaver.llm.openai_ import OpenAITextWriter

        writer = OpenAITextWriter(
            api_key=config.openai_api_key,
            model=config.text_model,
            base_url=config.text_base_url,
            effort=config.text_effort,
            refine_model=config.refine_model,
        )
        return writer.rewrite_goal(errand, url, _NAME_THE_ITEMS)

    return name


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Line:
    """One line of what the SITE says it holds."""

    title: str
    quantity: int


GroundTruth = Callable[[Controller], Sequence[Line]]
"""Reads the world's own account of itself through the run's browser. Read-only."""

_CART_JS = """async () => {
  const reply = await fetch('/cart.js', {headers: {accept: 'application/json'}});
  if (!reply.ok) return JSON.stringify({error: reply.status});
  const cart = await reply.json();
  return JSON.stringify({
    item_count: cart.item_count,
    items: cart.items.map(i => ({title: i.product_title || i.title, quantity: i.quantity})),
  });
}"""


def shopify_cart(controller: Controller) -> Sequence[Line]:
    """A Shopify shop's cart, from its own ``GET /cart.js`` in the run's own session (so
    it is THIS browser's cart). One read-only fetch; never the checkout.

    Raises:
        SkillWeaverError: the page cannot be asked, or the shop did not answer.
    """
    evaluate = getattr(controller, "evaluate", None)
    if not callable(evaluate):
        raise SkillWeaverError("this controller cannot ask the page anything")
    try:
        raw = json.loads(evaluate(_CART_JS))
    except (TypeError, ValueError) as exc:
        raise SkillWeaverError(f"/cart.js did not answer with JSON: {exc}") from exc
    if "error" in raw:
        raise SkillWeaverError(f"/cart.js answered HTTP {raw['error']}")
    return tuple(Line(str(i["title"]), int(i["quantity"])) for i in raw.get("items", ()))


def quantity_of(item: str, lines: Sequence[Line]) -> int:
    """How many of ``item`` the site holds: the summed quantity of every line whose title
    IS the item, ignoring case and spacing. Strict on purpose - a title merely containing
    the words is a different product in a shop that sells a base set and its add-ons."""
    wanted = " ".join(item.casefold().split())
    return sum(line.quantity for line in lines if " ".join(line.title.casefold().split()) == wanted)


@dataclass(frozen=True, slots=True)
class TruthReport:
    """What the site said after the errand, beside what was asked."""

    lines: tuple[Line, ...]
    counts: dict[str, int]
    extras: tuple[Line, ...]

    @property
    def ok(self) -> bool:
        """Every requested item exactly once, and nothing the errand did not ask for."""
        return bool(self.counts) and all(n == 1 for n in self.counts.values()) and not self.extras

    def __str__(self) -> str:
        said = ", ".join(f"{name!r} x{n}" for name, n in self.counts.items())
        extra = "".join(f"; UNASKED {x.title!r} x{x.quantity}" for x in self.extras)
        return f"{'PASS' if self.ok else 'FAIL'} - the site holds {said}{extra}"


def check_truth(items: Sequence[str], lines: Sequence[Line]) -> TruthReport:
    """``lines`` judged against ``items``."""
    counts = {item: quantity_of(item, lines) for item in items}
    extras = tuple(x for x in lines if not any(quantity_of(i, [x]) for i in items))
    return TruthReport(tuple(lines), counts, extras)


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------

PathTaken = Literal[
    "cold", "warm", "warm-missed->cold", "warm-failed->rescued-cold", "warm re-add", "failed"
]


@dataclass(frozen=True, slots=True)
class ItemResult:
    """One item of the errand, as the agent reported it and as the site then said.

    ``act_ms`` is the navigation back to the start page plus every attempt's own
    ``wall_ms``; ``learn_ms`` is everything else inside the item - the admission gate -
    and ``learn_calls``/``learn_usd`` are what the model client spent there, read from its
    own ``total_usage`` across the item minus what the attempts were charged.
    ``on_site`` is the site's count for this item straight after it, or ``None`` when no
    ground-truth reader was given.
    """

    item: str
    task: str
    path: PathTaken
    ok: bool
    act_ms: float
    learn_ms: float
    steps: int
    llm_calls: int
    usd: float
    learned: str = ""
    note: str = ""
    on_site: int | None = None
    learn_calls: int = 0
    learn_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class ErrandReport:
    """The whole errand. ``truth`` is ``None`` when the site was never asked."""

    plan: Decomposition
    results: tuple[ItemResult, ...] = ()
    truth: TruthReport | None = None
    truth_error: str = ""
    setup_ms: float = 0.0
    wall_ms: float = 0.0
    notes: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Every item reported done AND the site agreeing, when it was asked."""
        done = bool(self.results) and all(r.ok for r in self.results)
        return done and (self.truth is None or self.truth.ok)

    def measured(self, path: str) -> list[float]:
        """``act_ms`` of the items on ``path`` a timing may be computed from: the ones
        the SITE confirmed, through the project's one door for timings."""
        from skillweaver.eval.metrics import RunPoint, _measured

        points = [
            RunPoint(
                attempt=n,
                ok=r.ok and r.on_site == 1,
                wall_ms=r.act_ms,
                llm_calls=r.llm_calls,
                usd=r.usd,
                used_library=r.path.startswith("warm"),
            )
            for n, r in enumerate(self.results, 1)
            if r.path == path
        ]
        return [p.wall_ms for p in _measured(points)]

    def explain(self) -> str:
        """The report a person reads. Seconds per item appear ONLY under a passing
        ground truth; otherwise the reason there is no number takes their place."""
        out = [f"errand: {self.plan.errand}", f"split:  {self.plan.how}, {len(self.plan.tasks)}"]
        if not self.plan:
            return "\n".join([*out, f"NOT RUN - {self.plan.why}"])
        for n, r in enumerate(self.results, 1):
            site = "" if r.on_site is None else f" site={r.on_site}"
            out.append(
                f"  {n}. {r.item!r}: {r.path} {'ok' if r.ok else 'FAILED'}{site} "
                f"act={r.act_ms / 1000:.1f}s learn={r.learn_ms / 1000:.1f}s actions={r.steps} "
                f"model_calls={r.llm_calls} usd={r.usd:.4f}"
                + (
                    f" learn_calls={r.learn_calls} learn_usd={r.learn_usd:.4f}"
                    if r.learn_calls
                    else ""
                )
                + (f" learned={r.learned}" if r.learned else "")
                + (f" - {r.note}" if r.note else "")
            )
        out.extend(f"  note: {n}" for n in self.notes)
        calls = sum(r.llm_calls for r in self.results)
        usd = sum(r.usd for r in self.results)
        out.append(
            f"totals: wall={self.wall_ms / 1000:.1f}s (browser+model setup "
            f"{self.setup_ms / 1000:.1f}s) model_calls={calls} usd={usd:.4f}"
        )
        if self.truth is None:
            why = self.truth_error or "no ground-truth reader was configured"
            out.append(f"ground truth: NOT READ ({why})")
            out.append("no efficiency number is printed: nothing confirmed what was done.")
            return "\n".join(out)
        out.append(f"ground truth: {self.truth}")
        if not self.truth.ok:
            out.append("no efficiency number is printed: the site does not hold the errand.")
            return "\n".join(out)
        act = sum(r.act_ms for r in self.results)
        out.append(
            f"seconds per item, learning excluded: {act / 1000 / len(self.results):.1f}s "
            f"over {len(self.results)} row(s)"
        )
        for path in ("cold", "warm"):
            kept = self.measured(path)
            if kept:
                each = ", ".join(f"{ms / 1000:.1f}" for ms in kept)
                out.append(
                    f"  {path}: {sum(kept) / len(kept) / 1000:.1f}s per item "
                    f"(n={len(kept)}, site-confirmed: {each})"
                )
        dropped = [r.item for r in self.results if not (r.ok and r.on_site == 1)]
        if dropped:
            out.append(f"  excluded from every timing (not site-confirmed): {dropped}")
        return "\n".join(out)

    def to_json(self) -> dict[str, Any]:
        """The same report as data. Carries ``efficiency`` only when the truth passed."""
        body: dict[str, Any] = {
            "errand": self.plan.errand,
            "how": self.plan.how,
            "tasks": list(self.plan.tasks),
            "ok": self.ok,
            "wall_ms": round(self.wall_ms),
            "setup_ms": round(self.setup_ms),
            "items": [
                {
                    "item": r.item,
                    "path": r.path,
                    "ok": r.ok,
                    "on_site": r.on_site,
                    "act_ms": round(r.act_ms),
                    "learn_ms": round(r.learn_ms),
                    "actions": r.steps,
                    "model_calls": r.llm_calls,
                    "usd": round(r.usd, 4),
                    "learn_calls": r.learn_calls,
                    "learn_usd": round(r.learn_usd, 4),
                    "learned": r.learned,
                    "note": r.note,
                }
                for r in self.results
            ],
            "ground_truth": None
            if self.truth is None
            else {
                "ok": self.truth.ok,
                "counts": self.truth.counts,
                "lines": [[x.title, x.quantity] for x in self.truth.lines],
            },
            "notes": list(self.notes),
        }
        if self.truth is not None and self.truth.ok:
            body["efficiency"] = {
                path: {"n": len(kept), "mean_ms": round(sum(kept) / len(kept))}
                for path in ("cold", "warm")
                if (kept := self.measured(path))
            }
        return body


def _check_orchestrator_seam() -> None:
    """Fail at start-up, by name, if an opener this module borrows was renamed."""
    from skillweaver import orchestrator

    wanted = (
        "_open_world",
        "_open_model",
        "_open_policy",
        "_open_move_critic",
        "_open_done_critic",
        "_reset_for",
    )
    missing = [name for name in wanted if not callable(getattr(orchestrator, name, None))]
    if missing:
        raise SkillWeaverError(
            f"agent.errand builds its session from orchestrator.{', '.join(missing)}, "
            "which no longer exist(s); see Workbench.session for what replaced it"
        )


class ErrandSession:
    """One browser, one model client and one policy, and a fresh ``Agent`` per task.

    Mirrors ``build_workbench``'s ``session`` line for line except for WHEN the agent is
    built; see the module docstring for why that matters to the warm critic.
    """

    def __init__(self, bench: Any, first: TaskSpec, budget: Budget) -> None:
        from skillweaver import orchestrator as o

        _check_orchestrator_seam()
        self._bench, self._budget, self._o = bench, budget, o
        self._config = bench.settings
        self.controller, self.perceiver = o._open_world(self._config, first)
        try:
            bench.graph.load(first.domain)
            self._llm = o._open_model(self._config)
            self._policy = o._open_policy(self._config, self.perceiver, self._llm)
            self._move_critic = o._open_move_critic(self._config, self.perceiver)
            self._done_critic = o._open_done_critic(
                self._config, self.perceiver, self.controller, self._policy
            )
        except BaseException:
            self.controller.close()
            raise

    def agent_for(self, task: TaskSpec) -> Any:
        """The real agent for ``task``, wired by ``build_agent`` against the library as it
        stands NOW."""
        from skillweaver.trajectory.record import Recorder

        o, bench = self._o, self._bench
        return o.build_agent(
            task,
            controller=self.controller,
            perceiver=self.perceiver,
            llm=self._llm,
            policy=self._policy,
            move_critic=self._move_critic,
            done_critic=self._done_critic,
            store=bench.store,
            retriever=o.build_retriever(bench.store),
            graph=bench.graph,
            trajectories=bench.trajectories,
            recorder=Recorder(self._config.trajectories_dir),
            environment=o.navigating_environment(
                self.controller,
                self.perceiver,
                graph=bench.graph,
                restore=self.restore_for(task),
                read_only=bool(task.params.get(o.READ_ONLY_PARAM, False)),
            ),
            budget=self._budget,
        )

    def restore_for(self, task: TaskSpec) -> Any:
        """The task's own undo, through the one call site allowed to build it."""
        return self._o._reset_for(task, self.controller, self.perceiver)

    def usage(self) -> tuple[int, float]:
        """``(calls, usd)`` spent so far by the model client AND the acting policy, which
        meters itself (``JevPolicy.total_usage``); zeros from whoever keeps no meter. Both,
        because a cold attempt's ``llm_calls`` counts both and learning is what is left."""
        calls, usd = 0, 0.0
        for source in (self._llm, self._policy):
            total = getattr(source, "total_usage", None)
            spent = total() if callable(total) else None
            calls += getattr(spent, "calls", 0)
            usd += getattr(spent, "cost_usd", 0.0)
        return calls, usd

    def close(self) -> None:
        """Close the browser this session opened."""
        self.controller.close()


@contextmanager
def errand_session(bench: Any, first: TaskSpec, budget: Budget) -> Iterator[ErrandSession]:
    """Open an :class:`ErrandSession` and close its browser whatever happens."""
    session = ErrandSession(bench, first, budget)
    try:
        yield session
    finally:
        session.close()


def run_errand(
    bench: Any,
    errand: str,
    *,
    url: str,
    reset_steps: Any = None,
    budget: Budget | None = None,
    truth: GroundTruth | None = None,
    learn: bool = True,
    empty_first: bool = True,
    empty_after: bool = True,
    name_items: Callable[[str], str] | None = None,
) -> ErrandReport:
    """Do ``errand`` at ``url``, one item at a time, in one browser session.

    Args:
        bench: A ``Workbench`` (``orchestrator.build_workbench``).
        reset_steps: The ``--reset-steps`` undo, as ``task_spec`` takes it. It is what the
            admission gate uses for the learning item, and what ``empty_first`` and
            ``empty_after`` perform around the errand.
        truth: Reads the site after every item and at the end. Without it the report
            prints no efficiency number.
        learn: Whether the FIRST cold item may be offered to the gate. Later ones never
            are - see the module docstring.
    """
    from skillweaver import orchestrator as o

    began = time.monotonic()
    plan = decompose(errand, name_items=name_items)
    if not plan:
        return ErrandReport(plan)
    budget = budget if budget is not None else o.budget_from(bench.settings)
    where = o.resolve_domain(
        plan.tasks[0],
        url=url,
        retriever=bench.retriever,
        graph=bench.graph,
        path=bench.settings.perception,
    )
    specs = [
        o.task_spec(text, domain=where.domain, url=where.start_url or url, reset_steps=reset_steps)
        for text in plan.tasks
    ]
    start_url = where.start_url or url
    results: list[ItemResult] = []
    notes: list[str] = []
    found: TruthReport | None = None
    truth_error = ""

    with errand_session(bench, specs[0], budget) as session:
        setup_ms = (time.monotonic() - began) * 1000.0
        restore = session.restore_for(specs[0])
        if empty_first and restore is not None:
            notes.append(f"before: {o.reset_world(restore)}")

        def on_site(item: str) -> int | None:
            if truth is None:
                return None
            try:
                return quantity_of(item, truth(session.controller))
            except SkillWeaverError as exc:
                log.warning("errand.truth.unreadable", item=item, error=str(exc))
                return None

        for n, (item, spec) in enumerate(zip(plan.items, specs, strict=True)):
            may_learn = learn and not any(r.ok for r in results)
            result = _run_item(session, item, spec, start_url, n > 0 or empty_first, may_learn)
            result = _with_site(result, on_site(item))
            results.append(result)
            log.info("errand.item", n=n + 1, item=item, path=result.path, ok=result.ok)
            if result.learned and result.ok and result.on_site == 0:
                # The gate's last act left the world without the item it just proved.
                notes.append(
                    f"{item!r}: learned, but the admission gate's resets left it off the "
                    "site, so it is added once more through the warm path"
                )
                again = _run_item(session, item, spec, start_url, True, False, cold=False)
                again = _with_site(again, on_site(item))
                results.append(replace(again, path="warm re-add" if again.ok else "failed"))

        if truth is not None:
            try:
                found = check_truth(plan.items, truth(session.controller))
            except SkillWeaverError as exc:
                truth_error = str(exc)
        if empty_after and restore is not None:
            notes.append(f"after: {o.reset_world(restore)}")
            if truth is not None:
                try:
                    left = sum(x.quantity for x in truth(session.controller))
                    notes.append(f"after the undo the site holds {left} item(s)")
                except SkillWeaverError as exc:
                    notes.append(f"after the undo the site could not be read: {exc}")

    return ErrandReport(
        plan,
        tuple(results),
        found,
        truth_error,
        setup_ms=setup_ms,
        wall_ms=(time.monotonic() - began) * 1000.0,
        notes=tuple(notes),
    )


def _with_site(result: ItemResult, count: int | None) -> ItemResult:
    return replace(result, on_site=count)


def _run_item(
    session: ErrandSession,
    item: str,
    spec: TaskSpec,
    start_url: str,
    navigate: bool,
    may_learn: bool,
    *,
    cold: bool = True,
) -> ItemResult:
    """One item through ``Agent.run``, timed over the same boundary for every item: from
    leaving the last item's end screen to the report coming back."""
    began = time.monotonic()
    if navigate and session.controller.supports("navigate"):
        session.controller.perform(Navigate(start_url))
    nav_ms = (time.monotonic() - began) * 1000.0
    mark = session.usage()
    try:
        report = session.agent_for(spec).run(spec, learn=may_learn, cold=cold)
    except SkillWeaverError as exc:
        spent = (time.monotonic() - began) * 1000.0
        return ItemResult(item, spec.text, "failed", False, spent, 0.0, 0, 0, 0.0, note=str(exc))
    total_ms = (time.monotonic() - began) * 1000.0
    attempts_ms = sum(a.wall_ms for a in report.attempts)
    note = (
        "" if report.ok else "; ".join(f"{a.path}@{a.stage}: {a.reason}" for a in report.attempts)
    )
    if report.ok and not may_learn and report.decision == "cold":
        note = "not offered to the gate: its resets would undo the items already done"
    elif report.decision == "cold" and report.learned is None and report.learning_note:
        note = f"not learned: {report.learning_note}"
    return ItemResult(
        item=item,
        task=spec.text,
        path=_path_of(report),
        ok=report.ok,
        act_ms=nav_ms + attempts_ms,
        learn_ms=max(0.0, total_ms - nav_ms - attempts_ms),
        steps=report.steps,
        llm_calls=report.llm_calls,
        usd=sum(a.usd for a in report.attempts),
        learned=report.learned.name if report.learned is not None else "",
        note=note[:400],
        learn_calls=max(0, session.usage()[0] - mark[0] - report.llm_calls),
        learn_usd=max(0.0, session.usage()[1] - mark[1] - sum(a.usd for a in report.attempts)),
    )


def _path_of(report: Any) -> PathTaken:
    """Which path did the item, with a fall-through named as one. An EMPTY library is
    the one warm decline that is an ordinary cold start and not a miss."""
    if not report.ok:
        return "failed"
    if report.decision == "warm":
        return "warm"
    warm = report.warm
    if warm is None or warm.stage == "empty_library":
        return "cold"
    return "warm-failed->rescued-cold" if report.rescued else "warm-missed->cold"
