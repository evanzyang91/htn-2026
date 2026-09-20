"""One live, step-driven run: the thing the inspector's buttons drive.

:class:`LiveSession` is :meth:`skillweaver.agent.explorer.Explorer.explore` taken apart
at its seams so a person can stand between the halves. The loop that module runs is::

    catalog = ElementCatalog(run.current.elements)
    answer  = explorer._ask(task, run, catalog)          # <- DECIDE
    move    = explorer._ground(answer, catalog, ctl)
    explorer._refuse_repeat(move, run)
    explorer._make_move(task, move, catalog, ctl, run)   # <- ACT

``predict`` is the first three lines and ``act`` is the last one; ``tick`` is both, which
is exactly Aaron's ``predict`` / ``act`` / ``tick`` vocabulary landing on this project's
stack. Everything downstream of the split - the failure memory, the grounding, the taped
controller, the critic, the graph writing, the trajectory and all four budget limits - is
the SAME code the ordinary ``learn`` path runs, because it IS that code: this module owns
no policy, no grounding and no judgement of its own.

Why it reaches into a private loop
----------------------------------

:meth:`Explorer.explore` is all-or-nothing by design, and ``agent/explorer.py`` belongs to
another worker this week, so there is no stepped entry point to add and none is added here.
The four private names above are therefore called directly, from :meth:`LiveSession._decide`
and :meth:`LiveSession._perform` and nowhere else, so that a rename upstream breaks two
methods rather than a module. :func:`_check_explorer_seam` states the dependency out loud
at construction time instead of halfway through a demo.

What is OBSERVED rather than reconstructed
------------------------------------------

Three wrappers go in through :class:`Explorer`'s own public constructor arguments, so what
the log reports is what the run did rather than what this module guessed it did:

* :class:`_WatchedController` records every action that reached the browser and what came
  back - the explorer wraps it again in its own tape, and both see the same actions.
* :class:`_WatchedCritic` keeps each verdict, because ``_make_move`` consumes it into a
  history line and a person wants the verdict, its confidence and who gave it.
* :class:`_WatchedPolicy` keeps each :class:`~skillweaver.llm.jev_.PolicyDecision`, which
  is the only way to show Jev's operation, target and confidence as NUMBERS: the driver
  renders them into a sentence and hands on an answer object, and a demo that parsed that
  sentence back would be reporting its own regex.

What is NOT available, and is shown as absent
---------------------------------------------

:class:`~skillweaver.llm.jev_.PolicyDecision` carries the CHOSEN target's probability and
the operation head's confidence, and no distribution over either. So the inspector ranks
the alternatives it can see - every control the screen offered, with the withheld ones
marked - and says plainly that the per-target odds are not exposed. Upstream Jev's own
inspector draws bars from ``target_probabilities``; this one will too, the day a decision
carries them. Nothing here fabricates a number to fill the space.

Whose time a step took
----------------------

Upstream's inspector splits every step three ways - MODEL, LOAD and FRAME - because one
number called "latency" hid a screenshot that cost more than the site did. The same split
is drawn here from clocks this project already keeps, read and never re-derived:
``JevDriver.policy_ms`` is the policy's own models, ``DomPerceiver.site_ms`` is performing,
settling, observing and resting, and :class:`_WatchedController` times every ``capture``
that passes through it. A capture happens INSIDE an observation, so it is taken back out of
the site's share: the picture is what a viewer looks at, and on a heavy page charging it to
the site reports the website as slow for work the website never did. What is left of the
wall clock is judging and recording - the critic, the graph, the trajectory - and is shown
as its own number so the three never have to be stretched to add up. Every read is
tolerant (``getattr``): a perceiver or a policy that keeps no clock yields a split that
says less, never one that raises.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillweaver.agent.critic import TieredCritic
from skillweaver.agent.explorer import (
    ElementCatalog,
    Explorer,
    FailureMemory,
    Move,
    PolicyBlocked,
    signature_move,
)
from skillweaver.agent.explorer import _Invalid as InvalidAnswer
from skillweaver.agent.explorer import _Run as ExplorerRun
from skillweaver.config import DEFAULT_TEXT_MODEL, TEXT_EFFORTS, Settings
from skillweaver.contracts import (
    Action,
    ActionResult,
    Box,
    Budget,
    Controller,
    Fingerprint,
    Navigate,
    Observation,
    Perceiver,
    Screenshot,
    Spend,
    TaskSpec,
    Usage,
    Verdict,
    Wait,
    action_to_dict,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.graph.model import InMemorySiteGraph
from skillweaver.graph.store import JSONGraphStore
from skillweaver.llm.jev_ import PolicyDecision
from skillweaver.logging_ import get_logger
from skillweaver.orchestrator import (
    RESET_URL_PARAM,
    build_retriever,
    reset_world,
    task_spec,
    world_reset_from_url,
)
from skillweaver.perception.dom import DomSnapshot
from skillweaver.perception.fingerprint import SAME_STATE_THRESHOLD
from skillweaver.render_mode import mode_name
from skillweaver.reset_actions import (
    RESET_ACTIONS_PARAM,
    chain_resets,
    reset_steps_from,
    world_reset_from_actions,
)
from skillweaver.skills.store import FileSkillStore
from skillweaver.trajectory.record import Recorder
from skillweaver.trajectory.store import TrajectoryFileStore

log = get_logger(__name__)

__all__ = [
    "STATES",
    "TEXT_MODEL_CHOICES",
    "LiveSession",
    "SessionBusy",
    "SessionError",
    "StepRecord",
    "run_overrides",
    "text_models",
]

STATES = ("idle", "ready", "predicted", "done", "blocked", "stopped")
"""Every value :attr:`LiveSession.status` takes.

``idle`` before a task is started; ``ready`` when a screen is observed and nothing is
decided; ``predicted`` when a move is chosen and not yet performed - the ONE state in
which ``act`` means anything; ``done`` when the critic agreed the task is complete;
``blocked`` when the policy or the loop reported no supported next move; ``stopped``
when a budget limit was reached. The last three are terminal for that run and only a
reset leaves them, which is what stops a paused demo from being quietly resumed into a
run whose budget is already spent.
"""

TERMINAL = frozenset({"done", "blocked", "stopped"})
"""The states a reset is the only way out of."""

MAX_LOG_STEPS = 400
"""How many step records are kept. A run that reaches this has long since hit a budget
limit; the cap is here so a paused browser left running overnight cannot grow without
bound."""


class SessionError(SkillWeaverError):
    """The session was asked for something it cannot do in the state it is in.

    Carries a sentence meant to be READ by whoever pressed the button, so the page can
    show it verbatim rather than inventing its own wording.
    """


class SessionBusy(SessionError):
    """A step is already running. Never raised into the agent thread - only at the door,
    when a second request arrives while the first still holds the session."""


# --------------------------------------------------------------------------------------
# What the page may change about a run
# --------------------------------------------------------------------------------------

TEXT_MODEL_CHOICES = (DEFAULT_TEXT_MODEL, "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5.4-nano")
"""The text writers the page offers beside the configured one: the rows of the table at
``DEFAULT_TEXT_MODEL`` in ``config.py`` that answered in under a second. An allowlist and
not a text box, because the value goes into a request to a paid endpoint and a typo there
is a run that dies at its first ``TYPE_TEXT``. Anything else is still reachable the way it
always was, through ``SKILLWEAVER_TEXT_MODEL``, which :func:`text_models` lists first."""


def text_models(settings: Settings) -> list[str]:
    """What the model selectors offer: the configured models, then the measured ones."""
    configured = [settings.text_model, settings.refine_model or ""]
    names = [name for name in (*configured, *TEXT_MODEL_CHOICES) if name]
    return list(dict.fromkeys(names))


def run_overrides(body: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    """The run-scoped settings a ``start`` body asks for, validated, as ``replace`` fields.

    Only what is safe to change per run: whether the goal is refined for the policy's
    eyes, and the text writer's model and effort. A key that is absent is not an
    override, so a client that knows nothing of these gets the configured run.

    They apply when a run is STARTED and not in the middle of one, which is where this
    departs from upstream's selector. Upstream's helper reads ``TEXT_MODEL`` on every
    call; this project's ``OpenAITextWriter`` is built once with its model and effort and
    offers no way to change them, and the refiner is a closure handed to ``JevDriver`` at
    construction. Reaching into either from here would be the inspector rewriting another
    module's private state under a run in flight.

    Raises:
        SessionError: for a model that is not offered or an effort that does not exist.
    """
    out: dict[str, Any] = {}
    if "refine_goal" in body:
        out["refine_goal"] = bool(body["refine_goal"])
    offered = text_models(settings)
    for key in ("text_model", "refine_model"):
        value = str(body.get(key) or "").strip()
        if not value:
            continue
        if value not in offered:
            raise SessionError(f"{value!r} is not one of the offered models: {offered}.")
        out[key] = value
    if "text_effort" in body:
        effort = str(body["text_effort"] or "").strip().lower()
        if effort and effort not in TEXT_EFFORTS:
            raise SessionError(f"The effort must be one of {TEXT_EFFORTS}, or blank.")
        out["text_effort"] = effort or None
    return out


# --------------------------------------------------------------------------------------
# What one step leaves behind
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One entry in the running history: decided, executed, observed, judged.

    Every field is filled from something that HAPPENED. ``verdict_ok`` is ``None`` for a
    move that reached nothing - there was no after-screen to judge - and for one that was
    refused before it was performed; those two are told apart by ``refused``.

    ``wall_ms`` is ``decide_ms + act_ms``, and ``model_ms``, ``site_ms``, ``frame_ms`` and
    ``other_ms`` are whose time that was - see the module docstring. They never sum past
    the wall: :class:`_Split` is what holds that.
    """

    number: int
    headline: str
    summary: str
    thought: str
    expect: str
    claimed_done: bool
    signature: str
    operation: str
    target: str
    confidence: float | None
    probability: float | None
    decide_ms: float
    act_ms: float
    wall_ms: float
    model_ms: float
    site_ms: float
    frame_ms: float
    other_ms: float
    actions: tuple[str, ...]
    delivered: bool
    error: str | None
    url_before: str
    url_after: str
    screen_changed: bool
    similarity: float
    verdict_ok: bool | None
    verdict_reason: str
    verdict_source: str
    verdict_confidence: float
    refused: bool
    at: float

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Watching the run without changing it
# --------------------------------------------------------------------------------------


class _WatchedController:
    """A :class:`~skillweaver.contracts.Controller` that remembers what it was asked.

    Passed to :meth:`Explorer._make_move` as the controller, which wraps it in its own
    ``_TapedController``: the tape re-observes and this counts, so neither has to know
    about the other. It owns nothing and closes nothing - :class:`LiveSession` owns the
    real controller's lifetime.
    """

    __slots__ = ("_inner", "capture_ms", "performed_ms", "seen")

    def __init__(self, inner: Controller) -> None:
        self._inner = inner
        self.seen: list[tuple[Action, ActionResult]] = []
        # The FRAME clock: milliseconds spent inside ``capture``.
        self.capture_ms = 0.0
        # What the actions reported taking. Read only when the perceiver keeps no site
        # clock of its own, which is the pixel path.
        self.performed_ms = 0.0

    @property
    def inner(self) -> Controller:
        return self._inner

    def forget(self) -> None:
        self.seen.clear()

    def perform(self, action: Action) -> ActionResult:
        result = self._inner.perform(action)
        self.seen.append((action, result))
        self.performed_ms += float(getattr(result, "elapsed_ms", 0.0) or 0.0)
        return result

    def capture(self) -> Screenshot:
        started = time.perf_counter()
        try:
            return self._inner.capture()
        finally:
            self.capture_ms += (time.perf_counter() - started) * 1000

    def viewport(self) -> Box:
        return self._inner.viewport()

    def supports(self, action_kind: Any) -> bool:
        return self._inner.supports(action_kind)

    def url(self) -> str | None:
        return self._inner.url()

    def describe(self) -> str:
        return self._inner.describe()

    def close(self) -> None:
        """Deliberately does nothing: the session closes the real controller once."""

    def __getattr__(self, name: str) -> Any:
        """Everything this class does not record, straight from the real controller.

        Not a convenience. :class:`~skillweaver.perception.dom.DomPerceiver` reads the
        page through ``controller.evaluate``, which is no part of the
        :class:`~skillweaver.contracts.Controller` protocol, and the explorer re-observes
        THROUGH this object after every action of a code block. A wrapper that showed only
        the protocol made the first typed move on live Wikipedia stop after its click:
        the re-observation raised, the block died at line 2, and nothing was typed. A
        watcher has to be transparent to everything it is not watching.
        """
        return getattr(self._inner, name)


class _WatchedCritic:
    """A :class:`~skillweaver.contracts.Critic` that keeps its last verdict.

    The explorer folds a verdict into one history line and drops the rest. A person
    watching wants the confidence and the source too - "the fingerprints matched" and "a
    model looked at it and thinks so" are not the same claim, and only one of them costs
    money.
    """

    __slots__ = ("_inner", "last")

    def __init__(self, inner: TieredCritic) -> None:
        self._inner = inner
        self.last: Verdict | None = None

    def judge(
        self,
        goal: str,
        before: Observation,
        after: Observation,
        expectation: str | None = None,
    ) -> Verdict:
        verdict = self._inner.judge(goal, before, after, expectation)
        self.last = verdict
        return verdict


class _WatchedPolicy:
    """A :class:`~skillweaver.llm.jev_.BrowserPolicy` that keeps its last decision.

    The join this exists for: :class:`~skillweaver.agent.jev_driver.JevDriver` turns a
    :class:`~skillweaver.llm.jev_.PolicyDecision` into an explorer answer object and lets
    the decision go, so by the time a move exists the operation, the confidence and the
    probability survive only as prose inside ``thought``. Wrapping the policy keeps the
    numbers as numbers. The driver is unchanged and does not know this is here.
    """

    __slots__ = ("_inner", "last", "last_exclude", "last_reserved", "last_snapshot")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last: PolicyDecision | None = None
        self.last_snapshot: DomSnapshot | None = None
        self.last_exclude: dict[str, list[str]] = {}
        self.last_reserved: dict[str, list[str]] = {}

    def name(self) -> str:
        return str(self._inner.name())

    def total_usage(self) -> Usage:
        return self._inner.total_usage()

    def decide(
        self,
        goal: str,
        snapshot: DomSnapshot,
        history: Sequence[Mapping[str, Any]],
        exclude: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        decision = self._inner.decide(goal, snapshot, history, exclude)
        self.last = decision
        self.last_snapshot = snapshot
        # ``exclude`` is ``{operation: element ids}`` plus the two RESERVED keys, whose
        # values are not element ids at all; filed apart so nothing tests an id against
        # an operation name or a label.
        held = {key: sorted(map(str, ids)) for key, ids in (exclude or {}).items() if ids}
        self.last_reserved = {key: held.pop(key) for key in RESERVED_EXCLUDES if key in held}
        self.last_exclude = held
        return decision


RESERVED_EXCLUDES = ("CONTROLS", "LABELS")
"""The two keys of a policy's ``exclude`` mapping that name no operation: ``CONTROLS`` is
target-less operations withheld from this step's offer (a scroll, wait or back that has
changed nothing twice) and ``LABELS`` is control labels withheld from EVERY targeted
operation (one the run is churning between). ``agent/jev_driver.py`` produces them."""


def _check_explorer_seam() -> None:
    """Fail at construction if the explorer's loop no longer has the shape split here.

    Cheap insurance against the one thing this module cannot survive: a rename in
    ``agent/explorer.py``, which is owned by another worker. Better a clear sentence
    before a browser opens than an ``AttributeError`` in front of an audience.

    Raises:
        SessionError: naming the method that moved.
    """
    needed = ("_ask", "_ground", "_refuse_repeat", "_make_move", "_exhausted", "_charge")
    missing = [name for name in needed if not hasattr(Explorer, name)]
    if missing:
        raise SessionError(
            "the inspector drives the explorer's own loop one step at a time, and "
            f"Explorer no longer has {', '.join(missing)}. See skillweaver.inspector."
            "session - the split is in _decide and _perform and nowhere else."
        )


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Split:
    """Whose time a stretch of wall clock was. See the module docstring.

    The shares are CLAMPED to the wall, in the order frame, model, site. Two clocks kept
    by two modules can overlap - a text value written while a page settles is both model
    time and site time - and a split that sums past its own wall is a number nobody can
    check. What the clamp takes is never more than the overlap.
    """

    wall_ms: float = 0.0
    model_ms: float = 0.0
    site_ms: float = 0.0
    frame_ms: float = 0.0

    @property
    def other_ms(self) -> float:
        return max(self.wall_ms - self.model_ms - self.site_ms - self.frame_ms, 0.0)

    @classmethod
    def of(cls, wall: float, model: float, site: float, frame: float) -> _Split:
        wall = max(wall, 0.0)
        frame = min(max(frame, 0.0), wall)
        model = min(max(model, 0.0), wall - frame)
        site = min(max(site, 0.0), wall - frame - model)
        return cls(wall, model, site, frame)

    def __add__(self, other: _Split) -> _Split:
        return _Split(
            self.wall_ms + other.wall_ms,
            self.model_ms + other.model_ms,
            self.site_ms + other.site_ms,
            self.frame_ms + other.frame_ms,
        )

    def as_json(self) -> dict[str, int]:
        return {
            "wall_ms": round(self.wall_ms),
            "model_ms": round(self.model_ms),
            "site_ms": round(self.site_ms),
            "frame_ms": round(self.frame_ms),
            "other_ms": round(self.other_ms),
        }


@dataclass(frozen=True, slots=True)
class _Marks:
    """The three clocks read at one instant, so a stretch is two of these subtracted."""

    at: float
    model_ms: float
    site_ms: float | None
    frame_ms: float
    performed_ms: float


@dataclass(slots=True)
class _Pending:
    """A decision made and not yet performed."""

    move: Move
    catalog: ElementCatalog
    decide_ms: float
    at_fingerprint: str
    decision: PolicyDecision | None
    split: _Split


class LiveSession:
    """One inspected run: a browser, a task, and a loop a person advances by hand.

    Args:
        settings: The resolved configuration. Its ``headless``, ``chrome_profile``,
            ``chrome_attach``, ``perception`` and ``policy`` are honoured exactly as the
            ``learn`` and ``run`` commands honour them - the inspector forces no mode,
            because the one a live site accepts may be the only one that works.
        open_world: How to open a controller and a perceiver for a task. Defaults to the
            orchestrator's own ``_open_world``; taken as an argument so the browser can be
            replaced without this module knowing what replaced it.
        open_model: How to build the computer-use client. Defaults to the orchestrator's.
        open_policy: How to build the acting policy, or ``None`` for the prompted default.

    A session is NOT thread-safe and must be driven from the thread that built it,
    because a Playwright sync driver belongs to one thread. :mod:`skillweaver.inspector.
    server` is what arranges that.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        open_world: Callable[[Settings, TaskSpec], tuple[Controller, Perceiver]] | None = None,
        open_model: Callable[[Settings], Any] | None = None,
        open_policy: Callable[[Settings, Perceiver, Any], Any] | None = None,
    ) -> None:
        _check_explorer_seam()
        from skillweaver.orchestrator import _open_model, _open_policy, _open_world

        self._settings = settings
        # What THIS run was started with: the configured settings plus whatever the
        # prompt bar overrode. Every rebuild within a run (Reset browser) reads this one,
        # so a run cannot change its text writer halfway by being given a new browser.
        self._run_settings = settings
        self._open_world = open_world or _open_world
        self._open_model = open_model or _open_model
        self._open_policy = open_policy or _open_policy

        self._controller: Controller | None = None
        self._watched: _WatchedController | None = None
        self._perceiver: Perceiver | None = None
        self._explorer: Explorer | None = None
        self._critic: _WatchedCritic | None = None
        self._policy: _WatchedPolicy | None = None
        self._run: ExplorerRun | None = None
        self._task: TaskSpec | None = None
        self._pending: _Pending | None = None

        self._status = "idle"
        self._note = ""
        self._log: list[StepRecord] = []
        self._frame = 0
        self._frame_png = b""
        self._started_at: float | None = None
        self._resets: list[dict[str, Any]] = []
        self._concluded = True
        self._last_run_id = ""
        self._total = _Split()
        self._first_load = _Split()

        self._graph = InMemorySiteGraph(store=JSONGraphStore(settings.graphs_dir))
        self._trajectories = TrajectoryFileStore(settings.trajectories_dir)
        self._store = FileSkillStore(settings.skills_dir, render_mode=mode_name(settings.headless))

    # -- what the page reads -----------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def frame_png(self) -> bytes:
        """The current screen as PNG bytes, empty before the first observation.

        Served as an image rather than inlined as base64 in every poll: a 1280x800 frame
        is ~200 KB and base64 makes it ~270 KB, which a once-a-second poll turns into
        real bandwidth for a picture that usually has not changed. The page re-fetches it
        only when :attr:`frame` moves.
        """
        return self._frame_png

    @property
    def frame(self) -> int:
        """A counter that increments whenever the screen was re-observed."""
        return self._frame

    def snapshot(self) -> dict[str, Any]:
        """Everything the page draws, as JSON-safe data."""
        run, task = self._run, self._task
        observation = run.current if run is not None else None
        elapsed = (time.perf_counter() - self._started_at) if self._started_at else 0.0
        spend = run.spend if run is not None else None
        return {
            "status": self._status,
            "note": self._note,
            "goal": task.text if task else "",
            "domain": task.domain if task else "",
            "start_url": (task.params.get("start_url") if task else "") or "",
            "url": (observation.url if observation else None) or "",
            "frame": self._frame,
            "viewport": _box_json(observation.screenshot if observation else None),
            "elements": self._elements_json(),
            "decision": self._decision_json(),
            "history": [record.as_json() for record in self._log],
            "resets": self._resets[-8:],
            "elapsed_ms": round(elapsed * 1000),
            "timing": {
                **self._total.as_json(),
                "first_load_ms": round(self._first_load.site_ms + self._first_load.other_ms),
                "first_frame_ms": round(self._first_load.frame_ms),
                "site_clock": self._site_clock() is not None,
            },
            "run_id": self._last_run_id if self._concluded else "",
            "moves": run.moves if run else 0,
            "steps": run.steps if run else 0,
            "spend": _spend_json(spend),
            "mode": {
                "perception": self._settings.perception,
                "policy": self._settings.policy,
                "render": mode_name(self._settings.headless),
                "chrome_profile": str(self._settings.chrome_profile or ""),
                "chrome_attach": bool(self._settings.chrome_attach),
                "browser": self._controller.describe() if self._controller else "",
                "model": self._explorer_model(),
                "text_model": self._run_settings.text_model,
                "text_effort": self._run_settings.text_effort or "",
                "refine_goal": bool(self._run_settings.refine_goal),
                "refine_model": self._run_settings.refine_model or "",
                "goal_shown": self._goal_shown(),
            },
            "options": {
                "text_models": text_models(self._settings),
                "efforts": ["", *TEXT_EFFORTS],
                "configured": {
                    "text_model": self._settings.text_model,
                    "text_effort": self._settings.text_effort or "",
                    "refine_goal": bool(self._settings.refine_goal),
                    "refine_model": self._settings.refine_model or "",
                },
                # The writer and the refiner only exist under the Jev policy.
                "adjustable": self._settings.policy == "jev",
            },
            "can": {
                "advance": self._status in ("ready", "predicted"),
                "predict": self._status in ("ready", "predicted"),
                "act": self._status == "predicted",
                "reset_run": self._task is not None,
                "reset_browser": self._task is not None,
            },
        }

    # -- the prompt bar ----------------------------------------------------------------

    def start(
        self,
        *,
        url: str,
        goal: str,
        params: Mapping[str, Any] | None = None,
        reset_url: str | None = None,
        reset_steps: Any = None,
        read_only: bool = False,
        budget: Budget | None = None,
        domain: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Open a world on ``url`` and begin a run for ``goal``.

        ``overrides`` is :func:`run_overrides`' answer: settings fields this run is built
        with instead of the configured ones. They last until the next ``start``.

        Called again on an already-open session, this replaces the run and the browser:
        a new task may want a different start URL and a clean history, and reusing a
        browser that is halfway through somebody else's errand is how a demo shows a
        screen nobody asked for.

        Raises:
            SessionError: if the goal or URL is unusable, or the undo does not parse -
                all three before a browser opens, because finding out afterwards costs
                the launch.
        """
        goal = goal.strip()
        if not goal:
            raise SessionError("Give it a task: one sentence saying what to do.")
        if len(goal) > 2000:
            raise SessionError("That task is longer than 2,000 characters.")
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise SessionError("The address must be a full http:// or https:// URL.")
        try:
            spec = task_spec(
                goal,
                domain=domain or None,
                target="browser",
                url=url,
                reset_url=reset_url or None,
                reset_steps=reset_steps,
                read_only=read_only,
                params=dict(params or {}),
            )
        except ValueError as exc:
            raise SessionError(f"The undo does not parse: {exc}") from exc

        try:
            settings = dataclasses.replace(self._settings, **dict(overrides or {}))
        except TypeError as exc:
            raise SessionError(f"That is not a setting a run can be started with: {exc}") from exc

        self.close()
        self._task = spec
        self._run_settings = settings
        controller, perceiver = self._open_world(settings, spec)
        self._controller = controller
        self._watched = _WatchedController(controller)
        self._perceiver = perceiver
        try:
            llm = self._open_model(settings)
            policy = self._open_policy(settings, perceiver, llm)
            self._policy = _WatchedPolicy(policy._policy) if _has_inner_policy(policy) else None
            if self._policy is not None:
                policy._policy = self._policy  # noqa: SLF001 - see _WatchedPolicy
            self._critic = _WatchedCritic(TieredCritic(llm))
            self._explorer = Explorer(
                llm,
                perceiver,
                critic=self._critic,  # type: ignore[arg-type]
                graph=self._graph,
                recorder=Recorder(self._settings.trajectories_dir),
                retriever=build_retriever(self._store),
                policy=policy,
            )
        except Exception:
            self.close()
            raise
        self._graph.load(spec.domain)
        self._begin(budget or self._settings.default_budget, navigate=True)
        log.info("inspect.start", task=goal, url=url, domain=spec.domain)

    # -- the step buttons --------------------------------------------------------------

    def predict(self) -> None:
        """Choose the next move and show it. Executes NOTHING.

        A second ``predict`` on an already-predicted screen replaces the held decision,
        which is what re-asking should mean: the first answer was not performed, so it
        costs the model call and nothing else.

        Raises:
            SessionError: when there is no run, or the run has stopped.
        """
        explorer, run = self._live()
        if explorer._exhausted(run):  # noqa: SLF001 - see the module docstring
            self._stop("stopped", run.detail or "the run reached a budget limit")
            raise SessionError(self._note)
        self._decide(explorer, run)

    def act(self) -> None:
        """Perform the move that is being shown, then judge and record it.

        Raises:
            SessionError: when nothing is being shown, or the screen moved under the
                decision - a page that repainted between the choice and the button is a
                page the choice was not made about.
        """
        explorer, run = self._live()
        pending = self._pending
        if pending is None:
            raise SessionError("Choose a move first - there is nothing to execute.")
        if pending.at_fingerprint != run.current.fingerprint.value:
            self._drop_pending()
            self._status = "ready"
            raise SessionError(
                "The screen changed after that choice was made, so it has been "
                "discarded. Choose again."
            )
        self._pending = None
        self._perform(explorer, run, pending)

    def tick(self) -> None:
        """Decide and then execute, which is one whole step of the ordinary loop."""
        self.predict()
        if self._status == "predicted":
            self.act()

    def advance(self) -> None:
        """One move of an automatic run: execute what is being shown, or choose and execute.

        What ``tick`` is for a loop that may be switched on while a choice is already on
        the table. ``tick`` there would ask AGAIN and pay a second model call for a move
        the person was looking at when they pressed the switch.
        """
        if self._status == "predicted" and self._pending is not None:
            self.act()
        else:
            self.tick()

    # -- the reset buttons -------------------------------------------------------------

    def reset_run(self) -> None:
        """Forget this run and start another against the same browser.

        The history, the failure memory, the budget and the trajectory all go; the
        browser and its cookies stay, and the start URL is loaded again. This is the
        cheap reset - the one to press when a run went somewhere uninteresting.

        Raises:
            SessionError: if no task has been started.
        """
        if self._task is None or self._explorer is None:
            raise SessionError("There is no run to reset - start a task first.")
        budget = self._run.spend.budget if self._run else self._settings.default_budget
        self._begin(budget, navigate=True)
        self._remember_reset("run", True, "history, failure memory and budget cleared")
        log.info("inspect.reset.run")

    def reset_browser(self) -> None:
        """Close the browser and open a fresh one on the same task.

        Everything ``reset_run`` clears, plus the browser session itself: cookies, local
        storage and whatever a half-finished flow left in the page. A persistent Chrome
        profile is NOT deleted - that directory is the point of using one, and a
        verification page cleared by hand once should stay cleared.

        Raises:
            SessionError: if no task has been started.
        """
        task = self._task
        if task is None:
            raise SessionError("There is no browser to reset - start a task first.")
        explorer, critic, policy = self._explorer, self._critic, self._policy
        budget = self._run.spend.budget if self._run else self._settings.default_budget
        if self._controller is not None:
            self._controller.close()
        controller, perceiver = self._open_world(self._run_settings, task)
        self._controller = controller
        self._watched = _WatchedController(controller)
        self._perceiver = perceiver
        self._explorer, self._critic, self._policy = explorer, critic, policy
        self._begin(budget, navigate=True)
        self._remember_reset("browser", True, "a fresh browser on the same task")
        log.info("inspect.reset.browser")

    def reset_site(self, recipe: str) -> None:
        """Put the SITE back, using this project's own undo - not a second reset concept.

        ``recipe`` is ``"task"`` for whatever the running task declared (``reset_url``
        and/or ``reset_actions``, which is what the admission gate would use), or the
        stem of a file in :func:`undo_recipes`' directory - ``"splitkb-empty-cart"``.
        The steps run through :func:`~skillweaver.reset_actions.world_reset_from_actions`
        with the same DOM reader the gate hands it, so an inspector reset and a gate
        reset are the same undo and a recipe that works here works there.

        A failure is REPORTED, not raised past the button: a reset that did not converge
        is a fact about the site worth seeing, and it leaves the run usable.

        Raises:
            SessionError: if there is no session, or the recipe is unknown.
        """
        if self._controller is None or self._perceiver is None or self._task is None:
            raise SessionError("There is no session to reset - start a task first.")
        steps, url = self._recipe(recipe)
        if not steps and not url:
            raise SessionError(
                f"{recipe!r} names no undo. This task declared none, so there is "
                "nothing to put back with."
            )
        restore = (
            world_reset_from_actions(
                steps,
                controller=self._controller,
                perceiver=self._perceiver,
                truth=_dom_reader(self._controller),
            )
            if steps
            else None
        )
        if url:
            restore = chain_resets(world_reset_from_url(url), restore)
        report = reset_world(restore)
        self._remember_reset(f"site:{recipe}", report.restored, str(report))
        if url and report.restored:
            # An endpoint restores the SERVER; the page already open was never told. Seen
            # on the sandbox shop: /__state said the cart was empty while the frame still
            # read "$14.50 cart". The gate navigates after its own reset for this reason.
            here = self._controller.url()
            if here and self._controller.supports("navigate"):
                self._controller.perform(Navigate(here))
        self._observe()
        if not report.restored:
            log.warning("inspect.reset.site.failed", recipe=recipe, detail=str(report))
        else:
            log.info("inspect.reset.site", recipe=recipe)

    def undo_recipes(self) -> list[str]:
        """The undo recipes this checkout ships, plus ``"task"`` when the task has one.

        Read from ``undo/*.json`` beside the repository root, which is where the verified
        cart resets live. A checkout without that directory simply offers fewer.
        """
        names = ["task"] if self._task_has_undo() else []
        directory = _undo_dir()
        if directory.is_dir():
            names += sorted(path.stem for path in directory.glob("*.json"))
        return names

    # -- teardown ----------------------------------------------------------------------

    def close(self) -> None:
        """Close the browser and forget the run. Idempotent; never raises."""
        self._conclude()
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                log.warning("inspect.close.failed")
        self._controller = None
        self._watched = None
        self._perceiver = None
        self._explorer = None
        self._critic = None
        self._policy = None
        self._run = None
        self._pending = None
        self._status = "idle"
        self._note = ""
        self._log = []
        self._frame_png = b""
        self._started_at = None
        self._total = _Split()
        self._first_load = _Split()

    # -- the two halves of one step ----------------------------------------------------

    def _decide(self, explorer: Explorer, run: ExplorerRun) -> None:
        """``_ask`` + ``_ground`` + ``_refuse_repeat``: the DECIDE half of the loop.

        A refused answer is recorded exactly as the loop records one - it is a step that
        cost a model call and taught the run something - and the session stays ``ready``
        so the next press asks again with that refusal in hand.
        """
        catalog = ElementCatalog(run.current.elements)
        if self._policy is not None:
            # A driver may answer WITHOUT asking its policy; the decision on file would
            # then be the previous step's, shown beside a move it never chose.
            self._policy.last = None
        before = self._marks()
        try:
            answer = explorer._ask(run.task, run, catalog)  # noqa: SLF001
        except PolicyBlocked as exc:
            self._total += self._split_since(before, asked=True)
            self._stop("blocked", str(exc) or "the policy reported no supported operation")
            return
        split = self._split_since(before, asked=True)
        decide_ms = split.wall_ms
        try:
            move = explorer._ground(answer, catalog, self._watched)  # noqa: SLF001
            explorer._refuse_repeat(move, run)  # noqa: SLF001
        except InvalidAnswer as refusal:
            explorer._reject(run, refusal, answer)  # noqa: SLF001
            self._total += split
            self._record_refusal(run, str(refusal), split)
            self._status = "ready"
            self._note = "The answer was refused; press Choose again."
            return
        self._drop_pending()
        self._pending = _Pending(
            move=move,
            catalog=catalog,
            decide_ms=decide_ms,
            at_fingerprint=run.current.fingerprint.value,
            decision=self._policy.last if self._policy else None,
            split=split,
        )
        self._status = "predicted"
        self._note = ""

    def _perform(self, explorer: Explorer, run: ExplorerRun, pending: _Pending) -> None:
        """``_make_move``: the ACT half. Performs, judges, records, and checks ``done``."""
        assert self._watched is not None
        before, move = run.current, pending.move
        url_before = before.url or ""
        self._watched.forget()
        if self._critic is not None:
            self._critic.last = None
        marks = self._marks()
        try:
            explorer._make_move(run.task, move, pending.catalog, self._watched, run)  # noqa: SLF001
        finally:
            acted = self._split_since(marks, asked=False)
            self._total += pending.split + acted

        performed = list(self._watched.seen)
        verdict = self._critic.last if self._critic else None
        self._log_step(
            run,
            move,
            pending,
            performed,
            verdict,
            url_before,
            before.fingerprint,
            acted,
        )
        self._frame_from(run.current)
        if run.ok:
            self._stop("done", (run.verdict.reason if run.verdict else "the critic agreed"))
            return
        if explorer._exhausted(run):  # noqa: SLF001
            self._stop("stopped", run.detail or "the run reached a budget limit")
            return
        self._status = "ready"
        self._note = run.rejection or ""

    # -- keeping the record ------------------------------------------------------------

    def _log_step(
        self,
        run: ExplorerRun,
        move: Move,
        pending: _Pending,
        performed: Sequence[tuple[Action, ActionResult]],
        verdict: Verdict | None,
        url_before: str,
        before: Fingerprint,
        acted: _Split,
    ) -> None:
        decision = pending.decision
        whole = pending.split + acted
        failed = [result.error for _action, result in performed if not result.ok and result.error]
        self._append(
            StepRecord(
                number=run.moves,
                headline=self._headline(move, pending.catalog, decision),
                summary=move.summary,
                thought=move.thought,
                expect=move.expect,
                claimed_done=move.done,
                signature=move.signature,
                operation=_shown_operation(move, decision),
                target=_target_of(move, pending.catalog, decision),
                confidence=decision.confidence if decision else None,
                probability=decision.probability if decision else None,
                decide_ms=round(pending.decide_ms),
                act_ms=round(acted.wall_ms),
                **whole.as_json(),
                actions=tuple(_action_line(a, r) for a, r in performed),
                delivered=bool(performed) and all(r.ok for _a, r in performed),
                error=failed[0] if failed else (run.rejection if verdict is None else None),
                url_before=url_before,
                url_after=run.current.url or "",
                screen_changed=before.similarity(run.current.fingerprint) < SAME_STATE_THRESHOLD,
                similarity=round(before.similarity(run.current.fingerprint), 3),
                verdict_ok=verdict.ok if verdict else None,
                verdict_reason=verdict.reason if verdict else "",
                verdict_source=verdict.source if verdict else "",
                verdict_confidence=verdict.confidence if verdict else 0.0,
                refused=False,
                at=time.time(),
            )
        )

    def _record_refusal(self, run: ExplorerRun, why: str, split: _Split) -> None:
        """A proposal the loop would not use. It cost a call, so it is a step."""
        self._append(
            StepRecord(
                number=run.moves + 1,
                headline="Answer refused",
                summary="answer refused",
                thought=why,
                expect="",
                claimed_done=False,
                signature="refused",
                operation="REFUSED",
                target="",
                confidence=None,
                probability=None,
                decide_ms=round(split.wall_ms),
                act_ms=0.0,
                **split.as_json(),
                actions=(),
                delivered=False,
                error=why,
                url_before=run.current.url or "",
                url_after=run.current.url or "",
                screen_changed=False,
                similarity=1.0,
                verdict_ok=None,
                verdict_reason="",
                verdict_source="",
                verdict_confidence=0.0,
                refused=True,
                at=time.time(),
            )
        )

    # -- whose time it was -------------------------------------------------------------

    def _drop_pending(self) -> None:
        """Let go of a choice that will never be performed - re-asked, outrun by the page,
        or undone by a reset. It was paid for, so its time stays in the run's totals even
        though no step will ever carry it."""
        if self._pending is not None:
            self._total += self._pending.split
        self._pending = None

    def _site_clock(self) -> float | None:
        """``DomPerceiver.site_ms``, or ``None`` from a perceiver that keeps no such clock."""
        value = getattr(self._perceiver, "site_ms", None)
        return float(value) if isinstance(value, (int, float)) else None

    def _marks(self) -> _Marks:
        policy = self._explorer._policy if self._explorer is not None else None  # noqa: SLF001
        model = getattr(policy, "policy_ms", 0.0)
        watched = self._watched
        return _Marks(
            at=time.perf_counter(),
            model_ms=float(model) if isinstance(model, (int, float)) else 0.0,
            site_ms=self._site_clock(),
            frame_ms=watched.capture_ms if watched else 0.0,
            performed_ms=watched.performed_ms if watched else 0.0,
        )

    def _split_since(self, before: _Marks, *, asked: bool) -> _Split:
        """The stretch from ``before`` to now, shared out. See the module docstring.

        ``asked`` says the stretch was the DECIDE half. With no acting policy that half
        is one call to the prompted model and nothing else, so all of it is model time;
        with one, the policy's own clock says how much was.
        """
        now = self._marks()
        wall = (now.at - before.at) * 1000
        frame = now.frame_ms - before.frame_ms
        policy = self._explorer._policy if self._explorer is not None else None  # noqa: SLF001
        if hasattr(policy, "policy_ms"):
            model = now.model_ms - before.model_ms
        else:
            model = wall - frame if asked else 0.0
        if now.site_ms is not None and before.site_ms is not None:
            # Every capture is taken inside an observation, which that clock counts.
            site = now.site_ms - before.site_ms - frame
        else:
            site = now.performed_ms - before.performed_ms
        return _Split.of(wall, model, site, frame)

    def _goal_shown(self) -> str:
        """The refined goal the policy is being shown, or ``""`` when it sees the task as
        typed. Read off the driver tolerantly: it is another module's private memo, and
        the page loses a line, not a run, if it moves."""
        policy = self._explorer._policy if self._explorer is not None else None  # noqa: SLF001
        refined = getattr(policy, "_refined", None)
        task = self._task
        if not isinstance(refined, dict) or task is None:
            return ""
        shown = [str(v) for v in refined.values() if str(v) != task.text]
        return shown[-1] if shown else ""

    def _append(self, record: StepRecord) -> None:
        self._log.append(record)
        if len(self._log) > MAX_LOG_STEPS:
            del self._log[: len(self._log) - MAX_LOG_STEPS]

    # -- starting and stopping a run ---------------------------------------------------

    def _begin(self, budget: Budget, *, navigate: bool) -> None:
        """A fresh :class:`_Run` against the open browser, exactly as ``explore`` opens one."""
        explorer, task = self._explorer, self._task
        assert explorer is not None and task is not None and self._perceiver is not None
        assert self._controller is not None
        self._conclude()
        explorer._recorder.start(task.text, task.domain)  # noqa: SLF001
        self._concluded = False
        self._total = _Split()
        assert self._watched is not None
        marks = self._marks()
        if navigate:
            self._go_to_start(task)
        # Through the watcher, like every later observation, so this frame is timed too.
        observation = self._perceiver.observe(self._watched)
        self._first_load = self._split_since(marks, asked=False)
        run = ExplorerRun(
            task=task,
            spend=Spend(budget).start(),
            memory=FailureMemory(),
            usage_mark=explorer._spent(),  # noqa: SLF001
            first=observation,
            current=observation,
        )
        explorer._remember_state(task, observation)  # noqa: SLF001
        run.skills = explorer._retrieve(task)  # noqa: SLF001
        explorer._adopt_skeleton(task, run)  # noqa: SLF001
        self._run = run
        self._pending = None
        self._log = []
        self._started_at = time.perf_counter()
        self._status = "ready"
        self._note = ""
        self._frame_from(observation)

    def _go_to_start(self, task: TaskSpec) -> None:
        """Load the task's start URL. Best effort, exactly as the explorer's own is."""
        url = task.params.get("start_url")
        controller = self._controller
        if not isinstance(url, str) or not url or controller is None:
            return
        if not controller.supports("navigate") or controller.url() == url:
            return
        result = controller.perform(Navigate(url))
        if not result.ok:
            log.info("inspect.start_url.failed", url=url, error=result.error)

    def _stop(self, status: str, note: str) -> None:
        self._status = status
        self._note = note
        self._pending = None
        if self._run is not None and status == "blocked":
            self._run.stopped_by, self._run.detail = "blocked", note
        self._conclude()
        log.info("inspect.stopped", status=status, note=note)

    def _conclude(self) -> None:
        """Close the run's trajectory, exactly as :meth:`Explorer.explore` does on return.

        EVERY ending comes through here - solved, blocked, out of budget, reset, closed -
        because a :class:`~skillweaver.trajectory.record.Recorder` refuses to start a run
        while one is open, and a stepped loop has no ``return`` to close it on. Found by
        pressing Reset run after a finished task on live Wikipedia: ``run ... is still in
        progress``, and the button did nothing. A run abandoned by a reset is recorded as
        the failure it is (``stopped_by="gave-up"``), which keeps a half-finished
        demonstration out of anything that later reads trajectories as evidence.
        """
        explorer, run = self._explorer, self._run
        if explorer is None or run is None or self._concluded:
            return
        self._concluded = True
        try:
            outcome = explorer._finish(run)  # noqa: SLF001 - see the module docstring
        except SkillWeaverError as exc:
            log.warning("inspect.conclude.failed", error=str(exc))
            return
        self._last_run_id = outcome.trajectory.run_id

    def _live(self) -> tuple[Explorer, ExplorerRun]:
        """The explorer and run, or a sentence saying why there is not one.

        Raises:
            SessionError: when idle or terminal.
        """
        if self._explorer is None or self._run is None:
            raise SessionError("Start a task first - there is no run to step.")
        if self._status in TERMINAL:
            ended = {"done": "is finished", "blocked": "is blocked", "stopped": "has stopped"}
            raise SessionError(
                f"This run {ended[self._status]}, so there is nothing to step. "
                "Reset the run to start another."
            )
        return self._explorer, self._run

    # -- observation and rendering -----------------------------------------------------

    def _observe(self) -> None:
        """Re-observe without stepping - what a reset needs so the page shows the undo."""
        if self._perceiver is None or self._controller is None or self._run is None:
            return
        observation = self._perceiver.observe(self._watched or self._controller)
        self._run.current = observation
        self._drop_pending()
        if self._status == "predicted":
            self._status = "ready"
        self._frame_from(observation)

    def _frame_from(self, observation: Observation) -> None:
        self._frame_png = observation.screenshot.png
        self._frame += 1

    def _elements_json(self) -> list[dict[str, Any]]:
        """Every element on the current screen, under the id an answer would name it by."""
        run = self._run
        if run is None:
            return []
        catalog = ElementCatalog(run.current.elements)
        controls = self._controls_by_id()
        out: list[dict[str, Any]] = []
        for element in run.current.elements:
            element_id = catalog.id_for(element)
            control = controls.get(element_id or "")
            out.append(
                {
                    "id": element_id or "",
                    "kind": str(element.kind),
                    "text": element.text,
                    "box": [element.box.x, element.box.y, element.box.w, element.box.h],
                    "role": control.role if control else "",
                    "value": control.value if control else "",
                    "index": control.index if control else None,
                    "withheld": self._withheld_for(element_id),
                }
            )
        return out

    def _controls_by_id(self) -> Mapping[str, Any]:
        """The DOM controls behind this screen, when the DOM path is driving."""
        snapshot = self._policy.last_snapshot if self._policy else None
        if snapshot is None:
            snapshot = getattr(self._perceiver, "last", None)
        return getattr(snapshot, "by_element_id", {}) or {}

    def _withheld_for(self, element_id: str | None) -> list[str]:
        """Which operations this screen has stopped offering for one control.

        This is the one thing a target table in THIS project can say that a probability
        bar cannot: a dead end is withheld from the policy's target head, so a control
        that is still visible may no longer be choosable. See
        ``jev_driver._exclusions``.
        """
        if not element_id or self._policy is None:
            return []
        held = [op for op, ids in self._policy.last_exclude.items() if element_id in ids]
        label = getattr(self._controls_by_id().get(element_id), "label", "")
        if label and label in self._policy.last_reserved.get("LABELS", ()):
            held.append("every operation (label churned)")
        return held

    def _decision_json(self) -> dict[str, Any] | None:
        """The held decision, or ``None``. See the module docstring on what is absent."""
        pending = self._pending
        if pending is None:
            return None
        move, decision = pending.move, pending.decision
        return {
            "headline": self._headline(move, pending.catalog, decision),
            "summary": move.summary,
            "thought": move.thought,
            "expect": move.expect,
            "done": move.done,
            "signature": move.signature,
            "target": _target_of(move, pending.catalog, decision),
            "operation": _shown_operation(move, decision),
            "confidence": decision.confidence if decision else None,
            "probability": decision.probability if decision else None,
            "policy_ms": round(decision.policy_ms) if decision else None,
            "latency_ms": round(decision.latency_ms) if decision else round(pending.decide_ms),
            "text": decision.text if decision else None,
            "action": action_to_dict(move.action) if move.action else None,
            "code": move.code,
            "withheld": {
                op: list(ids)
                for op, ids in (self._policy.last_exclude if self._policy else {}).items()
            },
            "withheld_controls": list(
                self._policy.last_reserved.get("CONTROLS", ()) if self._policy else ()
            ),
            "withheld_labels": list(
                self._policy.last_reserved.get("LABELS", ()) if self._policy else ()
            ),
            "fresh_look": _is_fresh_look(move, decision),
            "ranked": self._policy is not None,
            "decided_by": self._explorer_model(),
        }

    def _headline(
        self, move: Move, catalog: ElementCatalog, decision: PolicyDecision | None
    ) -> str:
        """One move as a person would say it: ``TYPE_TEXT "Ada Lovelace" -> Search Wikipedia``.

        ``Move.summary`` is written for the model that will be asked again, and for a
        typed move it is the first line of a code block. This is the same move named by
        the control's own label, and it invents nothing: the label comes from the page's
        snapshot, falling back to the element's text and then to its id.
        """
        if _is_fresh_look(move, decision):
            return "DONE claimed → one fresh look before it is accepted"
        operation = _shown_operation(move, decision)
        element_id = decision.element_id if decision else None
        if element_id is None:
            aimed = signature_move(move.signature)
            element_id = aimed[1] if aimed else None
        if not element_id:
            return operation if move.acts or move.done else move.summary
        control = self._controls_by_id().get(element_id)
        label = getattr(control, "label", "")
        if not label:
            try:
                label = catalog.get(element_id).text
            except Exception:  # noqa: BLE001 - an id off this screen still names the move
                label = ""
        typed = f" “{decision.text}”" if decision is not None and decision.text else ""
        return f"{operation}{typed} → {label or element_id}"

    def _explorer_model(self) -> str:
        explorer = self._explorer
        if explorer is None:
            return ""
        policy = explorer._policy  # noqa: SLF001
        return str(policy.name() if policy is not None else explorer._llm.name())  # noqa: SLF001

    # -- resets ------------------------------------------------------------------------

    def _recipe(self, recipe: str) -> tuple[tuple[Any, ...], str]:
        """``(steps, url)`` for a named undo.

        Raises:
            SessionError: if the name is unknown or the file does not parse.
        """
        task = self._task
        assert task is not None
        if recipe == "task":
            steps = reset_steps_from(task.params.get(RESET_ACTIONS_PARAM) or ())
            return steps, str(task.params.get(RESET_URL_PARAM) or "")
        path = _undo_dir() / f"{recipe}.json"
        if "/" in recipe or ".." in recipe or not path.is_file():
            raise SessionError(f"There is no undo recipe called {recipe!r}.")
        try:
            return reset_steps_from(path.read_text(encoding="utf-8")), ""
        except (ValueError, OSError) as exc:
            raise SessionError(f"{recipe}.json does not parse: {exc}") from exc

    def _task_has_undo(self) -> bool:
        task = self._task
        if task is None:
            return False
        return bool(task.params.get(RESET_ACTIONS_PARAM) or task.params.get(RESET_URL_PARAM))

    def _remember_reset(self, what: str, ok: bool, detail: str) -> None:
        self._resets.append({"what": what, "ok": ok, "detail": detail, "at": time.time()})


# --------------------------------------------------------------------------------------
# Small renderers
# --------------------------------------------------------------------------------------


def _has_inner_policy(policy: Any) -> bool:
    """Whether ``policy`` is a :class:`~skillweaver.agent.jev_driver.JevDriver`.

    Asked by shape rather than by import so the inspector still starts when the Jev path
    is not installed or not configured, which is the common case on the pixel default.
    """
    return policy is not None and hasattr(policy, "_policy") and hasattr(policy, "propose")


def _dom_reader(controller: Controller) -> Any:
    """The DOM reader a ``via="dom"`` reset step needs, or ``None``.

    The same single call site rule the orchestrator's ``_dom_of`` keeps: scaffolding that
    puts the world back may read the page's own names, and nothing else here may. See
    ``AGENTS.md`` on why a real site's cart button has no readable name.
    """
    from skillweaver.controllers.browser import BrowserGroundTruth

    reads_pages = callable(getattr(controller, "evaluate", None))
    return BrowserGroundTruth(controller) if reads_pages else None


def _undo_dir() -> Path:
    """Where this checkout keeps its verified undo recipes."""
    return Path.cwd() / "undo"


def _operation_of(move: Move) -> str:
    """A move's operation in Jev's vocabulary, for the path that has no policy."""
    if move.code is not None:
        return "CODE"
    if move.action is None:
        return "DONE" if move.done else "NONE"
    return str(getattr(move.action, "kind", "")).upper()


def _is_fresh_look(move: Move, decision: PolicyDecision | None) -> bool:
    """Whether this move is the driver's second look at a ``DONE`` claim.

    The driver makes a ``DONE`` survive one fresh observation before passing it on - a
    cart badge or a navigation may not have landed - and what reaches the loop is a short
    wait that claims nothing. Shown raw that is a policy saying ``DONE`` beside an
    executed ``wait 150ms``, which reads as a bug. Recognised by that same disagreement,
    so it needs nothing from the driver and says nothing when the driver does not do it.
    """
    if decision is None or decision.operation != "DONE" or move.done:
        return False
    return isinstance(move.action, Wait)


def _shown_operation(move: Move, decision: PolicyDecision | None) -> str:
    """The operation to SHOW: the policy's, unless the move it became is a different one."""
    if decision is None or _is_fresh_look(move, decision):
        return _operation_of(move)
    return decision.operation


def _target_of(move: Move, catalog: ElementCatalog, decision: PolicyDecision | None) -> str:
    """What the move aims at, named the way the page names it."""
    if decision is not None and decision.element_id:
        return decision.element_id
    aimed = signature_move(move.signature)
    if aimed is None:
        return ""
    element_id = aimed[1]
    try:
        element = catalog.get(element_id)
    except Exception:  # noqa: BLE001 - a target off this screen is still worth naming
        return element_id
    return f"{element_id} ({element.text})" if element.text else element_id


def _action_line(action: Action, result: ActionResult) -> str:
    """One performed action as a line: what was sent, and what came back."""
    body = json.dumps(action_to_dict(action), separators=(",", ":"))
    outcome = "ok" if result.ok else f"failed: {result.error}"
    return f"{body} -> {outcome} ({result.elapsed_ms:.0f}ms)"


def _box_json(shot: Screenshot | None) -> list[int]:
    """The logical size of the frame, so the page can place overlay boxes as percentages."""
    return [shot.width, shot.height] if shot is not None else [0, 0]


def _spend_json(spend: Spend | None) -> dict[str, Any]:
    """What this run has spent against what it is allowed."""
    if spend is None:
        return {}
    budget = spend.budget
    return {
        "steps": spend.steps,
        "max_steps": budget.max_steps,
        "seconds": round(spend.elapsed_seconds(), 1),
        "max_seconds": budget.max_seconds,
        "usd": round(spend.usd, 4),
        "max_usd": budget.max_usd,
        "calls": spend.llm_calls,
        "max_llm_calls": budget.max_llm_calls,
    }
