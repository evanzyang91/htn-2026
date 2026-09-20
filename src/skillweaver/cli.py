"""``skillweaver``: the command line, a thin shell over ``orchestrator``.

No decision is made here - it parses arguments, opens a ``Workbench``, calls one
function and prints what came back.

Exit codes: ``0`` it did what it said; ``1`` it ran and the answer was no (task not
solved, no such skill, unknown run id); ``2`` it could not run at all (bad
configuration, a subcommand whose module has not landed).

Configuration comes from ``config`` - the environment and ``.env`` - and flags override
it per invocation; there is no second mechanism, and a flag that was not given never
resets a configured limit. ``--data-dir``, ``--log-level``, ``--headless`` and
``--chrome-profile`` are properties of the WHOLE invocation, so they go BEFORE the
subcommand and one of them covers ``learn``, ``run`` and ``eval run`` alike::

    skillweaver --headless eval run --suite eval/wikipedia.yaml
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from skillweaver.config import BROWSERS, POLICIES, Settings, check_settings, load_settings, settings
from skillweaver.contracts import Skill, Transition, UIState, action_to_dict
from skillweaver.errors import ConfigError, SkillNotFound, SkillWeaverError
from skillweaver.orchestrator import (
    DomainChoice,
    RunReport,
    Workbench,
    budget_from,
    build_workbench,
    resolve_domain,
    task_spec,
)
from skillweaver.perception_mode import PATHS

__all__ = ["app", "main"]

OK, NO, CANNOT = 0, 1, 2
"""Exit codes: it worked; it ran and the answer was no; it could not run at all."""


app = typer.Typer(
    name="skillweaver",
    no_args_is_help=True,
    add_completion=False,
    help=(
        "A computer-use agent that learns a task once and then repeats it from memory.\n"
        "\n"
        "Teach it with `learn`, which explores the task by trial and error and keeps "
        "what worked as a reusable skill. Then `run` the same task: it retrieves that "
        "skill and does the job with no model in the action loop. `skills` and `graph` "
        "show you what it remembers; `dashboard build` turns all of it into one page."
    ),
)

skills_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and prune the skill library - what the agent has learned to do.",
)
graph_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect the per-site action graph - the screens the agent knows and how to "
    "get between them.",
)
dashboard_app = typer.Typer(no_args_is_help=True, help="Build the review dashboard.")
eval_app = typer.Typer(no_args_is_help=True, help="Run the evaluation harness.")

app.add_typer(skills_app, name="skills")
app.add_typer(graph_app, name="graph")
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(eval_app, name="eval")


@app.callback()
def root(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Directory holding skills/, graphs/, trajectories/ and eval/. "
            "Overrides SKILLWEAVER_DATA_DIR for this invocation.",
            show_default=False,
        ),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="DEBUG, INFO, WARNING or ERROR. Overrides SKILLWEAVER_LOG_LEVEL.",
            show_default=False,
        ),
    ] = None,
    headless: Annotated[
        bool | None,
        typer.Option(
            "--headless/--headed",
            help="Run the browser without a visible window. Applies to learn, run and "
            "eval run. The default is HEADED, because a browser you can watch is what "
            "makes the agent legible; --headless is what a suite, CI or an SSH session "
            "wants. Overrides SKILLWEAVER_HEADLESS. NOTE: the two modes render one page "
            "differently enough that a skill learned in one cannot match a screen in "
            "the other - a run that crosses them says so rather than failing quietly.",
            show_default=False,
        ),
    ] = None,
    chrome_profile: Annotated[
        Path | None,
        typer.Option(
            "--chrome-profile",
            help="Drive the REAL Google Chrome on this machine out of this persistent "
            "profile directory, instead of the bundled Chromium with a throwaway "
            "profile. Applies to learn, run and eval run. Two ordinary browser "
            "settings, and what a site that refuses an automated browser needs; the "
            "directory persists, so a verification page cleared by hand once is not "
            "cleared again next run. Nothing here defeats such a page - one that "
            "appears fails the run. GIVE EACH RUN ITS OWN DIRECTORY: a profile is "
            "exclusive, and concurrent runs sharing one fight over the lock. Overrides "
            "SKILLWEAVER_CHROME_PROFILE.",
            show_default=False,
        ),
    ] = None,
    chrome_attach: Annotated[
        bool | None,
        typer.Option(
            "--chrome-attach/--no-chrome-attach",
            help="Start that real Chrome as an ORDINARY PROCESS and attach to it over "
            "Chrome's own debugging interface, instead of letting the automation "
            "framework launch it. Needs --chrome-profile, and applies to learn, run and "
            "eval run. This is the configuration a live DoorDash serves, and the "
            "difference is the flags the framework adds when IT starts the browser - "
            "started plainly, Chrome carries none of them. It makes us stop announcing "
            "ourselves; it does NOT make a site accept us, and nothing here defeats, "
            "masks or retries past a human-verification page - one that appears fails "
            "the run for a person to clear by hand. Overrides "
            "SKILLWEAVER_CHROME_ATTACH.",
            show_default=False,
        ),
    ] = None,
    perception: Annotated[
        str | None,
        typer.Option(
            "--perception",
            help="Which eyes to use: 'pixels' (a YOLO detector plus OCR over a "
            "screenshot - the DEFAULT, and what every stored skill was learned "
            "against) or 'dom' (the page's own list of named controls, browser only). "
            "The DOM path runs no OCR, which is 84-97% of a pixel observation, and it "
            "sees nothing the page declines to name - text in an image, a canvas, a "
            "cross-origin frame. THE TWO KEEP SEPARATE SKILL LIBRARIES: a skill stored "
            "by one is never offered to the other, because its code was written against "
            "element text a different reader produced. Overrides SKILLWEAVER_PERCEPTION.",
            show_default=False,
        ),
    ] = None,
    policy: Annotated[
        str | None,
        typer.Option(
            "--policy",
            help="Who chooses each exploratory move: 'claude' (the acting prompt - the "
            "DEFAULT) or 'jev' (TypeSafe's Jev, one request per step that picks both "
            "the operation and its target). 'jev' requires --perception dom and "
            "TYPESAFE_API_KEY. It replaces the move decision ONLY: the critic, the "
            "admission gate, the synthesizer and every stored skill are unchanged. "
            "Overrides SKILLWEAVER_POLICY.",
            show_default=False,
        ),
    ] = None,
    browser: Annotated[
        str | None,
        typer.Option(
            "--browser",
            help="Which browser a run drives: 'harness' (the DEFAULT - the Chrome you "
            "already have open, through Browser Harness and raw DevTools-protocol calls, "
            "with no automation framework; the run works in a background tab of its own "
            "and closes it) or 'playwright' (a browser this project starts, which most "
            "real shops refuse). 'harness' IS YOUR REAL, LOGGED-IN BROWSER: a task "
            "addressed to it can act on your accounts. Chrome asks once to allow remote "
            "debugging. --headless and --chrome-profile describe a browser this project "
            "starts, so either one selects 'playwright'. Overrides SKILLWEAVER_BROWSER.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Open the memories every subcommand reads from. Nothing expensive happens here."""
    if ctx.obj is not None:  # a caller (a test, an embedder) supplied its own
        return
    try:
        ctx.obj = build_workbench(
            _settings(
                data_dir,
                log_level,
                headless,
                chrome_profile,
                chrome_attach,
                perception,
                policy,
                browser,
            )
        )
    except ConfigError as exc:
        _die(f"configuration is invalid: {exc}")


def _settings(
    data_dir: Path | None,
    log_level: str | None,
    headless: bool | None = None,
    chrome_profile: Path | None = None,
    chrome_attach: bool | None = None,
    perception: str | None = None,
    policy: str | None = None,
    browser: str | None = None,
) -> Settings:
    """Configuration with the global flags on top. Every flag is TRI-STATE: ``None`` means
    it was not written and the configured value stands. A plain ``bool`` for
    ``--headless/--headed`` would make every silent invocation say ``--headed`` and
    override a ``SKILLWEAVER_HEADLESS=1`` set on purpose."""
    resolved = load_settings()
    changes: dict[str, Any] = {}
    if data_dir is not None:
        changes["data_dir"] = Path(data_dir)
    if log_level is not None:
        level = log_level.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError(f"--log-level {log_level!r} is not a log level")
        changes["log_level"] = level
    if headless is not None:
        changes["headless"] = bool(headless)
    if chrome_profile is not None:
        changes["chrome_profile"] = Path(chrome_profile).expanduser()
    if chrome_attach is not None:
        changes["chrome_attach"] = bool(chrome_attach)
    if perception is not None:
        chosen = perception.lower()
        if chosen not in PATHS:
            raise ConfigError(f"--perception {perception!r} must be one of {PATHS}")
        changes["perception"] = chosen
    if policy is not None:
        chosen = policy.lower()
        if chosen not in POLICIES:
            raise ConfigError(f"--policy {policy!r} must be one of {POLICIES}")
        changes["policy"] = chosen
    if browser is not None:
        chosen = browser.lower()
        if chosen not in BROWSERS:
            raise ConfigError(f"--browser {browser!r} must be one of {BROWSERS}")
        changes["browser"] = chosen
    if not changes:
        return resolved
    # Re-checked after the flags land: --chrome-attach and the directory it needs can
    # arrive from opposite sides, one in the environment and one on the command line.
    return check_settings(dataclasses.replace(resolved, **changes))


# --------------------------------------------------------------------------------------
# learn / run
# --------------------------------------------------------------------------------------

TaskArg = Annotated[str, typer.Argument(help="What to do, in plain English.", show_default=False)]
TargetOpt = Annotated[
    str,
    typer.Option("--target", help="Which world to drive: 'browser' or 'desktop'."),
]
UrlOpt = Annotated[
    str | None,
    typer.Option(
        "--url",
        help="Page to start on. Also sets the domain the skill is filed under, "
        "unless --domain says otherwise. Browser tasks only.",
        show_default=False,
    ),
]
ResetUrlOpt = Annotated[
    str | None,
    typer.Option(
        "--reset-url",
        help="A URL that puts this site back to its starting state - the sandbox "
        "site's http://localhost:8765/__reset, a demo app's 'restore seed data' "
        "endpoint. The admission gate calls it before it re-runs a candidate skill. "
        "WITHOUT IT, A TASK THAT CHANGES ANYTHING CANNOT BE LEARNED: archiving a "
        "message is not undone by re-opening the inbox, so the gate can never stand "
        "where the recording stood and no skill is ever stored.",
        show_default=False,
    ),
]
ResetStepsOpt = Annotated[
    str | None,
    typer.Option(
        "--reset-steps",
        help="An undo PERFORMED on the screen, for a site that has no reset endpoint "
        "and never will: a JSON array of steps, or a path to a .json file holding "
        "one. Each step is an ordinary action - the same vocabulary a skill uses - "
        "plus 'find' to aim it at whatever the page SAYS rather than at a pixel, and "
        "'until_gone' to repeat it until that text is off the screen. That loop is "
        "what makes an undo safe to run between the gate's re-runs: on an already "
        "clean world it performs nothing, and a loop that cannot converge reports a "
        "FAILED reset rather than handing the gate a dirty screen. Combines with "
        "--reset-url; the endpoint runs first. Example: empty a cart with "
        '\'[{"kind": "click", "find": "cart"}, {"kind": "click", '
        '"find": "$ each", "dx": 660, "until_gone": "each"}]\'.',
        show_default=False,
    ),
]
DomainOpt = Annotated[
    str | None,
    typer.Option(
        "--domain",
        help="Site or app the task belongs to ('acme.test', 'desktop:finder'). "
        "Defaults to the host of --url. With neither, the whole library is searched "
        "and a stored skill that accounts for this task names its own domain - which "
        "is what lets a warm repeat drop the --url the learn run needed.",
        show_default=False,
    ),
]
ParamOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--param",
        "-p",
        help="A concrete value the task needs, as KEY=VALUE. Repeatable. These are "
        "what a stored skill's parameters are bound from on the warm path, so "
        "-p company='Acme Corp' is what makes the next run model-free.",
        show_default=False,
    ),
]
StepsOpt = Annotated[
    int | None, typer.Option("--max-steps", help="Cap on actions.", show_default=False)
]
SecondsOpt = Annotated[
    float | None,
    typer.Option("--max-seconds", help="Cap on wall-clock seconds.", show_default=False),
]
UsdOpt = Annotated[
    float | None,
    typer.Option("--max-usd", help="Cap on model spend, in US dollars.", show_default=False),
]
CallsOpt = Annotated[
    int | None,
    typer.Option("--max-llm-calls", help="Cap on model calls.", show_default=False),
]
SkillSecondsOpt = Annotated[
    float | None,
    typer.Option(
        "--skill-max-seconds",
        help="Cap on the seconds a STORED SKILL may spend running its own code - "
        "the sandbox's runaway-loop tripwire, not the run budget that --max-seconds "
        "sets. Time the skill sits waiting on a screenshot or on OCR is not counted "
        "against it, so raise this only for a skill that genuinely computes. "
        "Overrides SKILLWEAVER_SKILL_MAX_SECONDS for this invocation.",
        show_default=False,
    ),
]
JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Print machine-readable JSON instead of prose."),
]


@app.command("learn")
def learn_command(
    ctx: typer.Context,
    task: TaskArg,
    target: TargetOpt = "browser",
    url: UrlOpt = None,
    domain: DomainOpt = None,
    reset_url: ResetUrlOpt = None,
    reset_steps: ResetStepsOpt = None,
    param: ParamOpt = None,
    warm_first: Annotated[
        bool,
        typer.Option(
            "--warm-first/--always-explore",
            help="Let the library answer first if it already can. The default explores "
            "even when a skill exists, which is what 'learn' should mean.",
        ),
    ] = False,
    max_steps: StepsOpt = None,
    max_seconds: SecondsOpt = None,
    max_usd: UsdOpt = None,
    max_llm_calls: CallsOpt = None,
    skill_max_seconds: SkillSecondsOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Learn a task by trial and error, and keep what worked as a reusable skill.

    The slow path: a model proposes each move, the agent performs and judges it, and the
    whole run is recorded. On success the admission gate rewrites the recording as a Python
    skill, RE-RUNS it from the starting screen, and stores it only if it works a second
    time - nothing enters the library on one lucky run.

    That re-run needs the starting screen back, so a task that CHANGES anything must say
    how to undo it or nothing is stored. `--reset-url` is one GET that restores the
    application; `--reset-steps` is the undo performed ON the screen, which is the only
    kind a real site offers. A task that only reads should say `-p read_only=true`.

    Afterwards, `skillweaver run` on the same task takes the warm path.

        skillweaver learn "Confirm payment of the Acme Corp invoice" \\
            --url https://acme.test/invoices --reset-url https://acme.test/__reset \\
            -p company="Acme Corp"

        skillweaver learn "Order a Pad Thai from Copper Kettle" \\
            --url https://food.test/ --reset-steps undo/empty-cart.json
    """
    _do(
        ctx,
        task,
        target=target,
        url=url,
        domain=domain,
        reset_url=reset_url,
        reset_steps=reset_steps,
        param=param,
        warm=warm_first,
        cold=True,
        learn=True,
        budget_flags=(max_steps, max_seconds, max_usd, max_llm_calls),
        skill_max_seconds=skill_max_seconds,
        as_json=as_json,
    )


@app.command("run")
def run_command(
    ctx: typer.Context,
    task: TaskArg,
    target: TargetOpt = "browser",
    url: UrlOpt = None,
    domain: DomainOpt = None,
    reset_url: ResetUrlOpt = None,
    reset_steps: ResetStepsOpt = None,
    param: ParamOpt = None,
    library_only: Annotated[
        bool,
        typer.Option(
            "--library-only",
            help="Never explore. The run fails honestly when the library cannot do "
            "the task, instead of paying a model to work it out.",
        ),
    ] = False,
    no_learn: Annotated[
        bool,
        typer.Option(
            "--no-learn",
            help="Do not offer a successful exploration to the admission gate. "
            "Use when you want the run without growing the library.",
        ),
    ] = False,
    max_steps: StepsOpt = None,
    max_seconds: SecondsOpt = None,
    max_usd: UsdOpt = None,
    max_llm_calls: CallsOpt = None,
    skill_max_seconds: SkillSecondsOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Do a task the fastest way the agent knows.

    The warm path runs first: retrieve a stored skill, route to the screen it starts on,
    run its Python, verify. A warm hit consults NO model in the action loop, and the report
    says so with a call count.

    `--url` is not needed for a task already learned. With neither `--url` nor `--domain`
    the library is searched across every domain, and a skill that accounts for this request
    names both the domain and the page to open; without that a repeat would look under the
    literal name of the target and never find it.

    A library that cannot plan the task - or plans it, runs it, and fails verification -
    falls through to exploration, and that is ALWAYS reported rather than printed as a
    plain success.
    """
    _do(
        ctx,
        task,
        target=target,
        url=url,
        domain=domain,
        reset_url=reset_url,
        reset_steps=reset_steps,
        param=param,
        warm=True,
        cold=not library_only,
        learn=not no_learn,
        budget_flags=(max_steps, max_seconds, max_usd, max_llm_calls),
        skill_max_seconds=skill_max_seconds,
        as_json=as_json,
    )


def _do(
    ctx: typer.Context,
    text: str,
    *,
    target: str,
    url: str | None,
    domain: str | None,
    reset_url: str | None,
    reset_steps: str | None,
    param: Sequence[str] | None,
    warm: bool,
    cold: bool,
    learn: bool,
    budget_flags: tuple[int | None, float | None, float | None, int | None],
    skill_max_seconds: float | None,
    as_json: bool,
) -> None:
    """Open a session, run one task, print the report, exit with its verdict."""
    bench = _bench(ctx)
    if target not in ("browser", "desktop"):
        _die(f"--target must be 'browser' or 'desktop', not {target!r}")
    _apply_skill_seconds(skill_max_seconds)
    steps, seconds, usd, calls = budget_flags
    budget = budget_from(
        bench.settings,
        max_steps=steps,
        max_seconds=seconds,
        max_usd=usd,
        max_llm_calls=calls,
    )
    params = _params(param)
    where = resolve_domain(
        text,
        target=target,  # type: ignore[arg-type]
        url=url,
        domain=domain,
        params=params,
        retriever=bench.retriever if warm else None,
        graph=bench.graph if warm else None,
        path=bench.settings.perception,
    )
    try:
        spec = task_spec(
            text,
            domain=where.domain,
            target=target,  # type: ignore[arg-type]
            url=where.start_url,
            reset_url=reset_url,
            reset_steps=_reset_steps(reset_steps),
            params=params,
        )
    except ValueError as exc:
        # Before a browser opens and before a model is paid: finding a typo in the undo
        # at the admission gate would cost the whole run.
        _die(f"--reset-steps: {exc}")
    try:
        with bench.session(spec, budget) as agent:
            report = agent.run(spec, learn=learn, warm=warm, cold=cold)
    except SkillWeaverError as exc:
        _die(f"the run could not start: {exc}")
    if where.looked_up and not as_json:
        _say_where(where)
    _emit(_report_json(report, where) if as_json else report.explain(), as_json)
    if not as_json:
        _hint_at_cross_mode(report)
        _hint_at_reset(report, reset_url or reset_steps)
    raise typer.Exit(OK if report.ok else NO)


def _say_where(where: DomainChoice) -> None:
    """Announce a domain the caller did not name and the library did - a decision made on
    their behalf that nobody could otherwise correct."""
    typer.echo(
        f"resolved: {where.domain} - {where.why}"
        + (f"\n          starting at {where.start_url}" if where.start_url else "")
    )


def _apply_skill_seconds(seconds: float | None) -> None:
    """Set the env var rather than pass an argument: the limit is read by
    ``skills.api.default_max_seconds`` when a ``SkillLimits`` is built, several layers below
    anything this file could pass down, and one configuration mechanism is the promise."""
    if seconds is None:
        return
    if seconds <= 0:
        _die(f"--skill-max-seconds must be greater than zero, got {seconds!r}")
    os.environ["SKILLWEAVER_SKILL_MAX_SECONDS"] = repr(float(seconds))
    settings.cache_clear()


def _hint_at_cross_mode(report: RunReport) -> None:
    """Repeat the render-mode crossing on its own line. The attempt line already carries it
    in brackets, but that line is dense and this is the rare finding a reader must not scan
    past: the library is INTACT and one flag fixes the run."""
    warm = report.warm
    if warm is None or warm.cross_mode is None:
        return
    typer.echo(f"\nhint: {warm.cross_mode}", err=True)


def _hint_at_reset(report: RunReport, undo: str | None) -> None:
    """Say what to do when the task was DONE but could not be learned - usually the task
    changed something nothing could change back, which is one flag away from working."""
    admission = report.admission
    if admission is None or not admission.unproved or undo:
        return
    typer.echo(
        "\nhint: this task changes something, so the admission gate could not put the "
        "site back to re-run the skill it wrote. Pass --reset-url with an endpoint "
        "that restores this site (the sandbox site has /__reset); on a site that has "
        "no such endpoint, pass --reset-steps with the actions that undo it - "
        "'until_gone' makes those steps safe to run between the gate's re-runs. Then "
        "learn it again.",
        err=True,
    )


def _reset_steps(given: str | None) -> Any:
    """``--reset-steps`` as JSON ``task_spec`` can parse; a value naming a readable file is
    read from it, and anything else passes through so ONE place decides what a step is."""
    if given is None:
        return None
    text = given.strip()
    if text.startswith(("[", "{")):
        return text
    path = Path(text).expanduser()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _die(
            f"--reset-steps is a JSON array or a path to a file holding one, and "
            f"{text!r} is neither: {exc}"
        )


def _params(pairs: Sequence[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` as a mapping, parsing JSON values when they parse: ``-p count=3`` gives
    an int, so a skill with a numeric parameter binds without a second flag."""
    out: dict[str, Any] = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            _die(f"--param expects KEY=VALUE, got {pair!r}")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def _recorded_in(bench: Workbench, skill: Skill) -> str:
    """``" (recorded headless)"`` when the library knows, else ``""``. Printed against
    "starts on" because it qualifies THAT screen, which only the same renderer can match."""
    recorded = getattr(bench.store, "recorded_render_mode", None)
    if recorded is None:
        return ""
    mode = recorded(skill.name, skill.domain, skill.version)
    return f"  (recorded {mode})" if mode else ""


def _report_json(report: RunReport, where: DomainChoice | None = None) -> dict[str, Any]:
    """The report as data, for a harness to read. The field names are ``PerceptionCounts``'
    own, so this and ``explain()`` can never quote different numbers for one run."""
    eyes = report.perception
    return {
        "ok": report.ok,
        "task": report.task.text,
        "domain": report.task.domain,
        "decision": report.decision,
        "domain_resolved_by": where.source if where is not None else "named",
        "rescued": report.rescued,
        "warm_missed": report.warm_missed,
        "llm_calls": report.llm_calls,
        "steps": report.steps,
        "perception": {
            "observations": eyes.observations,
            "captures": eyes.captures,
            "detections": eyes.detections,
            "ocr_reads": eyes.ocr_reads,
            "ocr_hits": eyes.ocr_hits,
            "hit_rate": round(eyes.hit_rate, 4),
        },
        "run_id": report.run_id,
        "candidates": [
            {"skill": c.skill.name, "score": round(c.score, 4), "why": c.why}
            for c in report.candidates
        ],
        "attempts": [
            {
                "path": a.path,
                "ok": a.ok,
                "stage": a.stage,
                "reason": a.reason,
                "skills_used": list(a.skills_used),
                "steps": a.steps,
                "llm_calls": a.llm_calls,
                "usd": round(a.usd, 6),
                "cross_mode": a.cross_mode,
                "demoted": a.demoted,
            }
            for a in report.attempts
        ],
        "learned": (
            None
            if report.learned is None
            else {"name": report.learned.name, "version": report.learned.version}
        ),
        "learning_note": report.learning_note,
    }


@skills_app.command("ls")
def skills_ls(
    ctx: typer.Context,
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="Only this site or app.", show_default=False),
    ] = None,
    include_demoted: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Also list skills that were retired from retrieval, with the reason.",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """List what the agent has learned to do.

    One row per skill, latest version only, with how often it has run and how well. A
    retired skill is hidden unless you pass --all; retirement is what happens when a skill
    ran on the warm path and did not work.
    """
    bench = _bench(ctx)
    found = bench.store.list(domain=domain, include_demoted=include_demoted)
    if as_json:
        _emit([_skill_json(s) for s in found], True)
        return
    if not found:
        where = f" for domain {domain!r}" if domain else ""
        typer.echo(f'no skills stored{where}. Teach it one: skillweaver learn "<task>"')
        return
    rows = [
        (
            s.name,
            s.domain,
            f"v{s.version}",
            f"{s.stats.successes}/{s.stats.runs}",
            _ms(s.stats.mean_ms),
            "RETIRED" if s.demoted_reason else "",
            s.summary,
        )
        for s in found
    ]
    _table(("skill", "domain", "ver", "ok/runs", "mean", "state", "summary"), rows)


@skills_app.command("show")
def skills_show(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The skill's name.", show_default=False)],
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            help="The site or app it belongs to. Needed only when the same name "
            "exists for more than one domain.",
            show_default=False,
        ),
    ] = None,
    version: Annotated[
        int | None,
        typer.Option(
            "--version", help="A specific version. Default: the latest.", show_default=False
        ),
    ] = None,
    code: Annotated[
        bool, typer.Option("--code", help="Print the skill's Python source and verifier.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Show one skill: what it does, where it starts, and what it has cost.

    With --code, print the Python the agent wrote for itself.
    """
    bench = _bench(ctx)
    skill = _find(bench, name, domain, version)
    if as_json:
        _emit(_skill_json(skill, code=True), True)
        return
    lines = [
        f"{skill.name}  (v{skill.version}, {skill.domain})",
        f"  {skill.summary}",
        "",
        f"  runs        {skill.stats.successes} ok of {skill.stats.runs}"
        f"  (mean {_ms(skill.stats.mean_ms)})",
        f"  starts on   {skill.precondition.value[:16] if skill.precondition else 'any screen'}"
        f"{_recorded_in(bench, skill)}",
        f"  parameters  {', '.join(skill.params) or '(none)'}",
        f"  requires    {', '.join(skill.requires) or '(nothing)'}",
        f"  verifier    {'yes' if skill.verifier_code else 'no'}",
        f"  learned     {skill.provenance.created_at:%Y-%m-%d %H:%M} by "
        f"{skill.provenance.model} from run {skill.provenance.trajectory_id}",
        f"  taught by   {skill.provenance.task_text!r}",
    ]
    if skill.demoted_reason:
        lines.append(f"  RETIRED     {skill.demoted_reason}")
    lines += ["", "  " + skill.docstring.replace("\n", "\n  ")]
    if code:
        lines += ["", "--- code ---", skill.code]
        if skill.verifier_code:
            lines += ["--- verifier ---", skill.verifier_code]
    _emit("\n".join(lines), False)


@skills_app.command("rm")
def skills_rm(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The skill's name.", show_default=False)],
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="The site or app it belongs to.", show_default=False),
    ] = None,
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why you are retiring it. Recorded with the skill."),
    ] = "removed from the command line",
) -> None:
    """Retire a skill so retrieval stops offering it.

    Nothing is deleted: every stored version stays on disk, and a later `learn` of the same
    task stores a fresh, healthy version. The next `run` of its task explores instead.
    """
    bench = _bench(ctx)
    skill = _find(bench, name, domain, None)
    bench.store.demote(skill.name, skill.domain, reason)
    typer.echo(
        f"retired {skill.name} (v{skill.version}, {skill.domain}): {reason}\n"
        f"retrieval will no longer offer it; its versions are still on disk."
    )


def _find(bench: Workbench, name: str, domain: str | None, version: int | None) -> Skill:
    """One skill by name, resolving the domain when there is only one candidate."""
    if domain is not None:
        try:
            return bench.store.get(name, domain, version)
        except SkillNotFound as exc:
            _die(str(exc), NO)
    matches = [s for s in bench.store.list(include_demoted=True) if s.name == name]
    if not matches:
        _die(f"no skill named {name!r}. Try: skillweaver skills ls", NO)
    if len(matches) > 1:
        domains = ", ".join(sorted(s.domain for s in matches))
        _die(f"{name!r} exists for several domains ({domains}); pass --domain", NO)
    return bench.store.get(name, matches[0].domain, version)


def _skill_json(skill: Skill, *, code: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": skill.name,
        "domain": skill.domain,
        "version": skill.version,
        "summary": skill.summary,
        "params": dict(skill.params),
        "requires": list(skill.requires),
        "precondition": skill.precondition.value if skill.precondition else None,
        "has_verifier": skill.verifier_code is not None,
        "demoted_reason": skill.demoted_reason,
        "stats": {
            "runs": skill.stats.runs,
            "successes": skill.stats.successes,
            "mean_ms": round(skill.stats.mean_ms, 2),
        },
        "provenance": {
            "trajectory_id": skill.provenance.trajectory_id,
            "task_text": skill.provenance.task_text,
            "model": skill.provenance.model,
        },
    }
    if code:
        out["docstring"] = skill.docstring
        out["code"] = skill.code
        out["verifier_code"] = skill.verifier_code
    return out


@graph_app.command("show")
def graph_show(
    ctx: typer.Context,
    domain: Annotated[
        str, typer.Argument(help="The site or app, e.g. 'acme.test'.", show_default=False)
    ],
    svg: Annotated[
        bool,
        typer.Option("--svg", help="Draw the graph as SVG instead of listing it."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write the SVG here instead of to standard output.",
            show_default=False,
        ),
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Show the screens the agent knows for a site, and how to get between them.

    Each node is one recognizable screen; each edge is an action sequence observed to lead
    between two, with how often it worked. The warm path routes only over edges with at
    least one success, so an edge listed at 0/3 is one the agent will not trust.

        skillweaver graph show acme.test --svg --out acme.svg
    """
    bench = _bench(ctx)
    # Only what is not already in memory: an in-memory graph with no store answers
    # `load` by FORGETTING the domain, wiping exactly what we were asked to show.
    if not bench.graph.states(domain):
        try:
            bench.graph.load(domain)
        except SkillWeaverError as exc:
            _die(f"could not load the graph for {domain!r}: {exc}")
    states = bench.graph.states(domain)
    edges = _edges_of(bench, states)

    if svg:
        drawing = _svg(domain, states, edges)
        if out is not None:
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(drawing, encoding="utf-8")
            except OSError as exc:
                _die(f"cannot write {out}: {exc}")
            typer.echo(f"wrote {out} ({len(states)} screen(s), {len(edges)} edge(s))")
        else:
            typer.echo(drawing)
        return

    if as_json:
        _emit(
            {
                "domain": domain,
                "states": [
                    {
                        "fingerprint": s.fingerprint.value,
                        "label": s.label,
                        "url_pattern": s.url_pattern,
                    }
                    for s in states
                ],
                "edges": [
                    {
                        "src": e.src.value,
                        "dst": e.dst.value,
                        "actions": [action_to_dict(a) for a in e.actions],
                        "attempts": e.attempts,
                        "successes": e.successes,
                        "mean_ms": round(e.mean_ms, 2),
                    }
                    for e in edges
                ],
            },
            True,
        )
        return

    if not states:
        typer.echo(
            f"nothing known about {domain!r} yet. The graph grows as a by-product of "
            f'running tasks: skillweaver learn "<task>" --url https://{domain}/'
        )
        return
    typer.echo(f"{domain}: {len(states)} screen(s), {len(edges)} edge(s)\n")
    _table(
        ("screen", "label", "url"),
        [(s.fingerprint.value[:12], s.label or "-", s.url_pattern or "-") for s in states],
    )
    if edges:
        typer.echo("")
        _table(
            ("from", "to", "ok/tries", "mean", "actions"),
            [
                (
                    e.src.value[:12],
                    e.dst.value[:12],
                    f"{e.successes}/{e.attempts}",
                    _ms(e.mean_ms),
                    _actions(e.actions),
                )
                for e in edges
            ],
        )


def _edges_of(bench: Workbench, states: Sequence[UIState]) -> list[Transition]:
    """Every outgoing edge of every known screen, de-duplicated. Uses ``neighbors`` from the
    ``GraphView`` Protocol, not a concrete graph's richer listing, so any graph works."""
    seen: dict[tuple[str, str, tuple[Any, ...]], Transition] = {}
    for state in states:
        for edge in bench.graph.neighbors(state.fingerprint):
            seen.setdefault((edge.src.value, edge.dst.value, edge.actions), edge)
    return sorted(seen.values(), key=lambda e: (e.src.value, -e.successes, e.dst.value))


def _actions(actions: Sequence[Any]) -> str:
    """Borrowed from the dashboard so the two views of one edge read the same way."""
    from skillweaver.dashboard.build import actions_summary

    return actions_summary(actions)


# -- drawing ---------------------------------------------------------------------------

_NODE_W, _NODE_H = 190, 54
_GAP_X, _GAP_Y = 168, 44
_MARGIN = 28
_EDGE_LABEL_CHARS = 18
"""Characters of an edge's action summary that fit between two nodes.

At font-size 10 a character is about 5.6px wide, and the label carries a ``(n/m)``
suffix on top of the summary, so this has to stay well under ``_GAP_X`` or the text
runs back over the boxes it sits between.
"""


def _svg(domain: str, states: Sequence[UIState], edges: Sequence[Transition]) -> str:
    """The graph as a standalone SVG, in breadth-first layers from the oldest known state -
    almost always where runs begin - so it reads in the direction the agent moves. No script
    and no external reference, so it survives being emailed."""
    if not states:
        return _empty_svg(domain)
    layers = _layers(states, edges)
    at: dict[str, tuple[float, float]] = {}
    for column, row_states in enumerate(layers):
        for row, value in enumerate(row_states):
            at[value] = (
                _MARGIN + column * (_NODE_W + _GAP_X),
                _MARGIN + 40 + row * (_NODE_H + _GAP_Y),
            )
    width = _MARGIN * 2 + len(layers) * (_NODE_W + _GAP_X) - _GAP_X
    height = _MARGIN * 2 + 40 + max(len(c) for c in layers) * (_NODE_H + _GAP_Y) - _GAP_Y

    labels = {s.fingerprint.value: (s.label or s.fingerprint.value[:10]) for s in states}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif, system-ui, sans-serif">',
        "<defs>"
        '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#64748b"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{_MARGIN}" y="{_MARGIN + 6}" font-size="15" font-weight="600" '
        f'fill="#0f172a">{_esc(domain)}</text>',
        f'<text x="{_MARGIN}" y="{_MARGIN + 24}" font-size="11" fill="#64748b">'
        f"{len(states)} screen(s), {len(edges)} edge(s) - a dashed edge has never "
        f"succeeded and the planner will not route over it</text>",
    ]
    for edge in edges:
        if edge.src.value not in at or edge.dst.value not in at:
            continue
        parts.append(_edge_svg(edge, at))
    for value, (x, y) in at.items():
        parts.append(
            f'<g><rect x="{x}" y="{y}" width="{_NODE_W}" height="{_NODE_H}" rx="9" '
            f'fill="#f8fafc" stroke="#cbd5e1"/>'
            f'<text x="{x + 12}" y="{y + 23}" font-size="13" fill="#0f172a">'
            f"{_esc(_fit(labels.get(value, value), 24))}</text>"
            f'<text x="{x + 12}" y="{y + 41}" font-size="10" fill="#94a3b8" '
            f'font-family="ui-monospace, monospace">{_esc(value[:14])}</text></g>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _edge_svg(edge: Transition, at: dict[str, tuple[float, float]]) -> str:
    """One edge as a curve, labelled with what it does and how often it works."""
    x1, y1 = at[edge.src.value]
    x2, y2 = at[edge.dst.value]
    sx, sy = x1 + _NODE_W, y1 + _NODE_H / 2
    ex, ey = x2, y2 + _NODE_H / 2
    if x2 <= x1:  # a backward edge leaves and re-enters from below
        sx, sy = x1 + _NODE_W / 2, y1 + _NODE_H
        ex, ey = x2 + _NODE_W / 2, y2 + _NODE_H
    mid = (sx + ex) / 2
    dashed = ' stroke-dasharray="5 4"' if edge.successes == 0 else ""
    label = f"{_fit(_actions(edge.actions), _EDGE_LABEL_CHARS)} ({edge.successes}/{edge.attempts})"
    return (
        f'<path d="M{sx:.0f},{sy:.0f} C{mid:.0f},{sy:.0f} {mid:.0f},{ey:.0f} '
        f'{ex:.0f},{ey:.0f}" fill="none" stroke="#64748b" stroke-width="1.6"'
        f'{dashed} marker-end="url(#a)"/>'
        f'<text x="{mid:.0f}" y="{(sy + ey) / 2 - 6:.0f}" font-size="10" fill="#475569" '
        f'text-anchor="middle">{_esc(label)}</text>'
    )


def _layers(states: Sequence[UIState], edges: Sequence[Transition]) -> list[list[str]]:
    """Breadth-first columns from the oldest state; anything unreached gets its own."""
    order = [s.fingerprint.value for s in states]
    ahead: dict[str, list[str]] = {value: [] for value in order}
    for edge in edges:
        if edge.src.value in ahead and edge.dst.value in ahead:
            ahead[edge.src.value].append(edge.dst.value)
    placed: set[str] = set()
    columns: list[list[str]] = []
    frontier = [order[0]]
    while frontier:
        column = [v for v in frontier if v not in placed]
        if not column:
            break
        columns.append(column)
        placed.update(column)
        frontier = [nxt for v in column for nxt in ahead[v] if nxt not in placed]
    orphans = [v for v in order if v not in placed]
    if orphans:
        columns.append(orphans)
    return columns


def _empty_svg(domain: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="90" '
        'viewBox="0 0 420 90" font-family="ui-sans-serif, system-ui, sans-serif">'
        '<rect width="420" height="90" fill="#ffffff"/>'
        f'<text x="20" y="40" font-size="14" fill="#0f172a">{_esc(domain)}</text>'
        '<text x="20" y="62" font-size="11" fill="#64748b">nothing known yet - '
        "the graph grows as tasks are run</text></svg>"
    )


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@app.command("replay")
def replay_command(
    ctx: typer.Context,
    run_id: Annotated[
        str | None,
        typer.Argument(
            help="The run to replay. Omit to list the runs on record.", show_default=False
        ),
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Show exactly what happened on one recorded run, action by action.

    Every run - warm or cold, successful or not - is recorded with the screen before and
    after each action. This prints that back: what was done, what the critic made of it,
    and where the screen went. With no run id, lists the runs on record, newest last.
    """
    bench = _bench(ctx)
    if run_id is None:
        _list_runs(bench, as_json)
        return
    try:
        trajectory = bench.trajectories.load(run_id)
    except SkillWeaverError as exc:
        _die(f"{exc}. Run `skillweaver replay` with no arguments to see what is stored.", NO)

    if as_json:
        _emit(
            {
                "run_id": trajectory.run_id,
                "task": trajectory.task,
                "domain": trajectory.domain,
                "ok": trajectory.ok,
                "note": trajectory.note,
                "steps": [
                    {
                        "index": s.index,
                        "action": action_to_dict(s.action),
                        "ok": s.result.ok,
                        "error": s.result.error,
                        "ms": round(s.result.elapsed_ms, 1),
                        "from": s.before.fingerprint.value[:12],
                        "to": s.after.fingerprint.value[:12],
                        "url": s.after.url,
                        "verdict": None if s.verdict is None else s.verdict.ok,
                        "reason": "" if s.verdict is None else s.verdict.reason,
                        "note": s.note,
                    }
                    for s in trajectory.steps
                ],
            },
            True,
        )
        raise typer.Exit(OK if trajectory.ok else NO)

    verdict = "SOLVED" if trajectory.ok else "NOT SOLVED"
    typer.echo(
        f"run {trajectory.run_id}  [{verdict}]  {trajectory.domain}\n"
        f"task: {trajectory.task}\n"
        f"{len(trajectory.steps)} step(s), "
        f"{trajectory.started_at:%Y-%m-%d %H:%M:%S} to {trajectory.finished_at:%H:%M:%S}\n"
    )
    if trajectory.steps:
        _table(
            ("#", "action", "result", "screen after", "critic"),
            [
                (
                    str(s.index),
                    _actions((s.action,)),
                    "ok" if s.result.ok else f"FAILED: {s.result.error or '?'}",
                    s.after.fingerprint.value[:12],
                    _verdict(s),
                )
                for s in trajectory.steps
            ],
        )
    if trajectory.note:
        typer.echo("\n" + trajectory.note)
    raise typer.Exit(OK if trajectory.ok else NO)


def _verdict(step: Any) -> str:
    if step.verdict is None:
        return "-"
    return ("ok: " if step.verdict.ok else "no: ") + _fit(step.verdict.reason or "", 40)


def _list_runs(bench: Workbench, as_json: bool) -> None:
    """Every recorded run, oldest first."""
    try:
        run_ids = bench.trajectories.list()
    except SkillWeaverError as exc:
        _die(f"could not read the recorded runs: {exc}")
    if as_json:
        _emit(run_ids, True)
        return
    if not run_ids:
        typer.echo("no runs recorded yet. Every learn and run writes one.")
        return
    typer.echo(f"{len(run_ids)} recorded run(s); replay one with `skillweaver replay <id>`:")
    for run_id in run_ids:
        typer.echo(f"  {run_id}")


@dashboard_app.command("build")
def dashboard_build(
    ctx: typer.Context,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write the HTML. Default: <data-dir>/dashboard.html.",
            show_default=False,
        ),
    ] = None,
    open_it: Annotated[
        bool, typer.Option("--open", help="Open the page in the default browser afterwards.")
    ] = False,
) -> None:
    """Build the one-page review of everything the agent has learned.

    Reads the skill library, the site graphs, the recorded runs and any evaluation results,
    and writes one self-contained HTML file - no server, no network, no sibling files.
    Missing data is never an error: the page explains what is not there yet.
    """
    bench = _bench(ctx)
    from skillweaver.dashboard.build import build_dashboard

    destination = out if out is not None else bench.data_dir / "dashboard.html"
    try:
        written = build_dashboard(bench.data_dir, destination)
    except SkillWeaverError as exc:
        _die(f"could not build the dashboard: {exc}")
    typer.echo(f"wrote {written} ({written.stat().st_size // 1024} KB)")
    if open_it:
        import webbrowser

        webbrowser.open(written.resolve().as_uri())


@app.command("inspect")
def inspect_command(
    ctx: typer.Context,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            help="Loopback port to serve on. 0 asks the OS for a free one. "
            "Default: the inspector's own.",
            show_default=False,
        ),
    ] = None,
    open_it: Annotated[
        bool, typer.Option("--open", help="Open the page in the default browser.")
    ] = False,
) -> None:
    """Drive a live browser session by hand, one move at a time.

    Not `dashboard build`, which writes a report of what already happened. This serves
    a page on 127.0.0.1 with a prompt bar, the browser's screen with the controls the
    agent found drawn over it, buttons to CHOOSE a move without making it, EXECUTE the
    one shown, or do both, a running log of every decision and verdict, resets for the
    run, the browser and the site, and a live view of the skill library that updates
    while a `learn` in another terminal adds to it.

    It drives the same agent `learn` does, so the global flags mean what they always
    mean and are written before the subcommand:

        skillweaver --perception dom --policy jev inspect --open
        skillweaver --chrome-profile ~/.sw-chrome --chrome-attach inspect

    The server listens on the loopback only and every request must carry a token minted
    at startup, because the browser it drives may be carrying your own logged-in profile.
    """
    bench = _bench(ctx)
    from skillweaver.inspector import DEFAULT_PORT, serve

    try:
        serve(
            bench.settings,
            port=DEFAULT_PORT if port is None else port,
            open_browser=open_it,
        )
    except OSError as exc:
        _die(f"the inspector could not listen on that port: {exc}")
    except SkillWeaverError as exc:
        _die(f"the inspector could not start: {exc}")


@eval_app.command("run")
def eval_run(
    ctx: typer.Context,
    suite: Annotated[
        Path | None,
        typer.Option(
            "--suite", help="Task suite to run. Default: the harness's own.", show_default=False
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write the report. Default: <data-dir>/eval/.",
            show_default=False,
        ),
    ] = None,
    repeat: Annotated[
        int,
        typer.Option(
            "--repeat",
            help="Warm runs per task after the cold one, for the cold-versus-warm comparison.",
        ),
    ] = 1,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Run just this task id. Repeatable. Default: every task in the suite.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run the evaluation suite and write a cold-versus-warm report.

    Measures the project's actual claim: the same tasks done once by exploration and again
    from the library, with each one's time and model calls side by side. `skillweaver
    dashboard build` renders the result.

    The suite file says what it runs against and how it is judged, including which referee
    reads ground truth. None of that is a flag here, so one suite cannot be scored two ways.

    `--only` bounds a run to named tasks - a suite that hits a real website costs real time
    and somebody else's bandwidth.
    """
    bench = _bench(ctx)
    entry = _eval_entry()
    if entry is None:
        typer.echo(
            "the evaluation harness is not built yet.\n"
            "\n"
            "This subcommand is wired and will work the moment `skillweaver.eval` "
            "grows a runnable entry point (`skillweaver.eval.harness:run` or "
            "`skillweaver.eval:run_suite`). Until then, nothing here can run.\n"
            "\n"
            "Everything else works today: try `skillweaver run`, `skillweaver skills ls` "
            "or `skillweaver dashboard build`.",
            err=True,
        )
        raise typer.Exit(CANNOT)
    try:
        entry(
            workbench=bench,
            suite=suite,
            out=out if out is not None else bench.settings.eval_dir,
            repeat=repeat,
            only=list(only) if only else None,
        )
    except SkillWeaverError as exc:
        # A missing or malformed suite is a command that could not run, not a crash.
        _die(f"the evaluation could not run: {exc}")


def _eval_entry() -> Any:
    """The evaluation harness's entry point, or ``None`` when it has not landed. Two names
    are tried; an ``ImportError`` from INSIDE a harness that does exist is not swallowed."""
    import importlib

    for module_name, attr in (
        ("skillweaver.eval.harness", "run"),
        ("skillweaver.eval", "run_suite"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        entry = getattr(module, attr, None)
        if callable(entry):
            return entry
    return None


def _bench(ctx: typer.Context) -> Workbench:
    """The workbench the root callback opened, or the one a caller injected."""
    bench = ctx.obj
    if bench is None:  # only reachable if a command is invoked without the callback
        bench = build_workbench()
        ctx.obj = bench
    return bench


def _die(message: str, code: int = CANNOT) -> Any:
    """Print one clear line to stderr and stop. Never a stack trace."""
    typer.echo(message, err=True)
    raise typer.Exit(code)


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo(payload)


def _ms(value: float) -> str:
    """Milliseconds a human can read, or a dash when nothing has been timed."""
    if value <= 0:
        return "-"
    return f"{value:.0f}ms" if value < 1000 else f"{value / 1000:.1f}s"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Aligned columns, deliberately not a rich table: this gets piped into grep and awk."""
    columns = list(zip(*([headers, *rows]), strict=True)) if rows else [(h,) for h in headers]
    widths = [max(len(str(cell)) for cell in column) for column in columns]
    typer.echo("  ".join(h.upper().ljust(w) for h, w in zip(headers, widths, strict=True)))
    for row in rows:
        typer.echo("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point, returning the exit code rather than exiting. ``python -m skillweaver.cli``
    is how it is invoked: a ``skillweaver`` console script needs a ``[project.scripts]`` in
    ``pyproject.toml``, which is shared surface."""
    try:
        app(args=list(argv) if argv is not None else None)
    except SystemExit as exc:
        # `standalone_mode=False` does NOT turn a UsageError into SystemExit the way
        # Typer's own dispatch does; unhandled, it reported success while printing a
        # failure.
        code = exc.code
        if code is None:
            return OK
        if isinstance(code, int):
            return code
        typer.echo(str(code), err=True)
        return NO
    return OK


if __name__ == "__main__":  # pragma: no cover - exercised by `python -m`
    raise SystemExit(main())
