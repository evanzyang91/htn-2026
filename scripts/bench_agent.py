"""Benchmark the real agent, cold then warm, over the fake app with REAL file stores.

The same wiring the shipped workbench uses - ``build_agent``, ``FileSkillStore``,
``JSONGraphStore``, ``TrajectoryFileStore``, the real ``Recorder`` - with only the
browser and the model replaced by the doubles in ``tests/fakes``. The model's script
holds exactly the calls one cold run and the warm bindings cost, so a warm run that
consulted the model once more than it should would fail loudly rather than skew a mean.

This measures the warm path's software overhead: retrieval, store I/O, routing,
the sandbox and the critic. Per run it prints wall milliseconds and model calls;
at the end, the pooled cold/warm speedup.

    uv run python scripts/bench_agent.py [--warm 5] [--tasks 3] [--json out.json]

``--tasks`` repeats the measurement across N distinct copies of the task (fresh
worlds), because a single cold run is one sample of a noisy quantity.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import tempfile
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# ultralytics installs a top-level `tests` package that shadows this repository's
# tests/ in a plain python process; bind ours before anything imports it.
_tests = types.ModuleType("tests")
_tests.__path__ = [str(REPO / "tests")]
sys.modules["tests"] = _tests

from skillweaver.contracts import Budget, TaskSpec, Trajectory  # noqa: E402
from skillweaver.eval.harness import Check, EvalTask, Suite, run_task  # noqa: E402
from skillweaver.graph.model import InMemorySiteGraph  # noqa: E402
from skillweaver.graph.store import JSONGraphStore  # noqa: E402
from skillweaver.orchestrator import build_agent  # noqa: E402
from skillweaver.skills.retrieve import SkillRetriever  # noqa: E402
from skillweaver.skills.store import FileSkillStore  # noqa: E402
from skillweaver.skills.synthesize import EnvironmentFactory, ReplayEnvironment  # noqa: E402
from skillweaver.trajectory.record import Recorder  # noqa: E402
from skillweaver.trajectory.store import TrajectoryFileStore  # noqa: E402
from tests.fakes import FakeLLM, Scenario, make_scenario  # noqa: E402
from tests.fakes.scenario import DOMAIN, GOAL  # noqa: E402

TASK = "Confirm payment of the Acme Corp invoice."


def _answer(**fields) -> str:
    fields.setdefault("thought", "the next move")
    fields.setdefault("expect", "the screen advances to the next step")
    fields.setdefault("done", False)
    return json.dumps(fields)


def click(element_id: str, **extra) -> str:
    return _answer(action={"kind": "click", "element_id": element_id}, **extra)


def type_text(text: str, **extra) -> str:
    return _answer(action={"kind": "type_text", "text": text}, **extra)


YES = json.dumps(
    {
        "ok": True,
        "evidence": "the screen advanced as expected",
        "reason": "the move did what it said",
        "confidence": 0.9,
    }
)

SKILL_CODE = (
    "def run(ctx, company):\n"
    '    ctx.ctl.type_text("acme")\n'
    '    rows = ctx.see.find_text("Acme Corp", fuzzy=False)\n'
    '    ctx.expect(bool(rows), "the search returned no row for " + company)\n'
    "    ctx.ctl.click(rows[0])\n"
    '    buttons = ctx.see.find_text("Confirm payment", fuzzy=False)\n'
    '    ctx.expect(bool(buttons), "no Confirm payment button on the invoice")\n'
    "    ctx.ctl.click(buttons[0])\n"
    "    return True\n"
)


def skill_reply() -> str:
    draft = {
        "name": "confirm_invoice_payment",
        "summary": "Confirm payment of a company's invoice from the invoice list.",
        "docstring": (
            "Searches the invoice list for the company and confirms payment.\n\n"
            "Assumes: the invoice list is on screen with its search field focused.\n"
            "Ends on: the payment-confirmed page."
        ),
        "params": {"company": {"type": "string", "description": "Company to pay."}},
        "example_args": {"company": "Acme Corp"},
        "requires": [],
        "code": SKILL_CODE,
        "verifier_code": (
            'def verify(ctx, result):\n    return bool(ctx.see.find_text("Payment confirmed"))\n'
        ),
    }
    return "```json\n" + json.dumps(draft) + "\n```"


BIND = json.dumps(
    {
        "steps": [{"skill": "confirm_invoice_payment", "args": {"company": "Acme Corp"}}],
        "why": "the library already knows this errand; the company comes from the sentence",
    }
)

LEARN_SCRIPT: tuple[str, ...] = (
    type_text("acme"),
    YES,
    click("row-1042"),
    YES,
    click("confirm", done=True),
    YES,
    YES,
    skill_reply(),
)


class FileWorld:
    """One installation with the REAL file-backed memories, in a temp directory."""

    def __init__(self, scenario: Scenario, replies, data_dir: Path) -> None:
        self.scenario = scenario
        self.llm = FakeLLM(list(replies))
        self.data_dir = data_dir
        self.store = FileSkillStore(data_dir / "skills")
        self.graph = InMemorySiteGraph(store=JSONGraphStore(data_dir / "graphs"))
        self.trajectories = TrajectoryFileStore(data_dir / "trajectories")
        self.recorder = Recorder(data_dir / "trajectories")

    def _environment(self, trajectory: Trajectory) -> EnvironmentFactory:
        def factory() -> ReplayEnvironment:
            self.scenario.controller.reset()
            return ReplayEnvironment(
                self.scenario.controller, self.scenario.perceiver, self.graph, restored=True
            )

        return factory

    @contextlib.contextmanager
    def session(self, task: TaskSpec, budget: Budget):
        yield build_agent(
            task,
            controller=self.scenario.controller,
            perceiver=self.scenario.perceiver,
            llm=self.llm,
            store=self.store,
            retriever=SkillRetriever(self.store),
            graph=self.graph,
            trajectories=self.trajectories,
            recorder=self.recorder,
            environment=self._environment,
            budget=budget,
        )


class ControllerReferee:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def reset(self) -> None:
        self._scenario.controller.reset()

    def state(self) -> dict:
        return {"app": {"screen": self._scenario.controller.state}}


EVAL_TASK = EvalTask(
    id="confirm_invoice_payment",
    text=TASK,
    tags=("multi-step",),
    expect=(Check("app.screen", "equals", GOAL),),
    params={"company": "Acme Corp"},
)


def a_suite() -> Suite:
    return Suite(
        tasks=(EVAL_TASK,),
        name="fake-app-file-stores",
        domain=DOMAIN,
        base_url="",
        source="scripts/bench_agent.py",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=int, default=5, help="warm runs per task")
    parser.add_argument("--tasks", type=int, default=3, help="independent worlds to measure")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    samples = []
    for index in range(args.tasks):
        with tempfile.TemporaryDirectory(prefix="skillweaver-bench-") as tmp:
            world = FileWorld(
                make_scenario(),
                (*LEARN_SCRIPT, *([BIND] * args.warm)),
                Path(tmp),
            )
            started = time.perf_counter()
            record = run_task(
                EVAL_TASK,
                workbench=world,
                referee=ControllerReferee(world.scenario),
                suite=a_suite(),
                warm_runs=args.warm,
                bound_runs=1,
                meter=world.llm.total_usage,
            )
            elapsed = time.perf_counter() - started
            metrics = record.metrics
            bound = record.bound_metrics
            samples.append(
                {
                    "world": index,
                    "cold_ms": metrics.cold_ms,
                    "warm_ms": metrics.warm_ms,
                    "speedup": metrics.speedup,
                    "bound_warm_ms": None if bound is None else bound.warm_ms,
                    "warm_ok_rate": metrics.warm_ok_rate,
                    "cold_ok": metrics.cold_ok,
                    "warm_calls": metrics.warm_llm_calls,
                    "cold_calls": metrics.cold_llm_calls,
                    "total_s": round(elapsed, 3),
                    "runs": [
                        {
                            "phase": r.phase,
                            "binding": r.binding,
                            "ok": r.ok,
                            "wall_ms": round(r.wall_ms, 1),
                            "llm_calls": r.llm_calls,
                            "skill_used": r.skill_used,
                        }
                        for r in record.runs
                    ],
                }
            )

    print(
        f"{'world':>5} {'cold ms':>9} {'warm ms':>9} {'bound ms':>9} {'speedup':>8} "
        f"{'cold ok':>7} {'warm ok':>7}"
    )
    for s in samples:
        print(
            f"{s['world']:>5} {s['cold_ms'] or 0:>9.1f} {s['warm_ms'] or 0:>9.1f} "
            f"{s['bound_warm_ms'] or 0:>9.1f} {(s['speedup'] or 0):>7.2f}x "
            f"{str(s['cold_ok']):>7} "
            f"{s['warm_ok_rate'] if s['warm_ok_rate'] is not None else '-':>7}"
        )

    ok = [s for s in samples if s["speedup"] is not None]
    summary = {
        "worlds": len(samples),
        "warm_runs_each": args.warm,
        "mean_cold_ms": round(statistics.fmean(s["cold_ms"] for s in ok), 1) if ok else None,
        "mean_warm_ms": round(statistics.fmean(s["warm_ms"] for s in ok), 1) if ok else None,
        "mean_bound_ms": round(
            statistics.fmean(s["bound_warm_ms"] for s in ok if s["bound_warm_ms"] is not None), 1
        )
        if ok
        else None,
        "pooled_speedup": round(sum(s["cold_ms"] for s in ok) / sum(s["warm_ms"] for s in ok), 2)
        if ok
        else None,
    }
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.json is not None:
        args.json.write_text(
            json.dumps({"samples": samples, "summary": summary}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
