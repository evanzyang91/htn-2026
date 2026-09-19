"""Run the evaluation suite against the live sandbox, end to end, with no model.

Everything is real except the model: a real Chromium through the project's own
controller, the real YOLO detector and OCR reader, the real skill sandbox, the real
site graph and trajectory store, the real admission gate, and the harness's own
referee reading the application's ground truth. The one simulated component is the
computer-use model, replaced by :class:`~scripted_operator.ScriptedOperator`, which
sees exactly the prompt a model would and nothing else.

    python3 apps/sandbox-site/serve.py --port 8765 &
    uv run python scripts/bench_live.py --repeat 3

What the number means, stated before it is printed
--------------------------------------------------

A real cold run additionally waits on a model - seconds per move, several moves per
task - while a warm run consults no model at all. Simulating the model removes time
from the COLD side only, so every speedup here is a **lower bound** on the speedup
with a real model. The report says so in its own header, so the figure cannot be
quoted without it.

What it does measure honestly is everything else: whether perception finds the
controls, whether a synthesized skill really replays from a reset world, whether the
warm path retrieves and binds the right skill, how long a screen takes to read, and -
through the referee, which the agent never sees - whether the task was actually done.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# ultralytics installs a top-level `tests` package that shadows this repository's
# tests/ in a plain python process; bind ours before anything imports it.
_tests = types.ModuleType("tests")
_tests.__path__ = [str(REPO / "tests")]
sys.modules["tests"] = _tests

from playbooks_sandbox import PLAYBOOKS  # noqa: E402
from scripted_operator import ScriptedOperator  # noqa: E402

from skillweaver.contracts import Budget, TaskSpec  # noqa: E402
from skillweaver.eval.harness import (  # noqa: E402
    HttpReferee,
    load_suite,
    run_suite,
)
from skillweaver.graph.model import InMemorySiteGraph  # noqa: E402
from skillweaver.graph.store import JSONGraphStore  # noqa: E402
from skillweaver.orchestrator import (  # noqa: E402
    ComposedPerceiver,
    build_agent,
    navigating_environment,
    world_reset_from_url,
)
from skillweaver.skills.retrieve import SkillRetriever  # noqa: E402
from skillweaver.skills.store import FileSkillStore  # noqa: E402
from skillweaver.trajectory.record import Recorder  # noqa: E402
from skillweaver.trajectory.store import TrajectoryFileStore  # noqa: E402


class LiveWorkbench:
    """A workbench over the real browser, with the operator standing in for the model.

    Mirrors :func:`skillweaver.orchestrator.build_workbench` - the same stores, the
    same ``build_agent``, the same navigating environment for the admission gate - so
    what is measured is the shipped wiring rather than a benchmark-shaped copy of it.
    """

    def __init__(self, data_dir: Path, operator: ScriptedOperator, *, headless: bool) -> None:
        self.data_dir = data_dir
        self.operator = operator
        self._headless = headless
        self.store = FileSkillStore(data_dir / "skills")
        self.graph = InMemorySiteGraph(store=JSONGraphStore(data_dir / "graphs"))
        self.trajectories = TrajectoryFileStore(data_dir / "trajectories")
        self.observations = 0

    @contextlib.contextmanager
    def session(self, task: TaskSpec, budget: Budget) -> Any:
        from skillweaver.controllers.browser import BrowserController
        from skillweaver.perception.detect_yolo import YoloDetector
        from skillweaver.perception.ocr import RapidOcrReader

        controller = BrowserController(
            headless=self._headless, start_url=task.params.get("start_url")
        )
        perceiver = _Counting(
            ComposedPerceiver(
                YoloDetector(REPO / "data" / "models" / "ui_detector.pt"), RapidOcrReader()
            ),
            self,
        )
        try:
            self.graph.load(task.domain)
            reset = task.params.get("reset_url")
            yield build_agent(
                task,
                controller=controller,
                perceiver=perceiver,
                llm=self.operator,
                store=self.store,
                retriever=SkillRetriever(self.store),
                graph=self.graph,
                trajectories=self.trajectories,
                recorder=Recorder(self.data_dir / "trajectories"),
                environment=navigating_environment(
                    controller,
                    perceiver,
                    graph=self.graph,
                    restore=world_reset_from_url(str(reset)) if reset else None,
                ),
                budget=budget,
            )
        finally:
            controller.close()


class _Counting:
    """A perceiver that counts and times what it is asked for.

    Observations are the warm path's whole remaining cost, so knowing how many a run
    made - and what each one took - is the difference between "it got faster" and
    knowing which part did.
    """

    def __init__(self, inner: Any, bench: LiveWorkbench) -> None:
        self._inner = inner
        self._bench = bench
        self.ms = 0.0

    def observe(self, controller: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._inner.observe(controller)
        finally:
            self.ms += (time.perf_counter() - started) * 1000.0
            self._bench.observations += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=REPO / "eval" / "tasks.yaml")
    parser.add_argument("--out", type=Path, default=None, help="where the report goes")
    parser.add_argument("--repeat", type=int, default=3, help="warm runs per task")
    parser.add_argument("--bound-runs", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None, help="task ids to run")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--model-latency-ms",
        type=float,
        default=0.0,
        help=(
            "pretend a model call takes this long. 0 measures the system with the "
            "model removed; a real computer-use model answers in seconds, and that "
            "time is paid per call - which is the cost the warm path avoids."
        ),
    )
    parser.add_argument(
        "--dump-prompt", type=Path, default=None, help="write the first prompt here"
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    referee = HttpReferee(suite.base_url, reset_path=suite.reset_path)
    try:
        referee.reset()
    except Exception as exc:  # noqa: BLE001 - a missing sandbox is a usage error
        print(f"the sandbox at {suite.base_url} is not answering: {exc}")
        print("start it with: python3 apps/sandbox-site/serve.py --port 8765")
        return 2

    covered = {task.id for task in suite.tasks if task.text in PLAYBOOKS}
    missing = sorted({task.id for task in suite.tasks} - covered)
    if missing:
        print(f"no playbook for: {', '.join(missing)} - those tasks will fail to explore")

    data_dir = args.data_dir or (REPO / "data" / "bench-live")
    operator = ScriptedOperator(playbooks=dict(PLAYBOOKS), latency_ms=args.model_latency_ms)
    operator.debug_to = args.dump_prompt
    bench = LiveWorkbench(data_dir, operator, headless=not args.headed)

    started = time.perf_counter()
    path = run_suite(
        workbench=bench,
        referee=referee,
        suite=suite,
        out_dir=args.out or (data_dir / "eval"),
        warm_runs=args.repeat,
        bound_runs=args.bound_runs,
        meter=operator.total_usage,
        only=args.only,
    )
    elapsed = time.perf_counter() - started

    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    bound = report.get("metrics_bound") or {}
    print()
    print(f"wrote {path}")
    print(f"  suite wall clock      {elapsed:8.1f} s over {bench.observations} observation(s)")
    print(f"  model latency assumed {args.model_latency_ms:8.0f} ms per call")
    print(f"  pooled speedup        {_fmt(metrics.get('pooled_speedup'))}x")
    if args.model_latency_ms <= 0:
        print("                                  (a LOWER bound: a model call cost nothing")
        print("                                   here, which only makes COLD cheaper)")
    print(f"  cold success rate     {_pct(metrics.get('cold_success_rate'))}")
    print(f"  warm success rate     {_pct(metrics.get('warm_success_rate'))}")
    print(
        f"  warm calls per run    {_fmt(metrics.get('warm_call_mean'))} in plain English, "
        f"{_fmt(bound.get('warm_call_mean'))} with parameters supplied"
    )
    print(f"  library was used      {metrics.get('library_was_used')}")
    if metrics.get("cold_failures"):
        print(f"  cold failures         {', '.join(metrics['cold_failures'])}")
    if metrics.get("warm_failures"):
        print(f"  warm failures         {', '.join(metrics['warm_failures'])}")
    print(f"  operator calls        {json.dumps(operator.calls)}")
    if operator.unmatched:
        print(f"  tasks with no playbook {sorted(set(operator.unmatched))}")
    return 0


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.0f}%"


if __name__ == "__main__":
    raise SystemExit(main())
