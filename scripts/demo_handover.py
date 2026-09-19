"""Let a PERSON open the door, then hand the same browser to the agent.

    uv run python scripts/demo_handover.py "Add cake ingredients to the cart" \
        --url https://www.walmart.ca --data-dir data/live

Some real sites will not talk to an automated browser until a human has done
something first: passed a "Robot or human?" challenge, signed in, accepted a
consent banner. None of that is the agent's job and none of it should be worked
around - a bot check exists to be answered by a person, and the person is right
here.

So this script opens a window, goes to the page, and STOPS. You do whatever the
site wants. When you press Enter, the real agent - the same one
``skillweaver learn`` builds, through :func:`~skillweaver.orchestrator.build_agent` -
takes over that very browser, with the cookies your session just earned.

Headed on purpose: you asked to watch. Be aware that headed Chromium does not
paint what the detector was trained on, so the agent perceives slightly less here
than it would headless - see ``WATCH_PARAM`` in ``skillweaver/orchestrator.py``.

Nothing is bought. There is no reset URL for a real shop, so the admission gate
cannot put the world back and will refuse to store whatever is learned; that is
the correct answer rather than a failure, and the run still shows what the agent
can and cannot do on the page.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from skillweaver.config import load_settings  # noqa: E402
from skillweaver.contracts import Navigate  # noqa: E402
from skillweaver.controllers.browser import BrowserController  # noqa: E402
from skillweaver.graph.model import InMemorySiteGraph  # noqa: E402
from skillweaver.graph.store import JSONGraphStore  # noqa: E402
from skillweaver.llm.anthropic_ import AnthropicClient  # noqa: E402
from skillweaver.orchestrator import (  # noqa: E402
    ComposedPerceiver,
    budget_from,
    build_agent,
    navigating_environment,
    task_spec,
)
from skillweaver.perception.detect_yolo import DEFAULT_WEIGHTS_NAME, YoloDetector  # noqa: E402
from skillweaver.perception.ocr import RapidOcrReader  # noqa: E402
from skillweaver.skills.retrieve import SkillRetriever  # noqa: E402
from skillweaver.skills.store import FileSkillStore  # noqa: E402
from skillweaver.trajectory.record import Recorder  # noqa: E402
from skillweaver.trajectory.store import TrajectoryFileStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="what to do, in plain English")
    parser.add_argument("--url", required=True, help="page to open and hand over")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "live")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--max-usd", type=float, default=5.0)
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="do not even offer the run to the admission gate",
    )
    args = parser.parse_args()

    config = load_settings(env={"SKILLWEAVER_DATA_DIR": str(args.data_dir)}, env_file=None)
    spec = task_spec(args.task, domain=args.domain, url=args.url)

    store = FileSkillStore(config.skills_dir)
    graph = InMemorySiteGraph(store=JSONGraphStore(config.graphs_dir))
    trajectories = TrajectoryFileStore(config.trajectories_dir)

    controller = BrowserController(headless=False)
    perceiver = ComposedPerceiver(
        YoloDetector(config.models_dir / DEFAULT_WEIGHTS_NAME), RapidOcrReader()
    )
    try:
        controller.perform(Navigate(args.url))
        time.sleep(3.0)

        print("\n" + "=" * 70)
        print(f"  A browser is open at {args.url}.")
        print("  Do whatever the site needs from a HUMAN:")
        print("    - answer any 'Robot or human?' challenge")
        print("    - sign in, if you want the agent signed in")
        print("    - dismiss any cookie or location banner")
        print("\n  Then come back here and press Enter. The agent takes over.")
        print("=" * 70)
        input("\n  press Enter when the page is ready > ")

        print(f"\n  handing over. url is now {controller.url()}\n")
        graph.load(spec.domain)
        agent = build_agent(
            spec,
            controller=controller,
            perceiver=perceiver,
            llm=AnthropicClient(model=config.claude_model, computer_use=True),
            store=store,
            retriever=SkillRetriever(store),
            graph=graph,
            trajectories=trajectories,
            recorder=Recorder(config.trajectories_dir),
            # No reset URL for a real shop: the gate will say it could not prove
            # whatever it learned, which is the honest answer.
            environment=navigating_environment(controller, perceiver, graph=graph, restore=None),
            budget=budget_from(
                config,
                max_steps=args.max_steps,
                max_seconds=args.max_seconds,
                max_usd=args.max_usd,
                max_llm_calls=args.max_llm_calls,
            ),
        )
        report = agent.run(spec, learn=not args.no_learn, warm=True, cold=True)
        print("\n" + report.explain())
        print(f"\nended on: {controller.url()}")
        input("\n  press Enter to close the browser > ")
        return 0 if report.ok else 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
