"""A guided demonstration of the one thing this project claims.

    uv run python scripts/demo.py             # headless, about a minute
    uv run python scripts/demo.py --watch     # opens a real browser you can watch

It picks one errand on the storefront - add a product to the cart and place the order
for a named person - and does it three times:

    COLD     never seen before: explore it, then write down what worked
    WARM     the same sentence again: run what was written down
    BOUND    the same, with the errand's parameters supplied as well

then prints what each cost and shows the Python the agent wrote for itself. The
application is reset to a byte-identical state before every one of the three, so what
is being compared is the same task done three ways rather than one task and its
leftovers.

Nothing here is staged: the browser is real, the agent sees only pixels, and the
verdict at the end comes from the application's own state, which the agent cannot
reach. See ``scripts/scripted_operator.py`` for the one thing that IS simulated - the
model - and what that does and does not prove.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "apps" / "shop-site" / "serve.py"
PORT = 8766
SUITE = REPO / "eval" / "shop.yaml"
TASK = "buy_the_drill_bits"


def serving() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/__state", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("─" * max(len(title), 72))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="open a real browser window")
    parser.add_argument("--model-latency-ms", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "demo")
    args = parser.parse_args()

    rule("1. The application")
    print("Harbour Supply, a storefront that keeps all of its state on the server, so its")
    print("own /__state is exact ground truth - and the agent can never see it.")
    server = None
    if serving():
        print(f"\n  already serving on http://127.0.0.1:{PORT}")
    else:
        server = subprocess.Popen(
            [sys.executable, str(APP), "--port", str(PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            if serving():
                break
            time.sleep(0.2)
        print(f"\n  started on http://127.0.0.1:{PORT}")

    try:
        rule("2. The errand")
        suite = json.loads(_task_json())
        print(f'  "{suite["text"]}"')
        print("\n  Scored afterwards against the shop's own state: an order exists, addressed")
        print("  to the right person, with the right item, and the cart left empty.")

        rule("3. Doing it three times")
        if args.watch:
            print("  A browser window is about to open. Watch it explore the first time and")
            print("  replay the second - the difference is the whole point.\n")
            time.sleep(2.5)
        else:
            print("  Headless. Pass --watch to see the browser do it.\n")

        started = time.perf_counter()
        command = [
            sys.executable,
            str(REPO / "scripts" / "bench_live.py"),
            "--suite",
            str(SUITE),
            "--data-dir",
            str(args.data_dir),
            "--only",
            TASK,
            "--repeat",
            "1",
            "--bound-runs",
            "1",
            "--model-latency-ms",
            str(args.model_latency_ms),
        ]
        if args.watch:
            command.append("--headed")
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-3000:])
            print(done.stderr[-3000:])
            return 1
        elapsed = time.perf_counter() - started

        report = json.loads(sorted((args.data_dir / "eval").glob("*.json"))[-1].read_text())
        runs = report["tasks"][0]["runs"]
        rule("4. What each one cost")
        print(f"  {'':<8} {'verdict':>9} {'wall':>10} {'model calls':>12}  how")
        for run, how in zip(
            runs,
            (
                "explored it, then wrote down what worked",
                "found the skill, read one parameter out of the sentence",
                "found the skill, was handed the parameters",
            ),
            strict=False,
        ):
            name = {"cold": "COLD", "warm": "WARM"}[run["phase"]]
            if run["binding"] == "bound":
                name = "BOUND"
            verdict = "did it" if run["ok"] else "FAILED"
            print(
                f"  {name:<8} {verdict:>9} {run['wall_ms'] / 1000:>9.1f}s "
                f"{run['llm_calls']:>12}  {how}"
            )
        cold, warm = runs[0]["wall_ms"], runs[1]["wall_ms"]
        print(f"\n  {cold / warm:.1f}x faster the second time, at a model cost of ", end="")
        print(f"{runs[0]['llm_calls']} calls against {runs[1]['llm_calls']}.")
        if args.model_latency_ms <= 0:
            print("  A model call cost nothing here, so that is a FLOOR: the time a real one")
            print("  takes is paid by the cold run and almost never by the warm one.")
            print("  Try --model-latency-ms 2500 to see it at a real model's speed.")

        rule("5. What it wrote down")
        print("  The library is ordinary Python on disk, which is why the second run needs")
        print("  no model to do the same work:\n")
        _show_skill(args.data_dir)

        rule("6. Where to look next")
        print(f"  uv run python -m skillweaver.cli --data-dir {args.data_dir} skills ls")
        print(
            f"  uv run python -m skillweaver.cli --data-dir {args.data_dir} dashboard build --open"
        )
        print("  uv run python scripts/bench_all.py --model-latency-ms 2500")
        print(f"\n  ({elapsed:.0f}s for this demo.)")
    finally:
        if server is not None:
            server.terminate()
    return 0


def _task_json() -> str:
    """The demo's task, read out of the suite so the two cannot drift apart."""
    import yaml

    suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    task = next(t for t in suite["tasks"] if t["id"] == TASK)
    return json.dumps({"text": " ".join(str(task["text"]).split())})


def _show_skill(data_dir: Path) -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "skillweaver.cli",
            "--data-dir",
            str(data_dir),
            "skills",
            "show",
            "buy_one_product",
            "--code",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    body = done.stdout
    # Only the part it wrote for THIS errand; the helpers above it are shared scaffolding.
    marker = "def run(ctx"
    if marker in body:
        body = body[body.index(marker) :]
    for line in body.splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    raise SystemExit(main())
