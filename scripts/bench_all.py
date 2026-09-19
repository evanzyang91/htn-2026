"""Run every suite against its own application and report the three together.

    uv run python scripts/bench_all.py [--repeat 3] [--model-latency-ms 2500]

Starts each application, runs its suite through ``scripts/bench_live.py``, and prints
one table. Three applications rather than one because "it works" on a single layout is
not generalisation: a mail console, a storefront and a kanban tracker share no idiom -
rows against cards against columns, a toolbar against a sidebar against a per-card
menu - and nothing in the agent is told which it is looking at.

What the numbers mean is decided by ``--model-latency-ms``; see ``bench_live.py`` for
why a zero there is a LOWER bound on the speedup rather than an optimistic one.
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

SUITES = (
    ("console", "eval/tasks.yaml", "apps/sandbox-site/serve.py", 8765),
    ("shop", "eval/shop.yaml", "apps/shop-site/serve.py", 8766),
    ("board", "eval/board.yaml", "apps/board-site/serve.py", 8767),
)


def serving(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/__state", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start(server: str, port: int) -> subprocess.Popen | None:
    """Start one application, or return ``None`` if something is already serving it."""
    if serving(port):
        return None
    process = subprocess.Popen(
        [sys.executable, str(REPO / server), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if serving(port):
            return process
        time.sleep(0.2)
    process.terminate()
    raise SystemExit(f"{server} did not come up on port {port}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="warm runs per task")
    parser.add_argument("--bound-runs", type=int, default=1)
    parser.add_argument("--model-latency-ms", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=REPO / "data" / "bench-all")
    args = parser.parse_args()

    started: list[subprocess.Popen] = []
    rows: list[dict] = []
    try:
        for name, suite, server, port in SUITES:
            process = start(server, port)
            if process is not None:
                started.append(process)
            print(f"\n=== {name} · {suite} ===", flush=True)
            done = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "bench_live.py"),
                    "--suite",
                    str(REPO / suite),
                    "--data-dir",
                    str(args.out / name),
                    "--repeat",
                    str(args.repeat),
                    "--bound-runs",
                    str(args.bound_runs),
                    "--model-latency-ms",
                    str(args.model_latency_ms),
                ],
                capture_output=True,
                text=True,
            )
            if done.returncode != 0:
                print(done.stdout[-2000:])
                print(done.stderr[-2000:])
                raise SystemExit(f"the {name} suite did not finish")
            print(
                done.stdout.strip().splitlines()[-10:]
                and "\n".join(done.stdout.strip().splitlines()[-10:])
            )
            rows.append(_read(args.out / name / "eval", name))
    finally:
        for process in started:
            process.terminate()

    print("\n" + "=" * 92)
    print(f"THREE APPLICATIONS, ONE AGENT · model at {args.model_latency_ms:.0f}ms per call")
    print("=" * 92)
    header = (
        f"{'suite':<10} {'tasks':>6} {'cold ok':>8} {'warm ok':>8} {'cold ms':>10} "
        f"{'warm ms':>10} {'speedup':>8} {'cold calls':>11} {'warm calls':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['suite']:<10} {row['tasks']:>6} {row['cold']:>7.0f}% {row['warm']:>7.0f}% "
            f"{row['cold_ms']:>10,.0f} {row['warm_ms']:>10,.0f} {row['speedup']:>7.2f}x "
            f"{row['cold_calls']:>11.2f} {row['warm_calls']:>11.2f}"
        )
    print("-" * len(header))
    total_tasks = sum(r["tasks"] for r in rows)
    cold_ms = sum(r["cold_ms"] * r["tasks"] for r in rows)
    warm_ms = sum(r["warm_ms"] * r["tasks"] for r in rows)
    print(
        f"{'all':<10} {total_tasks:>6} "
        f"{_mean(rows, 'cold'):>7.0f}% {_mean(rows, 'warm'):>7.0f}% "
        f"{cold_ms / total_tasks:>10,.0f} {warm_ms / total_tasks:>10,.0f} "
        f"{cold_ms / warm_ms:>7.2f}x "
        f"{_mean(rows, 'cold_calls'):>11.2f} {_mean(rows, 'warm_calls'):>11.2f}"
    )
    if args.model_latency_ms <= 0:
        print(
            "\nA model call cost nothing here, so every speedup above is a LOWER bound: "
            "the\ntime a real one takes is paid by the cold path and almost never by the "
            "warm one."
        )
    return 0


def _read(directory: Path, name: str) -> dict:
    report = json.loads(sorted(directory.glob("*.json"))[-1].read_text(encoding="utf-8"))
    metrics = report["metrics"]
    tasks = [t["metrics"] for t in report["tasks"]]
    comparable = [t for t in tasks if t["comparable"]]
    return {
        "suite": name,
        "tasks": len(tasks),
        "cold": 100.0 * (metrics.get("cold_success_rate") or 0.0),
        "warm": 100.0 * (metrics.get("warm_success_rate") or 0.0),
        "cold_ms": sum(t["cold_ms"] or 0.0 for t in comparable) / max(len(comparable), 1),
        "warm_ms": sum(t["warm_ms"] or 0.0 for t in comparable) / max(len(comparable), 1),
        "speedup": metrics.get("pooled_speedup") or 0.0,
        "cold_calls": sum(t["cold_llm_calls"] or 0 for t in comparable) / max(len(comparable), 1),
        "warm_calls": metrics.get("warm_call_mean") or 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    return sum(row[key] * row["tasks"] for row in rows) / sum(row["tasks"] for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
