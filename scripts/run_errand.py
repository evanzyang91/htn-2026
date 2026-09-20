"""Do a multi-item errand: one learned skill replayed per item, in one browser session.

    uv run python scripts/run_errand.py \\
        'Add the "A", "B" and "C" to the cart' \\
        --url https://splitkb.com/ --reset-steps undo/splitkb-empty-cart.json \\
        --truth shopify-cart --data-dir /tmp/errand-data --private-chrome

Not a ``skillweaver`` subcommand on purpose: ``cli.py`` is another worker's file. It drives
the same ``Agent.run`` that ``cli run`` does, through ``skillweaver.agent.errand``, on the
Jev/DOM path (``--perception dom --policy jev --browser harness``, fast moves on).

``--private-chrome`` starts a headless Chrome of this script's own and points the harness
at it, so the person's logged-in browser is never driven. Without it the harness attaches
to the Chrome already open, exactly as the CLI does.

It never goes to a checkout, and it empties the cart before and after through the SAME
undo recipe the admission gate uses. A human-verification page fails the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

_T0 = time.time()

TRUTHS = ("none", "shopify-cart")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("errand", help='e.g. \'Add the "A", "B" and "C" to the cart\'')
    parser.add_argument("--url", required=True, help="the page every item starts on")
    parser.add_argument("--reset-steps", help="the undo recipe: JSON, or a file holding it")
    parser.add_argument("--truth", choices=TRUTHS, default="none", help="the site's own account")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--private-chrome", action="store_true")
    parser.add_argument("--no-learn", action="store_true", help="never offer an item to the gate")
    parser.add_argument("--name-items", action="store_true", help="allow the unquoted fallback")
    parser.add_argument("--keep-cart", action="store_true", help="do not run the undo afterwards")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    recipe = args.reset_steps
    if recipe and Path(recipe).is_file():
        recipe = Path(recipe).read_text(encoding="utf-8")

    with ExitStack() as stack:
        if args.private_chrome:
            from skillweaver.controllers.chrome_launch import ChromeProcess

            profile = tempfile.mkdtemp(prefix="errand-chrome-")
            chrome = stack.enter_context(ChromeProcess(user_data_dir=profile, headless=True))
            name = f"errand{os.getpid()}"
            os.environ.update(BU_CDP_URL=chrome.endpoint, BU_NAME=name)
            stack.callback(subprocess.run, ["pkill", "-f", name], check=False)

        from skillweaver.agent.errand import refinement_namer, run_errand, shopify_cart
        from skillweaver.config import check_settings, load_settings
        from skillweaver.orchestrator import build_workbench

        config = check_settings(
            dataclasses.replace(
                load_settings(),
                data_dir=args.data_dir,
                perception="dom",
                policy="jev",
                browser="harness",
                fast_moves=True,
            )
        )
        bench = build_workbench(config)
        print(f"[{time.time() - _T0:7.2f}] imports and workbench ready", flush=True)
        report = run_errand(
            bench,
            args.errand,
            url=args.url,
            reset_steps=recipe,
            truth=shopify_cart if args.truth == "shopify-cart" else None,
            learn=not args.no_learn,
            empty_after=not args.keep_cart,
            name_items=refinement_namer(config, args.url) if args.name_items else None,
        )
        print(f"[{time.time() - _T0:7.2f}] errand finished", flush=True)

    print(json.dumps(report.to_json(), indent=2) if args.json else report.explain())
    print(f"process wall: {time.time() - _T0:.1f}s")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
