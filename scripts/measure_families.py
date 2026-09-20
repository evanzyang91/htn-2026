#!/usr/bin/env python3
"""What action signatures does this library hold, and how far apart are they?

    uv run python scripts/measure_families.py             # the table
    uv run python scripts/measure_families.py --backfill  # ...and write them down

This is the measurement :data:`skillweaver.skills.family.MAX_FAMILY_DISTANCE` was
chosen from, kept runnable so the number can be re-derived rather than nudged. It
reads every stored skill (demoted ones included - a retired skill still did what it
did), derives its signature from its code and from the recording it was synthesized
from when that is still on disk, and prints every pairwise distance, nearest first.

``--backfill`` stores the derived signature, and the admission run as the first
precedent, on skills that were admitted BEFORE signatures existed. It is not a way
round "earned, never guessed": it writes only to a skill that carries a verifier and
has at least one recorded success, which is the same evidence the admission gate
demands today, and it says which skills it skipped and why.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from itertools import combinations

from skillweaver.contracts import Skill
from skillweaver.errors import SkillWeaverError
from skillweaver.orchestrator import build_workbench
from skillweaver.skills.family import (
    MAX_FAMILY_DISTANCE,
    derive_signature,
    distance,
    render,
    signature_from_code,
    signature_from_trajectory,
)


def _recording(bench, skill: Skill):
    try:
        return bench.trajectories.load(skill.provenance.trajectory_id, screenshots=False)
    except (SkillWeaverError, OSError, TypeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backfill", action="store_true", help="store what was derived")
    args = parser.parse_args()

    bench = build_workbench()
    skills = bench.store.list(include_demoted=True)
    derived: dict[tuple[str, str], tuple[str, ...]] = {}
    print(f"{len(skills)} stored skill(s)\n")
    for skill in skills:
        recording = _recording(bench, skill)
        signature = derive_signature(skill.code, recording)
        derived[(skill.name, skill.domain)] = signature
        print(f"{skill.name} @ {skill.domain}{'  [demoted]' if skill.demoted_reason else ''}")
        print(f"    learned from : {skill.provenance.task_text}")
        print(f"    code alone   : {render(signature_from_code(skill.code))}")
        if recording is not None:
            print(
                f"    recording    : {render(signature_from_trajectory(recording))}"
                f"   ({len(recording.steps)} action(s))"
            )
        else:
            print("    recording    : (not on disk)")
        print(f"    SIGNATURE    : {render(signature)}")
        if skill.action_signature and skill.action_signature != signature:
            print(f"    stored       : {render(skill.action_signature)}   <- differs")
        print()

    print(f"pairwise distance, nearest first (family cut: <= {MAX_FAMILY_DISTANCE})\n")
    pairs = sorted(
        (distance(derived[a], derived[b]), a, b) for a, b in combinations(sorted(derived), 2)
    )
    for d, a, b in pairs:
        mark = "FAMILY" if d <= MAX_FAMILY_DISTANCE else "      "
        print(f"  {d:0.2f}  {mark}  {a[0]}@{a[1]}  ~  {b[0]}@{b[1]}")

    if not args.backfill:
        return 0
    record = getattr(bench.store, "_rewrite_meta", None)
    if record is None:
        print("\nthis store cannot be backfilled", file=sys.stderr)
        return 1
    print("\nbackfill")
    for skill in skills:
        signature = derived[(skill.name, skill.domain)]
        if not skill.verifier_code:
            print(f"  skipped {skill.name}: no verifier, so nothing ever proved it")
            continue
        if skill.stats.successes < 1:
            print(f"  skipped {skill.name}: no recorded success")
            continue
        if not signature:
            print(f"  skipped {skill.name}: its code performs no action")
            continue
        record(replace(skill, action_signature=signature))
        print(f"  wrote   {skill.name}@{skill.domain}: {render(signature)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
