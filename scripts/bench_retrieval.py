#!/usr/bin/env python3
"""Does ranking by meaning find a stored skill that word overlap misses?

    uv run python scripts/bench_retrieval.py            # both ways, side by side
    uv run python scripts/bench_retrieval.py --verbose  # every query's top candidate

The question this answers is the one the feature exists for: *the agent already knows
how to do something and is asked for it in DIFFERENT WORDS - does it find what it has?*
It is answered offline, against libraries that are already in this repository, so it
costs nothing and can be re-run on any machine that has fetched the weights
(``scripts/fetch_embedder.py``).

What is measured
----------------

Three numbers, because two of them are easy to buy with the third:

``recall``     the top-ranked skill is one that can do the errand.
``runnable``   ...AND the planner would actually RUN it: its arguments bind
               (:func:`~skillweaver.agent.planner.bind_args`) and its text plus those
               arguments account for enough of the request
               (:data:`~skillweaver.agent.planner.MIN_ACCOUNTED_FOR` over
               :func:`~skillweaver.skills.retrieve.accounted_for`). Retrieval RANKS and
               the planner DECIDES, so a ranking that improves without moving this
               number has not made a single run faster. This is the number to read.
``false hit``  a query nothing in the library can do came back with a candidate
               anyway. An embedder that matches everything is WORSE than none: a
               wrong skill that runs is more expensive than a miss that explores,
               because it leaves the world somewhere the task did not ask for.

The query set
-------------

Paraphrases written by hand BEFORE any of this was scored, one set per stored skill,
avoiding that skill's own distinctive nouns - which is precisely the case word overlap
cannot answer and the case the captain's brief is about. The learned sentence itself is
included as a control: it must stay a hit, since a verbatim repeat is the cheapest warm
run there is.

The libraries are the two real ones this repository carries:

``wikipedia``  ``data/skills``, written by ``claude-opus-5`` during the live Wikipedia
               run of 2026-09-19. Three usable skills; the other two are demoted and
               retrieval never offers them, which is left exactly as it is.
``console``    ``tests/dashboard_fixtures/full/skills``. Hand-written fixtures in the
               shape the synthesizer produces, not the output of a run - said plainly
               because it matters when reading the numbers. It is included because it
               is the only library here with near-neighbours in it
               (``open_records`` / ``open_record_detail``), which is the shape that
               breaks retrieval in a library that has grown.

NOT measured here: the ordering suite of ``eval/order.yaml``. Its library was never
committed - ``data/skills`` holds only the Wikipedia run - and rebuilding it means
twelve cold runs against a model, which needs an API key this worktree does not have.
Inventing those skills' text and then scoring against it would be measuring the
invention.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skillweaver.agent.planner import (  # noqa: E402
    MIN_ACCOUNTED_FOR,
    account_of,
    asks_for,
    bind_args,
)
from skillweaver.config import load_settings  # noqa: E402
from skillweaver.contracts import Skill, SkillStore, TaskSpec  # noqa: E402
from skillweaver.skills.embed import embedder_for  # noqa: E402
from skillweaver.skills.retrieve import SkillRetriever  # noqa: E402
from skillweaver.skills.store import FileSkillStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Query:
    """One request, and every stored skill that would be a correct answer to it.

    ``answers`` is a set because two of the Wikipedia skills really are the same
    errand learned twice (``search_wikipedia_article`` and
    ``search_and_open_wikipedia_article``); counting either as wrong would be scoring
    the library's duplication, not the ranking. Empty ``answers`` means nothing in the
    library can do this and the right result is no candidate at all.

    ``params`` are the values a SUITE would supply with this request
    (``eval/order.yaml`` carries ``{cuisine: Japanese}``); a person typing the same
    sentence supplies none. Both shapes are measured, because they fail in different
    places: with no values a skill with a required parameter cannot even be bound, and
    with them binding is free and the content gate decides.
    """

    text: str
    answers: frozenset[str]
    params: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_distractor(self) -> bool:
        return not self.answers


def q(text: str, *answers: str, **params: object) -> Query:
    return Query(text, frozenset(answers), params)


SEARCH = ("search_wikipedia_article", "search_and_open_wikipedia_article")

WIKIPEDIA_QUERIES = [
    # -- the learned sentences, as controls ------------------------------------------
    q('Search Wikipedia for "Ada Lovelace" and open her article.', *SEARCH, query="Ada Lovelace"),
    q(
        "Search Wikipedia for computer vision and open the article",
        *SEARCH,
        query="Computer vision",
    ),
    q(
        "From this article, open the linked article about Charles Babbage.",
        "open_linked_article",
        link_title="Charles Babbage",
    ),
    # -- the same errands, in other words --------------------------------------------
    q("Look up Grace Hopper on Wikipedia and bring up her page.", *SEARCH, query="Grace Hopper"),
    q("Find the encyclopedia entry for photosynthesis.", *SEARCH, query="Photosynthesis"),
    q(
        "Use the site's search box to pull up the page on the Apollo programme.",
        *SEARCH,
        query="Apollo program",
    ),
    q(
        "Follow the hyperlink in this text to the page about Alan Turing.",
        "open_linked_article",
        link_title="Alan Turing",
    ),
    q(
        "Click through to the entry that is linked here for Bletchley Park.",
        "open_linked_article",
        link_title="Bletchley Park",
    ),
    q(
        "From here, visit the referenced page on the analytical engine.",
        "open_linked_article",
        link_title="Analytical Engine",
    ),
    # -- nothing in this library can do these ----------------------------------------
    q("Recalibrate the telescope mirror."),
    q("Book a table for two at eight o'clock."),
    q("Reset the router and reconnect the printer."),
]

CONSOLE_QUERIES = [
    # -- controls ---------------------------------------------------------------------
    q("Export the current records view as CSV.", "export_csv", scope="page"),
    q("Open the records list from anywhere in the console.", "open_records"),
    q("Search the records table for a company.", "search_records", company="Acme Corp"),
    q("Open one record from a filtered list.", "open_record_detail", row=1),
    q("Reply to the currently open message.", "send_reply", body="Thanks, will do."),
    # -- the same errands, in other words ---------------------------------------------
    q("Download the visible rows as a spreadsheet file.", "export_csv", scope="page"),
    q("Save this table to a comma separated file.", "export_csv", scope="page"),
    q("Bring up the list of records.", "open_records"),
    q("Navigate to the records table.", "open_records"),
    q("Filter the table down to Initech.", "search_records", company="Initech"),
    q("Find the rows belonging to Acme Corp.", "search_records", company="Acme Corp"),
    q("Drill into a single entry from the filtered results.", "open_record_detail", row=1),
    q("Show the detail pane for one of the rows.", "open_record_detail", row=1),
    q("Write a response to this message and send it.", "send_reply", body="Thanks, will do."),
    q("Answer the open thread.", "send_reply", body="Thanks, will do."),
    # -- nothing in this library can do these -----------------------------------------
    q("Recalibrate the telescope mirror."),
    q("Book a table for two at eight o'clock."),
    q("Change the office wifi password."),
]


@dataclass
class Score:
    queries: int = 0
    recalled: int = 0
    runnable: int = 0
    distractors: int = 0
    false_hits: int = 0
    false_runs: int = 0
    stopped_by_binding: int = 0
    stopped_by_gate: int = 0

    def line(self) -> str:
        real = self.queries - self.distractors
        return (
            f"recall {self.recalled}/{real}  runnable {self.runnable}/{real}  "
            f"false hits {self.false_hits}/{self.distractors}  "
            f"false runs {self.false_runs}/{self.distractors}  "
            f"[stopped: {self.stopped_by_binding} unbindable, {self.stopped_by_gate} gate]"
        )


def would_run(
    text: str,
    domain: str,
    skill: Skill,
    params: Mapping[str, object],
    library: Callable[[], Sequence[Skill]],
) -> tuple[bool, str]:
    """``(would the planner run this, what stopped it)``, asked exactly this way.

    The planner's own two checks in its own order: bind the arguments, then ask
    whether the skill plus those arguments account for enough of the request. The
    The second return value is what makes the result actionable: a candidate that is
    now ranked first and still does not run was stopped by ONE of these, and which one
    it is decides whether better ranking could ever have helped.
    """
    task = TaskSpec(text=text, domain=domain, params=dict(params))
    args = bind_args(skill, task)
    if args is None:
        return False, "bind"
    # The planner's own account, family included: a candidate whose own words fall
    # short may be vouched for by relatives that EARNED a signature. Neither library
    # measured here was admitted after signatures existed, so unless one has been
    # backfilled (scripts/measure_families.py --backfill) this is the skill alone.
    if not asks_for(task, skill, args):
        return False, "intent"
    covered, _ = account_of(task, skill, args, library)
    return (True, "") if covered >= MIN_ACCOUNTED_FOR else (False, f"gate {covered:.2f}")


def run(
    store: SkillStore,
    domain: str,
    queries: list[Query],
    retriever,
    verbose: bool,
    *,
    with_params: bool,
) -> Score:
    score = Score()
    for query in queries:
        score.queries += 1
        hits = retriever.search(query.text, domain=domain, k=5)
        top = hits[0] if hits else None
        params = query.params if with_params else {}
        runs, stopped_by = (
            would_run(query.text, domain, top.skill, params, store.list)
            if top
            else (False, "no hit")
        )
        if stopped_by.startswith("bind"):
            score.stopped_by_binding += 1
        elif stopped_by.startswith("gate"):
            score.stopped_by_gate += 1
        if query.is_distractor:
            score.distractors += 1
            score.false_hits += int(top is not None)
            score.false_runs += int(runs)
        else:
            right = top is not None and top.skill.name in query.answers
            score.recalled += int(right)
            score.runnable += int(right and runs)
        if verbose:
            name = top.skill.name if top else "-"
            mark = "  " if query.is_distractor else ("ok" if right else "XX")
            print(
                f"  {mark} {query.text[:58]:<58} -> {name:<34}"
                f" {top.score if top else 0.0:.3f}"
                f" {'RUNS' if runs else stopped_by:<9}"
            )
    return score


def library(path: Path) -> FileSkillStore:
    return FileSkillStore(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # bind_args logs every argument it reads out of a task's wording, which is one line
    # per query here and drowns the table this script exists to print.
    logging.getLogger("skillweaver").setLevel(logging.WARNING)

    # SKILLWEAVER_EMBEDDER decides what the SHIPPED commands do; this script exists to
    # measure both rankings whatever that is set to, so it asks for one regardless.
    # `load_settings` ignores os.environ entirely once it is handed a mapping, so the
    # environment is merged in by hand here - otherwise a SKILLWEAVER_EMBEDDER_DIR
    # naming a shared download would be silently dropped and the bench would report
    # weights it can see as missing.
    embedder, reason = embedder_for(
        load_settings({**os.environ, "SKILLWEAVER_EMBEDDER": "true"}, env_file=".env")
    )
    if embedder is None:
        print(f"no embedder: {reason}")
        print("this bench compares the two rankings, so it needs the weights. stopping.")
        return 1

    suites = [
        ("wikipedia", library(ROOT / "data" / "skills"), "en.wikipedia.org", WIKIPEDIA_QUERIES),
        (
            "console",
            library(ROOT / "tests" / "dashboard_fixtures" / "full" / "skills"),
            "sandbox.test",
            CONSOLE_QUERIES,
        ),
    ]
    for shape, with_params in (("as a person types it", False), ("as a suite asks it", True)):
        print(f"\n#### {shape}")
        totals = {"keywords": Score(), "embedder": Score()}
        for name, store, domain, queries in suites:
            print(f"\n== {name}: {len(store.list(domain=domain))} skills, {len(queries)} queries")
            for label, retriever in (
                ("keywords", SkillRetriever(store)),
                ("embedder", SkillRetriever(store, embedder)),
            ):
                if args.verbose:
                    print(f"-- {label}")
                score = run(
                    store, domain, queries, retriever, args.verbose, with_params=with_params
                )
                print(f"  {label:<9} {score.line()}")
                total = totals[label]
                for name_ in vars(score):
                    setattr(total, name_, getattr(total, name_) + getattr(score, name_))
        print(f"\n== both libraries, {shape}")
        for label, total in totals.items():
            print(f"  {label:<9} {total.line()}")
    print(
        "\nread `runnable`: retrieval ranks, the planner decides, and a ranking that\n"
        "improves without moving that number has not made one run faster."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
