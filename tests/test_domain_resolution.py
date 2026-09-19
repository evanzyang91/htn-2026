"""Which library namespace a bare ``run`` means.

A skill is filed under a domain and a lookup happens in one, so the two commands this
project tells a new user to type only work if they agree about which::

    learn "Search Wikipedia for X and open the article" --url https://en.wikipedia.org/...
    run   "Search Wikipedia for X and open the article"

``learn`` is given a URL, so what it stores is filed under ``en.wikipedia.org``. The
repeat has no reason to pass one - the whole point is that the agent already knows
how - and resolving that silence to the literal target (``"browser"``) sent the lookup
to a namespace nothing is ever stored in. It did not fail: it missed, fell through to
exploration, and printed ``SOLVED by the cold path`` at full price, which is a success
report for the one thing this project claims not to do.

These tests hold both halves: the resolution, and the report that must never again let
a warm miss read as a plain success. Nothing here touches a browser, a model or a
network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from skillweaver.agent.planner import bind_args
from skillweaver.contracts import (
    Fingerprint,
    Provenance,
    Skill,
    TaskSpec,
    UIState,
)
from skillweaver.errors import ProviderError
from skillweaver.orchestrator import AttemptRecord, RunReport, resolve_domain
from skillweaver.skills.retrieve import SkillRetriever
from tests.fakes import InMemorySiteGraph, InMemorySkillStore

WIKI = "en.wikipedia.org"
GROCER = "grocer.test"
MAIN_PAGE = "https://en.wikipedia.org/wiki/Main_Page"
LEARNED = 'Search Wikipedia for "Ada Lovelace" and open her article.'


def _skill(
    name: str,
    domain: str,
    *,
    task_text: str,
    summary: str = "",
    params: dict[str, object] | None = None,
    precondition: Fingerprint | None = None,
) -> Skill:
    return Skill(
        name=name,
        domain=domain,
        summary=summary or f"{name.replace('_', ' ')}.",
        docstring=f"{summary or name}\n\nEnds on the screen the task asks for.",
        params=params if params is not None else {"query": {"type": "string"}},
        code="def run(ctx, **kw):\n    return None\n",
        requires=(),
        precondition=precondition,
        verifier_code=None,
        provenance=Provenance("run-1", task_text, "fake-llm", datetime(2026, 9, 19, tzinfo=UTC)),
    )


@pytest.fixture
def library() -> InMemorySkillStore:
    """One skill, learned on Wikipedia from a sentence with one argument in it."""
    store = InMemorySkillStore()
    store.put(
        _skill(
            "search_and_open_wikipedia_article",
            WIKI,
            task_text=LEARNED,
            summary="Search Wikipedia from the top search box and open the article for a term.",
        )
    )
    return store


@pytest.fixture
def graph() -> InMemorySiteGraph:
    return InMemorySiteGraph()


def _probe(text: str) -> TaskSpec:
    """The task as the binder sees it during resolution: no domain decided yet."""
    return TaskSpec(text=text, domain="", target="browser", params={})


def _knows_where(graph: InMemorySiteGraph, store: InMemorySkillStore, url: str) -> Fingerprint:
    """Give the one stored skill a precondition the graph has seen at ``url``."""
    fingerprint = Fingerprint("main-page")
    stored = store.get("search_and_open_wikipedia_article", WIKI)
    store.put(
        _skill(
            stored.name,
            stored.domain,
            task_text=stored.provenance.task_text,
            summary=stored.summary,
            precondition=fingerprint,
        )
    )
    graph.upsert_state(UIState(fingerprint=fingerprint, domain=WIKI, url_pattern=url))
    return fingerprint


# --------------------------------------------------------------------------------------
# THE DEFECT
# --------------------------------------------------------------------------------------


def test_a_task_with_no_url_is_filed_where_the_library_already_knows_it(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """THE BUG, AS ONE ASSERTION.

    A repeat that names neither a domain nor a URL used to resolve to ``"browser"``,
    a namespace nothing learned through ``--url`` is ever stored in. It now resolves
    to the domain of the skill that answers for this task.
    """
    _knows_where(graph, library, MAIN_PAGE)

    where = resolve_domain(LEARNED, retriever=SkillRetriever(library), graph=graph)

    assert where.domain == WIKI
    assert where.source == "library" and where.looked_up
    assert where.skill == "search_and_open_wikipedia_article"


def test_the_starting_page_comes_back_with_the_domain(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Resolving the namespace is only half of a bare repeat.

    The browser still has to open somewhere: a warm attempt launched at ``about:blank``
    consults exactly the right library and then fails at ``no_route``, which is the
    same miss wearing a different stage name. The site graph already records where
    each screen was seen, and the skill declares which screen it starts on.
    """
    _knows_where(graph, library, MAIN_PAGE)

    assert resolve_domain(LEARNED, retriever=SkillRetriever(library), graph=graph).start_url == (
        MAIN_PAGE
    )


def test_an_argument_the_task_changed_does_not_cost_it_the_domain(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """The same errand with a different argument resolves to the same place.

    A skill learned on *Ada Lovelace* has no text about photosynthesis, and asking it
    to search for one is exactly what it is for. The argument is what accounts for
    that word - here by binding out of the sentence, as it does in the planner.
    """
    _knows_where(graph, library, MAIN_PAGE)
    asked = 'Search Wikipedia for "photosynthesis" and open her article.'

    where = resolve_domain(asked, retriever=SkillRetriever(library), graph=graph)

    assert where.domain == WIKI and where.start_url == MAIN_PAGE


def test_a_verbatim_repeat_resolves_even_when_its_argument_cannot_be_bound(
    graph: InMemorySiteGraph,
) -> None:
    """THE CASE THAT BROKE THE FIRST FIX, MEASURED ON LIVE WIKIPEDIA.

    ``bind_args`` reads an argument out of a sentence only when it can point at the
    slot it sat in - a quoted span, or an example quoted in the parameter's schema.
    The skill live Wikipedia actually learned for this errand has neither: it was
    taught from *Search Wikipedia for computer vision and open the article*, with the
    query unquoted and a schema description of *e.g. an article title*. A verbatim
    repeat of that sentence therefore does NOT bind, and the warm path only runs it
    because the COMPOSER reads the sentence for one model call.

    Domain resolution cannot buy that model call - it runs before the browser is even
    open, and a namespace is not worth paying for - so requiring a bind here rejected
    the one skill that was learned for this exact sentence. The skill's own text
    accounts for the request instead, which is the strict direction: the words are
    all there because they are the words it was learned from.
    """
    store = InMemorySkillStore()
    store.put(
        _skill(
            "search_wikipedia_article",
            WIKI,
            task_text="Search Wikipedia for computer vision and open the article",
            summary="Search Wikipedia from the header search box and open the article.",
            params={"query": {"type": "string", "description": "What to search for."}},
        )
    )
    asked = "Search Wikipedia for computer vision and open the article"
    assert bind_args(store.get("search_wikipedia_article", WIKI), _probe(asked)) is None

    where = resolve_domain(asked, retriever=SkillRetriever(store), graph=graph)

    assert where.domain == WIKI and where.skill == "search_wikipedia_article"


# --------------------------------------------------------------------------------------
# The neighbouring trap: a ranking always has a winner
# --------------------------------------------------------------------------------------


def test_an_unrelated_task_does_not_borrow_the_only_domain_in_the_library(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Being the only thing stored is not a reason to answer for an errand.

    Retrieval RANKS, so a one-skill library ranks that skill first for every sentence
    in the world. If the top of the ranking named the domain, a grocery errand would
    be sent to Wikipedia - the same mistake ``MIN_ACCOUNTED_FOR`` was calibrated to
    stop, in a new place. A candidate only names a domain when it passes the planner's
    own admission question.
    """
    where = resolve_domain(
        "Add two litres of oat milk to my basket and check out",
        retriever=SkillRetriever(library),
        graph=graph,
    )

    assert where.domain == "browser" and where.source == "target"
    assert where.skill == ""
    assert "no stored skill accounts for this request" in where.why


def test_a_short_generic_name_that_ranks_perfectly_still_cannot_name_the_domain(
    graph: InMemorySiteGraph,
) -> None:
    """The measured failure that prompted ``MIN_ACCOUNTED_FOR``, asked here.

    ``open_order_screen`` is three stems, all of which appear in any sentence about
    the Order screen, so it ranks at the top of a composite ordering task while having
    the LOWEST account of it of any candidate. Ranking must not be the authority.
    """
    store = InMemorySkillStore()
    store.put(_skill("open_order_screen", GROCER, task_text="Open the Orders tab.", params={}))

    where = resolve_domain(
        "Order two Vegetable Rolls from Sakura Counter, then open the Orders tab and "
        "confirm the order is listed there",
        retriever=SkillRetriever(store),
        graph=graph,
    )

    assert where.domain == "browser" and where.source == "target"


def test_the_right_domain_wins_over_a_wrong_one_that_also_ranks(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """Two domains in the library, and the task decides which answers."""
    _knows_where(graph, library, MAIN_PAGE)
    library.put(
        _skill(
            "search_the_catalogue",
            GROCER,
            task_text="Search the catalogue for oat milk.",
            summary="Search the grocery catalogue and open a product.",
        )
    )

    assert resolve_domain(LEARNED, retriever=SkillRetriever(library), graph=graph).domain == WIKI


# --------------------------------------------------------------------------------------
# What the caller said always wins
# --------------------------------------------------------------------------------------


def test_an_explicit_domain_is_never_looked_up(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    where = resolve_domain(
        LEARNED, domain="chosen.test", retriever=SkillRetriever(library), graph=graph
    )

    assert where.domain == "chosen.test" and where.source == "named"


def test_the_host_of_a_url_is_never_looked_up(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    where = resolve_domain(
        LEARNED,
        url="https://acme.test/invoices?q=1",
        retriever=SkillRetriever(library),
        graph=graph,
    )

    assert where.domain == "acme.test" and where.source == "url"
    assert where.start_url == "https://acme.test/invoices?q=1"


def test_a_desktop_task_is_unchanged(library: InMemorySkillStore) -> None:
    """Only a browser task takes a domain from a URL, and only one is looked up."""
    where = resolve_domain("open finder", target="desktop", retriever=SkillRetriever(library))

    assert where.domain == "desktop" and where.source == "target"


# --------------------------------------------------------------------------------------
# Failing to look something up is not a reason to refuse to run
# --------------------------------------------------------------------------------------


def test_with_no_retriever_the_old_fallback_stands() -> None:
    assert resolve_domain("do a thing").domain == "browser"


def test_a_retrieval_failure_leaves_the_domain_unresolved_rather_than_raising(
    library: InMemorySkillStore,
) -> None:
    class Broken(SkillRetriever):
        def search(self, task: str, domain: str | None = None, k: int = 5) -> list[object]:  # type: ignore[override]
            raise ProviderError("the embedder is down")

    where = resolve_domain(LEARNED, retriever=Broken(library))

    assert where.domain == "browser" and where.source == "target"
    assert "could not be searched" in where.why


def test_a_graph_that_has_never_seen_the_start_screen_still_resolves_the_domain(
    library: InMemorySkillStore, graph: InMemorySiteGraph
) -> None:
    """The domain is still right; the warm attempt then reports what it could not
    route to, which is a real answer rather than a guessed URL."""
    where = resolve_domain(LEARNED, retriever=SkillRetriever(library), graph=graph)

    assert where.domain == WIKI and where.start_url is None


# --------------------------------------------------------------------------------------
# A warm miss must not read as a plain success
# --------------------------------------------------------------------------------------


def _report(*attempts: AttemptRecord, ok: bool, decision: str) -> RunReport:
    return RunReport(
        ok=ok,
        task=TaskSpec(text=LEARNED, domain="browser", target="browser", params={}),
        decision=decision,  # type: ignore[arg-type]
        attempts=attempts,
    )


def test_a_cold_success_after_a_warm_miss_says_so_in_its_headline() -> None:
    """``SOLVED by the cold path`` is true and useless after the library was consulted
    and missed. The headline is where that has to be said, because the headline is the
    line a measurement and a demo both quote."""
    report = _report(
        AttemptRecord(
            path="warm",
            ok=False,
            stage="empty_library",
            reason="the library holds no skill for domain 'browser'",
        ),
        AttemptRecord(path="cold", ok=True, reason="explored it", steps=5, llm_calls=5),
        ok=True,
        decision="cold",
    )

    assert report.warm_missed
    explained = report.explain()
    assert "SOLVED by the cold path AFTER A WARM MISS" in explained
    assert "the library was consulted first and missed at empty_library" in explained


def test_a_run_that_was_never_warm_is_reported_as_the_plain_cold_start_it_is() -> None:
    """``learn`` does not try the warm path, so it has no miss to report."""
    report = _report(
        AttemptRecord(path="cold", ok=True, reason="explored it", steps=5, llm_calls=5),
        ok=True,
        decision="cold",
    )

    assert not report.warm_missed
    assert "SOLVED by the cold path in 5 action(s)" in report.explain()


def test_a_warm_path_that_RAN_and_was_wrong_is_still_reported_as_a_rescue() -> None:
    """The louder cousin keeps its own wording: the library was WRONG, not merely
    absent, and flattening the two would lose the more serious of them."""
    report = _report(
        AttemptRecord(
            path="warm",
            ok=False,
            stage="skill_failed",
            reason="it ran and did not work",
            performed_nothing=False,
        ),
        AttemptRecord(path="cold", ok=True, reason="explored it", steps=5, llm_calls=5),
        ok=True,
        decision="cold",
    )

    assert report.rescued and not report.warm_missed
    assert "The library was WRONG about this task" in report.explain()
