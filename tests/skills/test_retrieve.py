"""``SkillRetriever`` ranking, with an embedder and - just as importantly - without one.

The no-embedder path is not a degraded mode kept alive out of politeness: it is what
runs when the demo laptop has no network, so it gets the same assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from skillweaver.contracts import Candidate, Provenance, Skill
from skillweaver.contracts import SkillRetriever as RetrieverProtocol
from skillweaver.errors import ProviderError, SkillWeaverError
from skillweaver.skills.retrieve import SkillRetriever, tokenize
from skillweaver.skills.store import FileSkillStore
from tests.fakes import FakeEmbedder, InMemorySkillStore

PROVENANCE = Provenance("run-7", "pay an invoice", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))


def make(name: str, domain: str, summary: str, docstring: str = "", **overrides: object) -> Skill:
    fields: dict[str, object] = {
        "name": name,
        "domain": domain,
        "summary": summary,
        "docstring": docstring or summary,
        "params": {},
        "code": "def run(ctx, **kw):\n    pass\n",
        "requires": (),
        "precondition": None,
        "verifier_code": None,
        "provenance": PROVENANCE,
    }
    fields.update(overrides)
    return Skill(**fields)  # type: ignore[arg-type]


SEARCH_INVOICE = make(
    "search_invoice",
    "example.com",
    "Search the invoice list for a company.",
    "Types a company name into the search box and waits for matching invoice rows.",
)
PAY_INVOICE = make(
    "pay_invoice",
    "example.com",
    "Confirm payment for the selected invoice.",
    "Clicks Confirm payment on the detail screen and waits for the receipt.",
)
EXPORT_CONTACTS = make(
    "export_contacts",
    "example.com",
    "Download the address book as a CSV file.",
    "Opens the contacts settings menu and exports every address as a CSV download.",
)


@pytest.fixture
def library() -> InMemorySkillStore:
    store = InMemorySkillStore()
    for skill in (SEARCH_INVOICE, PAY_INVOICE, EXPORT_CONTACTS):
        store.put(skill)
    return store


# -- ranking -------------------------------------------------------------------------


def test_a_planted_match_ranks_first_with_an_embedder(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    hits = SkillRetriever(library, fake_embedder).search("search the invoice list for Acme")
    assert hits[0].skill.name == "search_invoice"
    assert fake_embedder.calls >= 1  # the embedder really was consulted
    assert 0.0 < hits[0].score <= 1.0
    assert hits[0].score > hits[-1].score


def test_a_planted_match_ranks_first_with_no_embedder_at_all(
    library: InMemorySkillStore,
) -> None:
    retriever = SkillRetriever(library)  # no model, no network, no API key
    hits = retriever.search("search the invoice list for Acme")
    assert hits[0].skill.name == "search_invoice"
    assert 0.0 < hits[0].score <= 1.0
    assert "no embedder" in hits[0].why


def test_both_paths_agree_on_the_other_planted_match(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    task = "confirm the payment for this invoice"
    assert SkillRetriever(library, fake_embedder).search(task)[0].skill.name == "pay_invoice"
    assert SkillRetriever(library).search(task)[0].skill.name == "pay_invoice"


def test_results_are_ordered_best_first(library: InMemorySkillStore) -> None:
    hits = SkillRetriever(library).search("invoice", k=5)
    assert [c.score for c in hits] == sorted((c.score for c in hits), reverse=True)


def test_k_caps_the_result_and_zero_gives_nothing(library: InMemorySkillStore) -> None:
    assert len(SkillRetriever(library).search("invoice payment export", k=2)) == 2
    assert SkillRetriever(library).search("invoice", k=0) == []


def test_an_irrelevant_task_returns_an_empty_list(library: InMemorySkillStore) -> None:
    assert SkillRetriever(library).search("recalibrate the telescope mirror") == []


def test_an_empty_library_returns_an_empty_list(fake_embedder: FakeEmbedder) -> None:
    assert SkillRetriever(InMemorySkillStore(), fake_embedder).search("anything") == []


def test_word_endings_do_not_hide_a_match(library: InMemorySkillStore) -> None:
    # "invoices" must find a skill whose text says "invoice".
    assert SkillRetriever(library).search("searching invoices")[0].skill.name == "search_invoice"


# -- the domain filter -----------------------------------------------------------------


def test_the_domain_filter_excludes_a_near_identical_summary_elsewhere() -> None:
    store = InMemorySkillStore()
    store.put(SEARCH_INVOICE)
    decoy = make(
        "search_invoice",
        "other.test",
        "Search the invoice list for a company.",  # word for word the same
        "Types a company name into the search box and waits for matching invoice rows.",
    )
    store.put(decoy)

    task = "search the invoice list for Acme"
    for retriever in (SkillRetriever(store), SkillRetriever(store, FakeEmbedder())):
        both = retriever.search(task)
        assert {c.skill.domain for c in both} == {"example.com", "other.test"}
        only_here = retriever.search(task, domain="example.com")
        assert [c.skill.domain for c in only_here] == ["example.com"]
        assert [c.skill.domain for c in retriever.search(task, domain="other.test")] == [
            "other.test"
        ]
    assert SkillRetriever(store).search(task, domain="nowhere.test") == []


# -- demotion --------------------------------------------------------------------------


def test_a_demoted_skill_is_never_returned(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    task = "search the invoice list for Acme"
    assert SkillRetriever(library).search(task)[0].skill.name == "search_invoice"

    library.demote("search_invoice", "example.com", "selector moved")

    for retriever in (SkillRetriever(library), SkillRetriever(library, fake_embedder)):
        names = [c.skill.name for c in retriever.search(task)]
        assert "search_invoice" not in names


def test_demotion_through_the_real_file_store_also_hides_it(tmp_path: Path) -> None:
    store = FileSkillStore(tmp_path / "skills")
    store.put(SEARCH_INVOICE)
    store.put(PAY_INVOICE)
    retriever = SkillRetriever(store)
    assert retriever.search("search the invoice list")[0].skill.name == "search_invoice"

    store.demote("search_invoice", "example.com", "selector moved")
    assert "search_invoice" not in [c.skill.name for c in retriever.search("search the invoices")]


# -- why -------------------------------------------------------------------------------


def test_why_names_the_words_that_actually_matched(library: InMemorySkillStore) -> None:
    why = SkillRetriever(library).search("search the invoice list for Acme")[0].why
    assert "'search'" in why and "'invoice'" in why
    assert "name 'search_invoice' matches" in why
    # It says nothing about words the task never used.
    assert "'contacts'" not in why


def test_why_reports_the_cosine_when_an_embedder_ranked_it(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    hit = SkillRetriever(library, fake_embedder).search("search the invoice list")[0]
    assert hit.why.startswith("cosine 0.")
    assert "on the skill's searchable text" in hit.why


def test_why_mentions_a_skills_track_record(library: InMemorySkillStore) -> None:
    library.record_run("search_invoice", "example.com", ok=True, ms=100.0)
    library.record_run("search_invoice", "example.com", ok=False, ms=100.0)
    hit = SkillRetriever(library).search("search the invoice list")[0]
    assert "1/2 runs succeeded" in hit.why


# -- the embedder boundary -------------------------------------------------------------


class BrokenEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201 - a deliberately hostile double
        raise RuntimeError("connection reset by peer")


class RaggedEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        return [[1.0] * (index + 1) for index, _ in enumerate(texts)]


def test_an_embedder_failure_surfaces_as_a_provider_error(library: InMemorySkillStore) -> None:
    with pytest.raises(ProviderError, match="connection reset"):
        SkillRetriever(library, BrokenEmbedder()).search("invoice")  # type: ignore[arg-type]


def test_ragged_vectors_surface_as_a_provider_error(library: InMemorySkillStore) -> None:
    with pytest.raises(ProviderError, match="different lengths"):
        SkillRetriever(library, RaggedEmbedder()).search("invoice")  # type: ignore[arg-type]


def test_vectors_are_cached_so_a_stable_library_costs_one_embedding(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    retriever = SkillRetriever(library, fake_embedder)
    retriever.search("search the invoice list")
    after_first = fake_embedder.calls
    retriever.search("search the invoice list")  # identical task: nothing new to embed
    assert fake_embedder.calls == after_first


def test_a_nonsense_lexical_weight_is_refused(library: InMemorySkillStore) -> None:
    with pytest.raises(SkillWeaverError, match="lexical_weight"):
        SkillRetriever(library, lexical_weight=1.5)


# -- the contract ----------------------------------------------------------------------


def test_the_retriever_satisfies_the_protocol(library: InMemorySkillStore) -> None:
    assert isinstance(SkillRetriever(library), RetrieverProtocol)


def test_results_are_candidates_with_scores_inside_the_documented_range(
    library: InMemorySkillStore, fake_embedder: FakeEmbedder
) -> None:
    for hit in SkillRetriever(library, fake_embedder).search("invoice payment", k=5):
        assert isinstance(hit, Candidate)
        assert 0.0 < hit.score <= 1.0
        assert hit.why


def test_tokenize_drops_stopwords_and_noise() -> None:
    assert tokenize("Search THE invoice list for a company!") == [
        "search",
        "invoice",
        "list",
        "company",
    ]
    assert tokenize("") == []


# -- the sentence a skill was learned from ----------------------------------------------
#
# A summary is a model's description of a skill; the learned sentence is what a PERSON
# typed to get it. Asking for the same errand again tends to reuse the person's words.


LEARNED_FROM_WORDS = make(
    "archive_statement",
    "example.com",
    "Move a finished statement out of the active list.",
    "Selects the statement and moves it to the archive, where it no longer appears.",
    provenance=Provenance(
        "run-9",
        "Tidy away last quarter's paperwork",
        "fake-llm",
        datetime(2026, 9, 19, tzinfo=UTC),
    ),
)
"""Summary and learned sentence share no meaningful word, which is the whole point."""


def test_the_learned_sentence_is_matched_when_the_summary_says_it_differently() -> None:
    """A skill described as "move a statement to the archive" is found by the words
    the person who asked for it actually used, which the summary never repeats."""
    store = InMemorySkillStore()
    store.put(LEARNED_FROM_WORDS)
    store.put(EXPORT_CONTACTS)

    hits = SkillRetriever(store).search("Tidy away last quarter's paperwork")

    assert hits[0].skill.name == "archive_statement"
    # Every word of the task is covered; the remaining 0.4 is the name-hit share, and
    # "archive_statement" is the model's word for it, not the person's.
    assert hits[0].score == pytest.approx(0.6)


def test_a_word_for_word_repeat_is_called_out_in_why() -> None:
    """The planner can bind a repeat with no model at all, so a human reading the
    candidate over the demo's shoulder should be told that is what this is."""
    store = InMemorySkillStore()
    store.put(LEARNED_FROM_WORDS)

    hit = SkillRetriever(store).search("tidy away last quarter's paperwork.")[0]

    assert "the exact sentence this skill was learned from" in hit.why


def test_a_task_that_is_not_the_learned_sentence_is_not_called_a_repeat() -> None:
    store = InMemorySkillStore()
    store.put(LEARNED_FROM_WORDS)

    hit = SkillRetriever(store).search("tidy away this quarter's invoices")[0]

    assert "the exact sentence" not in hit.why


def test_the_learned_sentence_does_not_make_an_unrelated_task_match() -> None:
    """Widening what is searched is only safe if "nothing is relevant" still comes
    back empty: the point of the change is to reach the right skill sooner, not to
    have an answer for everything."""
    store = InMemorySkillStore()
    for skill in (LEARNED_FROM_WORDS, SEARCH_INVOICE, PAY_INVOICE, EXPORT_CONTACTS):
        store.put(skill)

    assert SkillRetriever(store).search("recalibrate the telescope mirror") == []
