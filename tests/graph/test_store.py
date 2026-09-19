"""Persistence: exact round-trips, merging instead of overwriting, and a refusal to
half-read a file this build does not understand."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from skillweaver.contracts import Navigate, Point, PressKey, Scroll, TypeText, UIState
from skillweaver.errors import SkillWeaverError
from skillweaver.graph.model import GraphPersistence, GraphSnapshot, InMemorySiteGraph
from skillweaver.graph.store import (
    SCHEMA_VERSION,
    JSONGraphStore,
    atomic_write,
    domain_filename,
    merge,
    snapshot_from_dict,
    snapshot_to_dict,
)
from tests.graph.builders import DOMAIN, click, edge, screen, state

LIST, DETAIL, EDIT = screen("list"), screen("detail"), screen("edit")


@pytest.fixture
def store(tmp_path) -> JSONGraphStore:
    return JSONGraphStore(tmp_path / "graphs")


def rich_snapshot() -> GraphSnapshot:
    """A snapshot exercising every field that has to survive a round-trip."""
    return GraphSnapshot(
        domain=DOMAIN,
        states=(
            UIState(
                fingerprint=LIST,
                domain=DOMAIN,
                label="invoice list",
                url_pattern="https://shop.test/invoices*",
                first_seen=datetime(2026, 1, 1, 12, 30, 45, 123456, tzinfo=UTC),
                thumbnail=b"\x89PNG\r\n\x1a\n binary \x00\xff",
            ),
            UIState(
                fingerprint=DETAIL,
                domain=DOMAIN,
                first_seen=datetime(2026, 2, 2, tzinfo=UTC),
            ),
        ),
        transitions=(
            edge(LIST, DETAIL, attempts=7, successes=5, mean_ms=123.75, day=3),
            edge(
                DETAIL,
                EDIT,
                actions=(
                    click(10, 20),
                    TypeText("hello, world"),
                    PressKey(("Meta", "a")),
                    Scroll(point=Point(5, 6), dx=0, dy=120),
                    Navigate("https://shop.test/edit"),
                ),
                attempts=3,
                successes=0,
                mean_ms=0.0,
                day=None,
            ),
        ),
    )


# -- the Protocol --------------------------------------------------------------------


def test_the_store_satisfies_the_persistence_protocol(store):
    assert isinstance(store, GraphPersistence)


# -- round-trips ---------------------------------------------------------------------


def test_save_then_load_round_trips_exactly(store):
    original = rich_snapshot()

    store.save(original)

    assert store.load(DOMAIN) == original


def test_a_round_trip_preserves_statistics(store):
    store.save(rich_snapshot())

    proven, unproven = store.load(DOMAIN).transitions

    assert (proven.attempts, proven.successes) == (7, 5)
    assert proven.mean_ms == pytest.approx(123.75)
    assert proven.last_verified == datetime(2026, 1, 3, tzinfo=UTC)
    assert (unproven.attempts, unproven.successes, unproven.mean_ms) == (3, 0, 0.0)
    assert unproven.last_verified is None


def test_a_round_trip_preserves_fingerprint_parts(store):
    store.save(rich_snapshot())

    loaded = store.load(DOMAIN).states[0]

    assert dict(loaded.fingerprint.parts) == dict(LIST.parts)
    assert loaded.fingerprint.similarity(LIST) == 1.0


def test_a_round_trip_preserves_a_binary_thumbnail(store):
    store.save(rich_snapshot())

    assert store.load(DOMAIN).states[0].thumbnail == b"\x89PNG\r\n\x1a\n binary \x00\xff"


def test_a_round_trip_preserves_every_action_kind(store):
    store.save(rich_snapshot())

    assert store.load(DOMAIN).transitions[1].actions == rich_snapshot().transitions[1].actions


def test_a_round_trip_through_a_whole_graph_keeps_it_routable(tmp_path):
    store = JSONGraphStore(tmp_path)
    live = InMemorySiteGraph(store=store)
    live.upsert_state(state(LIST, label="invoice list"))
    live.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    live.observe_transition(DETAIL, (click(2, 2),), EDIT, ok=True, ms=200.0)
    expected = live.route(LIST, EDIT)

    live.save()
    reloaded = InMemorySiteGraph(store=store)
    reloaded.load(DOMAIN)

    assert reloaded.route(LIST, EDIT) == expected
    assert [s.label for s in reloaded.states(DOMAIN)][0] == "invoice list"


def test_loading_a_domain_that_was_never_stored_is_empty_not_an_error(store):
    loaded = store.load("never-seen.test")

    assert loaded == GraphSnapshot(domain="never-seen.test")


def test_loading_replaces_what_is_in_memory_for_that_domain(tmp_path):
    store = JSONGraphStore(tmp_path)
    store.save(GraphSnapshot(DOMAIN, states=(state(LIST),)))
    graph = InMemorySiteGraph(store=store)
    graph.upsert_state(state(EDIT, label="stale"))

    graph.load(DOMAIN)

    assert [s.fingerprint.value for s in graph.states(DOMAIN)] == ["list"]


def test_an_empty_domain_gets_its_own_file(store):
    """``observe_transition`` assigns "" when the source domain is unknown; that
    graph still has to land somewhere."""
    store.save(GraphSnapshot("", states=(state(LIST, domain=""),)))

    assert store.load("").states[0].fingerprint == LIST


# -- file naming ---------------------------------------------------------------------


def test_an_ordinary_hostname_keeps_its_own_file_name():
    assert domain_filename("shop.test") == "shop.test.json"
    assert domain_filename("app.example.com") == "app.example.com.json"


@pytest.mark.parametrize("domain", ["", "..", "../../etc/passwd", "a/b\\c", "/", "a b"])
def test_a_domain_never_escapes_the_store_root(domain):
    name = domain_filename(domain)

    assert "/" not in name and "\\" not in name
    assert ".." not in name
    assert name.endswith(".json") and name != ".json"


def test_domains_that_sanitize_alike_still_get_separate_files():
    """Otherwise two sites' graphs merge into one file, and a merged graph looks
    exactly as valid as a real one."""
    assert domain_filename("a/b") != domain_filename("a_b")


def test_the_same_domain_always_maps_to_the_same_file():
    assert domain_filename("a/b") == domain_filename("a/b")


def test_the_store_reports_the_domains_it_holds(store):
    store.save(GraphSnapshot("b.test", states=(state(LIST, domain="b.test"),)))
    store.save(GraphSnapshot("a.test", states=(state(DETAIL, domain="a.test"),)))

    assert store.domains() == ["a.test", "b.test"]


def test_an_empty_store_reports_no_domains(store):
    assert store.domains() == []


# -- schema version ------------------------------------------------------------------


def test_the_stored_file_carries_the_schema_version(store):
    path = store.save(GraphSnapshot(DOMAIN, states=(state(LIST),)))

    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_a_newer_schema_version_raises_clearly(store):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 99, "domain": DOMAIN, "states": []}))

    with pytest.raises(SkillWeaverError) as caught:
        store.load(DOMAIN)

    message = str(caught.value)
    assert "99" in message
    assert str(SCHEMA_VERSION) in message
    assert str(path) in message


def test_a_missing_schema_version_raises_clearly(store):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"domain": DOMAIN, "states": []}))

    with pytest.raises(SkillWeaverError, match="schema_version"):
        store.load(DOMAIN)


@pytest.mark.parametrize("version", ["1", 1.0, None, True])
def test_a_non_integer_schema_version_raises_clearly(store, version):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": version, "domain": DOMAIN}))

    with pytest.raises(SkillWeaverError, match="schema_version"):
        store.load(DOMAIN)


def test_a_corrupt_file_raises_rather_than_loading_half_a_graph(store):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")

    with pytest.raises(SkillWeaverError, match="not valid JSON"):
        store.load(DOMAIN)


def test_a_json_file_that_is_not_an_object_raises(store):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]")

    with pytest.raises(SkillWeaverError, match="not a JSON object"):
        store.load(DOMAIN)


def test_a_malformed_record_names_the_file(store):
    path = store.path_for(DOMAIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "domain": DOMAIN, "states": [{"oops": 1}]})
    )

    with pytest.raises(SkillWeaverError, match="malformed"):
        store.load(DOMAIN)


def test_snapshot_from_dict_rejects_an_unknown_version_without_a_file():
    with pytest.raises(SkillWeaverError, match="schema version 7"):
        snapshot_from_dict({"schema_version": 7, "domain": DOMAIN})


def test_snapshot_to_dict_is_json_serializable():
    json.dumps(snapshot_to_dict(rich_snapshot()))


# -- merging -------------------------------------------------------------------------


def test_merging_two_graphs_sums_a_shared_edges_statistics():
    shared_left = edge(LIST, DETAIL, attempts=4, successes=3, mean_ms=100.0, day=1)
    shared_right = edge(LIST, DETAIL, attempts=6, successes=5, mean_ms=200.0, day=5)
    left = GraphSnapshot(DOMAIN, states=(state(LIST),), transitions=(shared_left,))
    right = GraphSnapshot(DOMAIN, states=(state(LIST),), transitions=(shared_right,))

    merged = merge(left, right)

    assert len(merged.transitions) == 1
    combined = merged.transitions[0]
    assert (combined.attempts, combined.successes) == (10, 8)
    assert combined.mean_ms == pytest.approx((100.0 * 3 + 200.0 * 5) / 8)
    assert combined.last_verified == datetime(2026, 1, 5, tzinfo=UTC)


def test_merging_keeps_the_states_of_both_sides():
    left = GraphSnapshot(DOMAIN, states=(state(LIST, day=1), state(DETAIL, day=2)))
    right = GraphSnapshot(DOMAIN, states=(state(DETAIL, day=2), state(EDIT, day=3)))

    merged = merge(left, right)

    assert {s.fingerprint.value for s in merged.states} == {"list", "detail", "edit"}


def test_merging_keeps_the_edges_of_both_sides():
    left = GraphSnapshot(DOMAIN, transitions=(edge(LIST, DETAIL),))
    right = GraphSnapshot(DOMAIN, transitions=(edge(DETAIL, EDIT),))

    merged = merge(left, right)

    assert {(e.src.value, e.dst.value) for e in merged.transitions} == {
        ("list", "detail"),
        ("detail", "edit"),
    }


def test_merging_a_shared_state_keeps_the_earliest_sighting_and_fills_gaps():
    left = GraphSnapshot(DOMAIN, states=(state(LIST, day=5),))
    right = GraphSnapshot(
        DOMAIN,
        states=(
            UIState(
                fingerprint=LIST,
                domain=DOMAIN,
                label="invoice list",
                url_pattern="/invoices",
                first_seen=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
    )

    merged = merge(left, right)

    assert len(merged.states) == 1
    assert merged.states[0].first_seen == datetime(2026, 1, 1, tzinfo=UTC)
    assert merged.states[0].label == "invoice list"
    assert merged.states[0].url_pattern == "/invoices"


def test_merging_treats_different_actions_as_different_edges():
    left = GraphSnapshot(DOMAIN, transitions=(edge(LIST, DETAIL, actions=(click(1, 1),)),))
    right = GraphSnapshot(DOMAIN, transitions=(edge(LIST, DETAIL, actions=(click(2, 2),)),))

    assert len(merge(left, right).transitions) == 2


def test_merging_with_an_empty_snapshot_changes_nothing():
    only = GraphSnapshot(DOMAIN, states=(state(LIST),), transitions=(edge(LIST, DETAIL),))

    assert merge(only, GraphSnapshot(DOMAIN)) == only


def test_merging_different_domains_is_refused():
    with pytest.raises(ValueError, match="different domains"):
        merge(GraphSnapshot("a.test"), GraphSnapshot("b.test"))


# -- save_merged: concurrent runs -----------------------------------------------------


def test_save_merged_keeps_an_earlier_runs_observations(store):
    """Two runs learn the same edge. Last-write-wins would lose the first run's
    four attempts; merging must keep all ten."""
    first = GraphSnapshot(
        DOMAIN,
        states=(state(LIST), state(DETAIL, day=2)),
        transitions=(edge(LIST, DETAIL, attempts=4, successes=3, mean_ms=100.0, day=1),),
    )
    second = GraphSnapshot(
        DOMAIN,
        states=(state(LIST), state(DETAIL, day=2)),
        transitions=(edge(LIST, DETAIL, attempts=6, successes=5, mean_ms=200.0, day=5),),
    )

    store.save_merged(first)
    combined = store.save_merged(second)

    assert combined.transitions[0].attempts == 10
    assert store.load(DOMAIN).transitions[0].attempts == 10


def test_save_merged_keeps_a_state_only_the_other_run_saw(store):
    store.save_merged(GraphSnapshot(DOMAIN, states=(state(LIST, day=1),)))

    store.save_merged(GraphSnapshot(DOMAIN, states=(state(EDIT, day=2),)))

    assert {s.fingerprint.value for s in store.load(DOMAIN).states} == {"list", "edit"}


def test_plain_save_overwrites_where_save_merged_would_not(store):
    store.save(GraphSnapshot(DOMAIN, states=(state(LIST),)))

    store.save(GraphSnapshot(DOMAIN, states=(state(EDIT),)))

    assert {s.fingerprint.value for s in store.load(DOMAIN).states} == {"edit"}


def test_graph_save_merges_every_loaded_domain(tmp_path):
    store = JSONGraphStore(tmp_path)
    first = InMemorySiteGraph(store=store)
    first.upsert_state(state(LIST, domain="a.test"))
    first.upsert_state(state(EDIT, domain="b.test"))
    first.save()

    second = InMemorySiteGraph(store=store)
    second.upsert_state(state(DETAIL, domain="a.test", day=2))
    second.save()

    assert store.domains() == ["a.test", "b.test"]
    assert {s.fingerprint.value for s in store.load("a.test").states} == {"list", "detail"}


def test_two_graphs_observing_the_same_edge_combine_through_the_store(tmp_path):
    """The end-to-end concurrency story: two independent runs, one file, no loss."""
    store = JSONGraphStore(tmp_path)
    actions = (click(1, 1),)
    for ms in (100.0, 300.0):
        run = InMemorySiteGraph(store=store)
        run.upsert_state(state(LIST))
        run.observe_transition(LIST, actions, DETAIL, ok=True, ms=ms)
        run.save()

    final = InMemorySiteGraph(store=store)
    final.load(DOMAIN)
    combined = final.transitions(DOMAIN)[0]

    assert (combined.attempts, combined.successes) == (2, 2)
    assert combined.mean_ms == pytest.approx(200.0)


# -- atomic writes --------------------------------------------------------------------


def test_atomic_write_creates_missing_directories(tmp_path):
    target = tmp_path / "deep" / "nested" / "graph.json"

    atomic_write(target, "hello")

    assert target.read_text() == "hello"


def test_atomic_write_leaves_no_temporary_files_behind(tmp_path):
    target = tmp_path / "graph.json"

    atomic_write(target, "one")
    atomic_write(target, "two")

    assert [p.name for p in tmp_path.iterdir()] == ["graph.json"]
    assert target.read_text() == "two"


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "graph.json"
    atomic_write(target, "original")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("skillweaver.graph.store.os.replace", boom)
    with pytest.raises(SkillWeaverError, match="cannot write"):
        atomic_write(target, "replacement")

    assert target.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["graph.json"]


def test_an_unwritable_destination_raises_a_skillweaver_error(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory")

    with pytest.raises(SkillWeaverError, match="cannot write site graph"):
        atomic_write(blocked / "graph.json", "{}")


# -- a count that means traversals ----------------------------------------------------
#
# The defect these cover shipped because every test above observes into a graph that
# was never LOADED. A load is what puts the store's own counts into memory, and a save
# that then hands the whole graph back to a store whose job is to SUM hands back the
# counts it was just given. See ``InMemorySiteGraph.unsaved``.


def test_loading_then_saving_does_not_double_a_stored_count(store):
    first = InMemorySiteGraph(store=store)
    first.upsert_state(state(LIST))
    first.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    first.save()

    second = InMemorySiteGraph(store=store)
    second.load(DOMAIN)
    second.save()

    stored = store.load(DOMAIN).transitions[0]
    assert (stored.attempts, stored.successes) == (1, 1), "one traversal, one attempt"


def test_ten_load_save_cycles_store_the_ten_traversals_performed(store):
    """The shape that gave the bug away: powers of two, not a count of anything."""
    for _ in range(10):
        run = InMemorySiteGraph(store=store)
        run.load(DOMAIN)
        run.upsert_state(state(LIST))
        run.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
        run.save()

    stored = store.load(DOMAIN).transitions[0]
    assert (stored.attempts, stored.successes) == (10, 10), "not 1023, which is 2**10 - 1"


def test_saving_twice_without_observing_anything_changes_no_count(store):
    """A run persists at several exits (see ``Agent._persist``), and must be able to."""
    run = InMemorySiteGraph(store=store)
    run.upsert_state(state(LIST))
    run.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)

    run.save()
    run.save()

    stored = store.load(DOMAIN).transitions[0]
    assert (stored.attempts, stored.successes) == (1, 1)


def test_a_save_after_a_load_still_carries_what_this_run_observed(store):
    seed = InMemorySiteGraph(store=store)
    seed.upsert_state(state(LIST))
    seed.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    seed.save()

    run = InMemorySiteGraph(store=store)
    run.load(DOMAIN)
    run.observe_transition(LIST, (click(1, 1),), DETAIL, ok=False, ms=0.0)
    run.save()

    stored = store.load(DOMAIN).transitions[0]
    assert (stored.attempts, stored.successes) == (2, 1), "the failure is recorded once"


def test_a_concurrent_runs_observations_survive_a_load_and_save(store):
    """What the summing merge exists for, and what the delta must not cost.

    While one run holds a loaded graph, another writes the same domain. The first
    run's save must add its own observation without erasing the other's.
    """
    seed = InMemorySiteGraph(store=store)
    seed.upsert_state(state(LIST))
    seed.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    seed.save()

    slow = InMemorySiteGraph(store=store)
    slow.load(DOMAIN)

    other = InMemorySiteGraph(store=store)
    other.load(DOMAIN)
    other.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    other.save()

    slow.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    slow.save()

    stored = store.load(DOMAIN).transitions[0]
    assert (stored.attempts, stored.successes) == (3, 3), "one seeded plus one each"


def test_the_stored_mean_still_averages_every_successful_traversal(store):
    """``mean_ms`` has to survive the delta, which averages only the NEW successes."""
    seed = InMemorySiteGraph(store=store)
    seed.upsert_state(state(LIST))
    seed.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    seed.save()

    run = InMemorySiteGraph(store=store)
    run.load(DOMAIN)
    run.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=400.0)
    run.save()

    stored = store.load(DOMAIN).transitions[0]
    assert stored.mean_ms == pytest.approx(250.0), "the mean of 100 and 400"


def test_an_edge_learned_after_a_load_is_stored_whole(store):
    seed = InMemorySiteGraph(store=store)
    seed.upsert_state(state(LIST))
    seed.observe_transition(LIST, (click(1, 1),), DETAIL, ok=True, ms=100.0)
    seed.save()

    run = InMemorySiteGraph(store=store)
    run.load(DOMAIN)
    run.observe_transition(DETAIL, (click(2, 2),), EDIT, ok=True, ms=50.0)
    run.save()

    fresh = {(t.src.value, t.dst.value): t for t in store.load(DOMAIN).transitions}
    assert (fresh["detail", "edit"].attempts, fresh["detail", "edit"].successes) == (1, 1)
    assert fresh["detail", "edit"].mean_ms == pytest.approx(50.0)
