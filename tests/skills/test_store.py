"""``FileSkillStore`` on a real (temporary) filesystem.

Nothing here is mocked: every assertion is about bytes that actually landed on disk,
because "the library survived a restart" is the one promise this project cannot fake.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skillweaver.config import Settings, settings
from skillweaver.contracts import Fingerprint, Provenance, Skill, SkillStats, SkillStore
from skillweaver.errors import SkillNotFound, SkillWeaverError
from skillweaver.skills.model import InvalidSkillDomain, InvalidSkillName
from skillweaver.skills.store import CODE_FILE, MANIFEST_FILE, META_FILE, FileSkillStore

PROVENANCE = Provenance("run-7", "pay an invoice", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))

# Deliberately awkward: a tab, a trailing blank line, a non-ASCII character and CRLF,
# so "byte-identical" means something.
GNARLY_CODE = (
    'def run(ctx, company):\r\n\t"""Search for ‹company›."""\n    ctx.ctl.type_text(company)\n\n'
)


@pytest.fixture
def store(tmp_path: Path) -> FileSkillStore:
    return FileSkillStore(tmp_path / "skills")


def make(name: str = "search_invoice", domain: str = "example.com", **overrides: object) -> Skill:
    fields: dict[str, object] = {
        "name": name,
        "domain": domain,
        "summary": f"Summary of {name}.",
        "docstring": f"What {name} does, at length.",
        "params": {"company": {"type": "string", "description": "Legal name"}},
        "code": f"def run(ctx, company):\n    ctx.ctl.type_text({name!r} + company)\n",
        "requires": (),
        "precondition": None,
        "verifier_code": None,
        "provenance": PROVENANCE,
    }
    fields.update(overrides)
    return Skill(**fields)  # type: ignore[arg-type]


# -- the layout and the round trip -----------------------------------------------------


def test_a_skill_round_trips_through_disk_byte_identically(store: FileSkillStore) -> None:
    original = make(
        code=GNARLY_CODE,
        params={
            "company": {"type": "string", "enum": ["Acme", "Globex"]},
            "limit": {"type": "integer", "default": 10},
            "rows": {"type": "array", "items": {"type": "object"}},
        },
        requires=("open_panel",),
        precondition=Fingerprint("fp-list", {"url": "u1", "layout": "l1"}),
        verifier_code="def verify(ctx, result):\n    return result.ok\n",
    )
    stored = store.put(original)
    loaded = store.get("search_invoice", "example.com")

    assert loaded == stored
    assert loaded == replace(original, version=1)
    # Byte-for-byte, not merely "equal after normalization".
    assert loaded.code == GNARLY_CODE
    code_path = store.version_dir("search_invoice", "example.com", 1) / CODE_FILE
    assert code_path.read_bytes() == GNARLY_CODE.encode("utf-8")
    assert loaded.params == original.params
    assert loaded.precondition == original.precondition
    assert loaded.precondition is not None and loaded.precondition.parts == {
        "url": "u1",
        "layout": "l1",
    }
    assert loaded.verifier_code == original.verifier_code


def test_a_fresh_store_reads_what_a_previous_process_wrote(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    FileSkillStore(root).put(make(code=GNARLY_CODE))
    assert FileSkillStore(root).get("search_invoice", "example.com").code == GNARLY_CODE


def test_the_layout_is_the_documented_one(store: FileSkillStore) -> None:
    store.put(make(domain="desktop:finder"))
    version_dir = store.root / "desktop:finder" / "search_invoice" / "v1"
    assert sorted(p.name for p in version_dir.iterdir()) == sorted([CODE_FILE, META_FILE])
    assert (store.root / MANIFEST_FILE).is_file()
    # The metadata holds no copy of the code: one file is the source of truth.
    meta = json.loads((version_dir / META_FILE).read_text())
    assert "code" not in meta and meta["name"] == "search_invoice"


def test_a_file_store_satisfies_the_protocol(store: FileSkillStore) -> None:
    assert isinstance(store, SkillStore)


def test_the_default_root_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILLWEAVER_DATA_DIR", "/tmp/skillweaver-settings-check")
    settings.cache_clear()
    try:
        assert (
            FileSkillStore().root
            == Settings(data_dir=Path("/tmp/skillweaver-settings-check")).skills_dir
        )
    finally:
        settings.cache_clear()


# -- versioning ------------------------------------------------------------------------


def test_put_twice_yields_v1_and_v2_and_both_stay_readable(store: FileSkillStore) -> None:
    first = store.put(make(code="def run(ctx, company):\n    pass  # v1\n"))
    second = store.put(make(code="def run(ctx, company):\n    pass  # v2\n"))
    assert (first.version, second.version) == (1, 2)

    assert store.get("search_invoice", "example.com", 1).code.endswith("# v1\n")
    assert store.get("search_invoice", "example.com", 2).code.endswith("# v2\n")
    # No version argument means the newest.
    assert store.get("search_invoice", "example.com").version == 2
    assert store.versions("search_invoice", "example.com") == [1, 2]
    # The older version is still a directory of its own: nothing was overwritten.
    assert (store.version_dir("search_invoice", "example.com", 1) / CODE_FILE).is_file()


def test_the_incoming_version_is_ignored(store: FileSkillStore) -> None:
    assert store.put(make(version=97)).version == 1


def test_a_version_that_was_never_written_is_not_found(store: FileSkillStore) -> None:
    store.put(make())
    for bad in (0, 2, 99):
        with pytest.raises(SkillNotFound):
            store.get("search_invoice", "example.com", bad)


def test_an_unknown_skill_is_not_found(store: FileSkillStore) -> None:
    with pytest.raises(SkillNotFound, match="no skill 'nope'"):
        store.get("nope", "example.com")
    store.put(make())
    with pytest.raises(SkillNotFound):
        store.get("search_invoice", "other.test")


def test_put_refuses_a_key_that_cannot_be_a_path(store: FileSkillStore) -> None:
    with pytest.raises(InvalidSkillName):
        store.put(make(name="../escape"))
    with pytest.raises(InvalidSkillDomain):
        store.put(make(domain="../escape"))
    assert not list(store.root.glob("**/escape"))


def test_put_never_overwrites_a_version_another_writer_claimed(store: FileSkillStore) -> None:
    store.put(make(code="def run(ctx, company):\n    pass  # v1\n"))
    # Simulate a second process that wrote v2 behind this store's back.
    other = FileSkillStore(store.root)
    other.put(make(code="def run(ctx, company):\n    pass  # other v2\n"))

    third = store.put(make(code="def run(ctx, company):\n    pass  # v3\n"))
    assert third.version == 3
    assert store.get("search_invoice", "example.com", 2).code.endswith("# other v2\n")


# -- listing ---------------------------------------------------------------------------


def test_list_gives_the_latest_of_each_sorted_by_domain_then_name(store: FileSkillStore) -> None:
    store.put(make("pay_invoice", "example.com"))
    store.put(make("search_invoice", "example.com"))
    store.put(
        make("search_invoice", "example.com", code="def run(ctx, company):\n    pass  # v2\n")
    )
    store.put(make("open_app", "desktop:finder"))

    listed = store.list()
    assert [(s.domain, s.name, s.version) for s in listed] == [
        ("desktop:finder", "open_app", 1),
        ("example.com", "pay_invoice", 1),
        ("example.com", "search_invoice", 2),
    ]
    assert [s.name for s in store.list(domain="example.com")] == ["pay_invoice", "search_invoice"]
    assert store.list(domain="nowhere.test") == []


def test_an_empty_store_lists_nothing(store: FileSkillStore) -> None:
    assert store.list() == []
    assert not store.root.exists()


def test_the_manifest_is_a_cache_that_can_be_rebuilt(store: FileSkillStore) -> None:
    store.put(make("pay_invoice"))
    store.put(make("search_invoice"))
    (store.root / MANIFEST_FILE).write_text("{ this is not json")

    fresh = FileSkillStore(store.root)  # falls back to a directory scan
    assert [s.name for s in fresh.list()] == ["pay_invoice", "search_invoice"]
    assert fresh.rebuild_manifest() == 2
    assert json.loads((store.root / MANIFEST_FILE).read_text())["manifest_version"] == 1


def test_an_interrupted_put_is_ignored_by_listing(store: FileSkillStore) -> None:
    store.put(make())
    # A crash between claiming the directory and writing meta.json.
    (store.root / "example.com" / "search_invoice" / "v2").mkdir()
    fresh = FileSkillStore(store.root)
    fresh.rebuild_manifest()
    assert fresh.get("search_invoice", "example.com").version == 1
    assert fresh.versions("search_invoice", "example.com") == [1]
    # And the next real put skips the claimed number rather than colliding with it.
    assert fresh.put(make()).version == 3


# -- statistics ------------------------------------------------------------------------


def test_record_run_folds_statistics_correctly_over_several_runs(store: FileSkillStore) -> None:
    store.put(make())
    assert store.get("search_invoice", "example.com").stats == SkillStats()

    # ok, 100ms: 1 run, 1 success, mean of {100} = 100.
    first = store.record_run("search_invoice", "example.com", ok=True, ms=100.0)
    assert (first.stats.runs, first.stats.successes, first.stats.mean_ms) == (1, 1, 100.0)
    assert first.stats.last_ok_at is not None

    # failure, 900ms: counted as a run, NOT as a success, and does not move the mean.
    second = store.record_run("search_invoice", "example.com", ok=False, ms=900.0)
    assert (second.stats.runs, second.stats.successes, second.stats.mean_ms) == (2, 1, 100.0)
    assert second.stats.last_ok_at == first.stats.last_ok_at

    # ok, 300ms: mean of {100, 300} = 200.
    third = store.record_run("search_invoice", "example.com", ok=True, ms=300.0)
    assert (third.stats.runs, third.stats.successes, third.stats.mean_ms) == (3, 2, 200.0)
    assert third.stats.last_ok_at >= first.stats.last_ok_at  # type: ignore[operator]

    # ok, 500ms: mean of {100, 300, 500} = 300.
    fourth = store.record_run("search_invoice", "example.com", ok=True, ms=500.0)
    assert (fourth.stats.runs, fourth.stats.successes, fourth.stats.mean_ms) == (4, 3, 300.0)

    # And all of it is on disk, not just in the returned object.
    assert FileSkillStore(store.root).get("search_invoice", "example.com").stats == fourth.stats


def test_record_run_never_touches_the_code(store: FileSkillStore) -> None:
    store.put(make(code=GNARLY_CODE))
    code_path = store.version_dir("search_invoice", "example.com", 1) / CODE_FILE
    before = code_path.read_bytes(), code_path.stat().st_mtime_ns
    for _ in range(3):
        store.record_run("search_invoice", "example.com", ok=True, ms=10.0)
    assert (code_path.read_bytes(), code_path.stat().st_mtime_ns) == before


def test_record_run_lands_on_the_latest_version_only(store: FileSkillStore) -> None:
    store.put(make())
    store.put(make())
    store.record_run("search_invoice", "example.com", ok=True, ms=50.0)
    assert store.get("search_invoice", "example.com", 1).stats.runs == 0
    assert store.get("search_invoice", "example.com", 2).stats.runs == 1


def test_record_run_on_an_unknown_skill_raises(store: FileSkillStore) -> None:
    with pytest.raises(SkillNotFound):
        store.record_run("nope", "example.com", ok=True, ms=1.0)


# -- demotion --------------------------------------------------------------------------


def test_demote_hides_a_skill_from_listing_but_keeps_it_on_disk(store: FileSkillStore) -> None:
    store.put(make("search_invoice"))
    store.put(make("pay_invoice"))

    demoted = store.demote("search_invoice", "example.com", "selector moved in the redesign")
    assert demoted.demoted_reason == "selector moved in the redesign"

    assert [s.name for s in store.list()] == ["pay_invoice"]
    assert [s.name for s in store.list(include_demoted=True)] == ["pay_invoice", "search_invoice"]
    # get() still returns it, reason attached, and the code is untouched.
    still_there = store.get("search_invoice", "example.com")
    assert still_there.demoted_reason == "selector moved in the redesign"
    assert still_there.code == make("search_invoice").code
    assert (store.version_dir("search_invoice", "example.com", 1) / CODE_FILE).is_file()

    # And a fresh store agrees, from the manifest it wrote.
    assert [s.name for s in FileSkillStore(store.root).list()] == ["pay_invoice"]


def test_a_later_put_of_a_fixed_version_is_healthy_again(store: FileSkillStore) -> None:
    store.put(make())
    store.demote("search_invoice", "example.com", "broken")
    fixed = store.put(make(code="def run(ctx, company):\n    pass  # fixed\n"))
    assert fixed.version == 2 and fixed.demoted_reason is None
    assert [s.name for s in store.list()] == ["search_invoice"]
    # The broken one is still recoverable for a post-mortem.
    assert store.get("search_invoice", "example.com", 1).demoted_reason == "broken"


def test_demote_needs_a_reason_a_human_can_read(store: FileSkillStore) -> None:
    store.put(make())
    with pytest.raises(SkillWeaverError, match="non-empty reason"):
        store.demote("search_invoice", "example.com", "  ")


def test_demote_on_an_unknown_skill_raises(store: FileSkillStore) -> None:
    with pytest.raises(SkillNotFound):
        store.demote("nope", "example.com", "broken")


# -- durability ------------------------------------------------------------------------


def test_no_temporary_files_are_left_behind(store: FileSkillStore) -> None:
    store.put(make())
    store.record_run("search_invoice", "example.com", ok=True, ms=1.0)
    store.demote("search_invoice", "example.com", "done with it")
    assert [p.name for p in store.root.rglob("*.tmp")] == []
