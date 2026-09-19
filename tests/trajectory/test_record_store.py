"""Recording and reloading: the format is lossless, cheap to list, and survives a
run that is killed halfway."""

from __future__ import annotations

import builtins
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skillweaver.contracts import (
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStore,
    action_to_dict,
)
from skillweaver.errors import SkillWeaverError
from skillweaver.trajectory.record import Recorder
from skillweaver.trajectory.store import (
    INCOMPLETE_NOTE,
    SCHEMA_VERSION,
    SCREENS_DIR,
    TRAJECTORY_FILE,
    TrajectoryFileStore,
)
from tests.fakes import Scenario
from tests.trajectory.conftest import RUN_ACTIONS, drive

# -- the Protocols hold -----------------------------------------------------------------


def test_recorder_and_store_satisfy_their_protocols(store_root: Path) -> None:
    assert isinstance(Recorder(store_root), TrajectoryRecorder)
    assert isinstance(TrajectoryFileStore(store_root), TrajectoryStore)


def test_root_defaults_to_the_configured_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    from skillweaver.config import settings

    monkeypatch.setenv("SKILLWEAVER_DATA_DIR", "/tmp/skillweaver-test-data")
    settings.cache_clear()
    try:
        expected = settings().trajectories_dir
        assert Recorder().root == expected
        assert TrajectoryFileStore().root == expected
        assert expected == Path("/tmp/skillweaver-test-data/trajectories")
    finally:
        settings.cache_clear()


# -- round trip -------------------------------------------------------------------------


def test_reloaded_run_has_the_identical_action_sequence(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)

    assert [s.action for s in loaded.steps] == list(RUN_ACTIONS)
    assert [s.action for s in loaded.steps] == [s.action for s in recorded.steps]
    assert [s.index for s in loaded.steps] == list(range(len(RUN_ACTIONS)))
    # every variant survived encoding, field for field
    for original, reloaded in zip(recorded.steps, loaded.steps, strict=True):
        assert action_to_dict(reloaded.action) == action_to_dict(original.action)
        assert type(reloaded.action) is type(original.action)


def test_reloaded_run_keeps_everything_around_the_actions(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    loaded = store.load(recorded.run_id)

    assert (loaded.run_id, loaded.task, loaded.domain) == (
        recorded.run_id,
        recorded.task,
        recorded.domain,
    )
    assert (loaded.ok, loaded.note) == (recorded.ok, recorded.note)
    assert loaded.started_at == recorded.started_at and loaded.finished_at == recorded.finished_at
    for original, reloaded in zip(recorded.steps, loaded.steps, strict=True):
        assert reloaded.result == original.result
        assert reloaded.verdict == original.verdict
        assert reloaded.note == original.note
        assert reloaded.before.elements == original.before.elements
        assert reloaded.after.fingerprint == original.after.fingerprint
        assert reloaded.after.fingerprint.parts == original.after.fingerprint.parts
        assert reloaded.after.url == original.after.url
        assert reloaded.after.screenshot.png == original.after.screenshot.png
        assert reloaded.after.screenshot.scale == original.after.screenshot.scale
    # the index is rebuilt, and it works
    confirm = loaded.steps[1].after.index.find_text("Confirm payment")
    assert confirm and confirm[0].box.center.x == 100


def test_the_failed_navigate_round_trips_as_a_failure(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    last = store.load(recorded.run_id).steps[-1]
    assert last.action.kind == "navigate"
    assert last.result.ok is False
    assert last.result.error and "navigate" in last.result.error


def test_store_save_replaces_the_run_in_place(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    store.save(recorded)
    store.save(recorded)  # idempotent: a recorder already wrote this file

    assert store.list() == [recorded.run_id]
    assert store.load(recorded.run_id).steps[0].action == RUN_ACTIONS[0]


# -- screenshots are stored once --------------------------------------------------------


def test_each_distinct_screen_is_stored_exactly_once(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    screens = sorted((store.path_of(recorded.run_id) / SCREENS_DIR).glob("*.png"))
    distinct = {s.before.screenshot.png for s in recorded.steps} | {
        s.after.screenshot.png for s in recorded.steps
    }

    # ten steps, twenty observations, four distinct screens on disk
    assert len(recorded.steps) == 10
    assert len(screens) == len(distinct) == 4
    assert {p.read_bytes() for p in screens} == distinct


def test_loading_without_screenshots_reads_no_png(
    recorded: Trajectory, store: TrajectoryFileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _spy_on_open(monkeypatch)

    lean = store.load(recorded.run_id, screenshots=False)

    assert [s.action for s in lean.steps] == list(RUN_ACTIONS)
    assert all(s.before.screenshot.png == b"" for s in lean.steps)
    assert lean.steps[0].before.screenshot.width == 800  # the metadata is still there
    assert not [p for p in opened if p.endswith(".png")]


# -- cheap listing ----------------------------------------------------------------------


def _spy_on_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path opened for reading, through either door into the file system."""
    opened: list[str] = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    return opened


def test_list_opens_no_file_at_all(
    recorded: Trajectory, store: TrajectoryFileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _spy_on_open(monkeypatch)

    assert store.list() == [recorded.run_id]

    assert opened == [], f"list() must be a scandir, but it opened {opened}"


def test_list_is_oldest_first_and_summaries_skip_the_pixels(
    scenario: Scenario,
    store_root: Path,
    store: TrajectoryFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a deterministic clock so the three runs cannot share a start timestamp
    moments = iter([datetime(2026, 9, 19, 12, minute, tzinfo=UTC) for minute in range(10)])
    ids = []
    for number in range(3):
        recorder = Recorder(store_root, clock=lambda: next(moments))
        ids.append(recorder.start(f"task {number}", "fake.test"))
        drive(scenario, recorder, RUN_ACTIONS[:1])
        recorder.finish(ok=number != 1, note=f"note {number}")
        scenario.controller.reset()

    assert store.list() == ids  # oldest first, by the timestamp in the directory name

    opened = _spy_on_open(monkeypatch)
    summaries = store.summaries()

    assert [s.run_id for s in summaries] == ids
    assert [s.task for s in summaries] == ["task 0", "task 1", "task 2"]
    assert [s.ok for s in summaries] == [True, False, True]
    assert all(s.complete for s in summaries)
    assert not [p for p in opened if p.endswith(".png")]
    assert len([p for p in opened if p.endswith(TRAJECTORY_FILE)]) == 3


def test_frames_give_screenshot_paths_without_reading_them(
    recorded: Trajectory, store: TrajectoryFileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = _spy_on_open(monkeypatch)

    frames = store.frames(recorded.run_id)

    assert [f.index for f in frames] == list(range(10))
    assert not [p for p in opened if p.endswith(".png")]
    # consecutive steps share a screen, so the filmstrip can skip repeats
    assert frames[0].after == frames[1].before
    assert all(f.before.is_file() and f.after.is_file() for f in frames)
    assert frames[0].before.read_bytes() == recorded.steps[0].before.screenshot.png


# -- crash safety -----------------------------------------------------------------------


def test_a_run_killed_before_finish_still_loads(
    scenario: Scenario, recorder: Recorder, store: TrajectoryFileStore
) -> None:
    run_id = recorder.start(scenario.task.text, scenario.task.domain)
    drive(scenario, recorder, RUN_ACTIONS[:2])
    # the process dies here: finish() is never called and save() never happens

    loaded = store.load(run_id)

    assert [s.action for s in loaded.steps] == list(RUN_ACTIONS[:2])
    assert loaded.ok is False
    assert loaded.note == INCOMPLETE_NOTE
    assert loaded.finished_at == loaded.steps[-1].after.taken_at
    assert store.summary(run_id).complete is False


def test_a_trajectory_truncated_mid_line_loads_the_earlier_steps(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    path = store.path_of(recorded.run_id) / TRAJECTORY_FILE
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = lines[:4]  # header + three steps
    torn = "".join(kept) + lines[4][:80]  # ... and a step cut off mid-write
    path.write_text(torn, encoding="utf-8")

    loaded = store.load(recorded.run_id)

    assert [s.action for s in loaded.steps] == list(RUN_ACTIONS[:3])
    assert loaded.ok is False and loaded.note == INCOMPLETE_NOTE
    assert loaded.task == recorded.task
    # and the cheap view agrees, without decoding a single step
    assert store.summary(recorded.run_id).complete is False


def test_corruption_before_the_last_line_is_reported(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    path = store.path_of(recorded.run_id) / TRAJECTORY_FILE
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = "{ this is not json\n"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(SkillWeaverError, match="not valid JSON"):
        store.load(recorded.run_id)


# -- schema version ---------------------------------------------------------------------


def test_every_file_carries_the_schema_version(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    path = store.path_of(recorded.run_id) / TRAJECTORY_FILE
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert header["type"] == "header" and header["schema"] == SCHEMA_VERSION


def test_an_unknown_schema_version_is_refused_clearly(
    recorded: Trajectory, store: TrajectoryFileStore
) -> None:
    path = store.path_of(recorded.run_id) / TRAJECTORY_FILE
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header = json.loads(lines[0])
    header["schema"] = SCHEMA_VERSION + 98
    lines[0] = json.dumps(header) + "\n"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(SkillWeaverError, match=f"schema version 99 .*version {SCHEMA_VERSION}"):
        store.load(recorded.run_id)
    with pytest.raises(SkillWeaverError, match="schema version"):
        store.summary(recorded.run_id)


# -- failure modes ----------------------------------------------------------------------


def test_unknown_run_id_is_an_error(store: TrajectoryFileStore) -> None:
    assert store.list() == []
    with pytest.raises(SkillWeaverError, match="no trajectory with run_id 'nope'"):
        store.load("nope")


def test_one_run_at_a_time(scenario: Scenario, recorder: Recorder) -> None:
    with pytest.raises(SkillWeaverError, match="no run in progress"):
        recorder.finish(ok=True)
    run_id = recorder.start(scenario.task.text, scenario.task.domain)
    with pytest.raises(SkillWeaverError, match=f"run {run_id} is still in progress"):
        recorder.start("another task", "fake.test")
    recorder.finish(ok=False)
    assert recorder.run_id is None and recorder.path is None
    recorder.start("a second run", "fake.test")  # the recorder is reusable
    assert recorder.steps == 0
