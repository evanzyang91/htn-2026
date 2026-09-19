"""Tests for ``skillweaver.dashboard.build``.

Everything here runs against the committed fixture data directories under
``tests/dashboard_fixtures`` (regenerate them with ``_generate.py`` in that
directory, which writes them through the real stores). Nothing touches a network, a
model or a browser.

Three things are worth stating about what these tests check:

*The four panels render their real content*, not just "some HTML came out": the
counts and labels asserted below are the ones a viewer reads off the page.

*The output is genuinely self-contained.* Two checks, deliberately: a structural one
that no element loads anything (no ``<script src>``, ``<link>``, ``@import``, or
``src``/``href`` that is not a ``data:`` URI), which must hold for any input; and a
blunt "the string ``http://`` does not appear at all" over the fixture corpus, which
holds because the builder strips URL schemes from everything it displays.

*Every empty state renders when its input is missing*, including the one that matters
most - a metrics file in a shape this module does not understand must produce the
panel's empty state and never an exception.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from skillweaver.dashboard.build import (
    EvalRun,
    EvalTask,
    build_dashboard,
    collect,
    fit,
    human_ms,
    percent,
    read_eval_reports,
    strip_scheme,
    success_rate,
)

FIXTURES = Path(__file__).parent / "dashboard_fixtures"
FULL = FIXTURES / "full"
BROKEN = FIXTURES / "broken"
COLD_ONLY = FIXTURES / "cold_only"

VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}  # fmt: skip


@pytest.fixture(scope="module")
def full_page(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The dashboard built once from the full fixture corpus."""
    out = tmp_path_factory.mktemp("dash") / "index.html"
    return build_dashboard(FULL, out).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def empty_page(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The dashboard built from a data directory that does not exist at all."""
    out = tmp_path_factory.mktemp("dash") / "index.html"
    return build_dashboard(tmp_path_factory.mktemp("nothing") / "absent", out).read_text("utf-8")


# --------------------------------------------------------------------------------------
# Panel 1: the skill library
# --------------------------------------------------------------------------------------


def test_every_stored_skill_is_on_the_page(full_page: str) -> None:
    for name in (
        "open_records",
        "search_records",
        "open_record_detail",
        "send_reply",
        "export_csv",
        "old_search",
    ):
        assert f">{name}</span>" in full_page, name
    assert full_page.count('<article class="skill') == 6


def test_skills_are_ordered_newest_first(full_page: str) -> None:
    order = [m.group(1) for m in re.finditer(r'<span class="n">([a-z_]+)</span>', full_page)]
    assert order == [
        "export_csv",  # learned 2026-09-19 08:02, the newest
        "send_reply",
        "open_record_detail",
        "search_records",
        "open_records",
        "old_search",  # learned 2026-09-16 13:20, the oldest
    ]
    assert order.index("export_csv") < order.index("old_search")


def test_each_skill_shows_what_it_does_its_params_and_its_record(full_page: str) -> None:
    assert "Search the records table for a company." in full_page
    assert "89% of 9 runs, 980ms mean" in full_page  # rate and run count
    assert "100% of 12 runs, 640ms mean" in full_page
    assert "not run yet" in full_page  # export_csv has no runs
    assert ">company</span>" in full_page and ">string</span>" in full_page
    assert "Company name to filter by" in full_page
    assert "takes no parameters" in full_page  # open_records
    assert "2026-09-17 15:40 UTC" in full_page  # when it was learned
    assert "by claude-opus-5" in full_page


def test_a_demoted_skill_is_shown_and_explained(full_page: str) -> None:
    assert "demoted</span>" in full_page
    assert "Retired: selector drifted after the console moved search" in full_page


def test_the_growth_curve_is_drawn_when_timestamps_support_it(full_page: str) -> None:
    assert "Skills known over time - 6 learned" in full_page
    assert full_page.count('<path class="line"') == 1
    assert full_page.count('<circle class="pt"') == 6  # one point per skill


# --------------------------------------------------------------------------------------
# Panel 2: the site graph
# --------------------------------------------------------------------------------------


def test_every_state_and_transition_is_drawn(full_page: str) -> None:
    assert full_page.count('<g class="gnode') == 5
    assert full_page.count('<g class="gedge') == 7
    for label in ("records list", "search results", "record detail", "mail inbox", "compose reply"):
        assert label in full_page, label
    assert "5 screens" in full_page and "7 actions" in full_page


def test_each_edge_is_labelled_with_its_success_rate(full_page: str) -> None:
    labels = re.findall(r'<text class="elabel"[^>]*>([^<]+)</text>', full_page)
    assert sorted(labels) == ["100%", "100%", "100%", "33%", "50%", "80%", "89%"]


def test_edge_labels_never_print_on_top_of_each_other(full_page: str) -> None:
    """Two percentages in the same 18 pixels are the one thing this panel must not do."""
    pills = [
        (float(m.group(1)), float(m.group(2)))
        for m in re.finditer(
            r'<rect class="epill" x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)"', full_page
        )
    ]
    widths = [float(m) for m in re.findall(r'<rect class="epill"[^>]*width="([\d.]+)"', full_page)]
    assert len(pills) == 7
    for i, ((x1, y1), w1) in enumerate(zip(pills, widths, strict=True)):
        for (x2, y2), w2 in list(zip(pills, widths, strict=True))[i + 1 :]:
            overlaps_x = abs((x1 + w1 / 2) - (x2 + w2 / 2)) < (w1 + w2) / 2
            overlaps_y = abs(y1 - y2) < 18
            assert not (overlaps_x and overlaps_y), f"labels overlap at {y1} and {y2}"


def test_node_text_is_shortened_to_fit_inside_its_box(full_page: str) -> None:
    """SVG text neither wraps nor clips, so an unshortened label spills over the canvas."""
    assert "sandbox.test/records?q…" in full_page  # shortened with an ellipsis
    assert "sandbox.test/records?q=acme</text>" not in full_page


def test_the_routed_path_is_precomputed_and_highlightable(full_page: str) -> None:
    routes = json.loads(
        re.search(r'id="route-data">(.*?)</script>', full_page, re.S).group(1)  # type: ignore[union-attr]
    )
    to_detail = routes["g0"]["n0_fp-record-detail"]
    # The direct list -> detail edge is faster per attempt but succeeds 1 time in 3, so
    # the cost model (mean_ms / success rate) routes around it through the search screen.
    assert to_detail["hops"] == 2
    assert to_detail["steps"] == 4
    assert to_detail["cost"] == pytest.approx(1332.5)
    assert len(to_detail["edges"]) == 2
    assert to_detail["nodes"][0] == "n0_fp-records-list"
    assert to_detail["nodes"][-1] == "n0_fp-record-detail"


def test_the_canvas_is_big_enough_for_everything_drawn_on_it(full_page: str) -> None:
    view_box = re.search(r'<svg data-graph="g0"[^>]*viewBox="([^"]+)"', full_page)
    min_x, min_y, width, height = (float(v) for v in view_box.group(1).split())  # type: ignore[union-attr]
    points = [
        (float(x), float(y))
        for d in re.findall(r'<path class="wire" d="([^"]+)"', full_page)
        for x, y in re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)
    ]
    assert points
    assert all(min_x <= x <= min_x + width for x, _ in points)
    assert all(min_y <= y <= min_y + height for _, y in points)


# --------------------------------------------------------------------------------------
# Panel 3: cold versus warm
# --------------------------------------------------------------------------------------


def test_cold_and_warm_are_charted_for_every_repeated_task(full_page: str) -> None:
    assert "Find the invoice for Acme Corp and open it" in full_page
    assert "Open the detail view of invoice INV-101" in full_page
    assert "Reply to the newest message in the inbox" in full_page
    assert full_page.count('class="bar cold"') == 6  # three tasks x (time, calls)
    assert full_page.count('class="bar warm"') == 6


def test_the_speedup_factors_are_computed_from_the_metrics(full_page: str) -> None:
    # find_invoice: 48120ms cold against the mean of 6210 and 5890 warm.
    assert "8.0x faster" in full_page
    assert "11.5x fewer" in full_page
    assert "48.1s" in full_page and "6.0s" in full_page


def test_a_failed_warm_run_is_excluded_from_the_warm_mean(full_page: str) -> None:
    """send_reply's third run failed after 41s; averaging it in would hide the win."""
    assert "15.8s" in full_page  # the one successful warm run, not (15800+41000)/2
    assert "2 warm run(s), 1 ok" in full_page


def test_headline_stats_carry_the_ten_second_read(full_page: str) -> None:
    assert "faster once learned" in full_page
    assert "fewer model calls" in full_page
    assert "skills in the library" in full_page
    assert "screens mapped" in full_page


def test_a_task_with_no_warm_run_is_listed_as_pending(full_page: str) -> None:
    assert "still waiting for a warm encounter" in full_page
    assert "export_csv" in full_page


# --------------------------------------------------------------------------------------
# Panel 4: the run filmstrip
# --------------------------------------------------------------------------------------


def test_the_filmstrip_shows_every_step_of_the_run(full_page: str) -> None:
    assert full_page.count('class="frame') == 5
    assert "Find the invoice for Acme Corp and open it" in full_page
    assert "a1b2c3d4e5f6" in full_page
    assert "5 steps" in full_page


def test_each_frame_carries_its_screenshot_action_and_verdict(full_page: str) -> None:
    assert full_page.count('class="shot" src="data:image/png;base64,') == 5
    for verb in ("navigate", "click", "type", "press"):
        assert f'<span class="verb">{verb}</span>' in full_page, verb
    assert "&#34;Acme Corp&#34;" in full_page
    assert full_page.count(">pass</span>") == 5
    assert "three matching invoices are listed (model)" in full_page
    assert "&ldquo;submit the search&rdquo;" in full_page  # the agent's stated reason


def test_a_navigate_action_shows_its_target_without_a_scheme(full_page: str) -> None:
    assert "sandbox.test/records" in full_page
    assert "//sandbox.test" not in full_page


# --------------------------------------------------------------------------------------
# Self-containment: the page must open with the network unplugged
# --------------------------------------------------------------------------------------


def test_nothing_on_the_page_loads_anything(full_page: str) -> None:
    assert "<script src=" not in full_page
    assert "<link " not in full_page
    assert "@import" not in full_page
    assert "url(http" not in full_page
    for attribute in re.findall(r'\b(?:src|href)="([^"]*)"', full_page):
        assert attribute.startswith(("data:", "#")), attribute


def test_no_absolute_url_survives_into_the_page(full_page: str) -> None:
    """Including the xmlns of inline SVG, whose value is an http:// URL."""
    assert "http://" not in full_page
    assert "https://" not in full_page
    assert "xmlns" not in full_page


def test_every_image_is_embedded_rather_than_linked(full_page: str) -> None:
    sources = re.findall(r'<(?:img|image)[^>]*(?:src|href)="([^"]{0,32})', full_page)
    assert sources, "the page should carry embedded screenshots"
    assert all(source.startswith("data:image/") for source in sources)


def test_the_page_is_one_file_with_balanced_markup(full_page: str) -> None:
    class Balance(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.stack: list[str] = []
            self.bad: list[str] = []

        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in VOID:
                self.stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in VOID:
                return
            if not self.stack or self.stack[-1] != tag:
                self.bad.append(f"</{tag}> closed {self.stack[-1:] or ['nothing']}")
            else:
                self.stack.pop()

    parser = Balance()
    parser.feed(full_page)
    assert parser.bad == []
    assert parser.stack == []


# --------------------------------------------------------------------------------------
# Empty states
# --------------------------------------------------------------------------------------


def test_a_missing_data_directory_still_produces_a_page(empty_page: str) -> None:
    assert empty_page.lstrip().startswith("<!doctype html>")
    assert empty_page.count('class="empty"') == 4  # one per panel


def test_every_panel_explains_what_is_missing(empty_page: str) -> None:
    assert "Nothing to compare yet" in empty_page
    assert "No evaluation results yet" in empty_page
    assert "No skills learned yet" in empty_page
    assert "The library is empty" in empty_page
    assert "No map yet" in empty_page
    assert "No site graph yet" in empty_page
    assert "No run to show" in empty_page
    assert "No runs recorded yet" in empty_page


def test_the_empty_page_documents_the_metrics_shape_it_wants(empty_page: str) -> None:
    """The eval harness does not exist yet; the empty panel is where its author looks."""
    assert "&lt;data&gt;/eval/*.json" in empty_page
    assert "&#34;wall_ms&#34;: 48120.0" in empty_page
    assert "&#34;llm_calls&#34;: 23" in empty_page


def test_the_empty_page_is_still_self_contained(empty_page: str) -> None:
    assert "http://" not in empty_page and "https://" not in empty_page
    assert "<script src=" not in empty_page and "<link " not in empty_page


def test_a_malformed_metrics_file_gives_the_empty_state_not_an_exception(
    tmp_path: Path,
) -> None:
    page = build_dashboard(BROKEN, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "Nothing to compare yet" in page
    assert "truncated.json: not readable JSON" in page
    assert "wrong_shape.json: has no &#39;tasks&#39; list" in page
    assert "8.0x faster" not in page  # nothing was charted


def test_tasks_run_only_once_get_their_own_empty_state(tmp_path: Path) -> None:
    page = build_dashboard(COLD_ONLY, tmp_path / "out.html").read_text(encoding="utf-8")
    assert "none has a second run yet" in page
    assert "find_invoice" in page
    assert 'class="bar cold"' not in page


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "not json at all",
        "[]",
        "{}",
        '{"tasks": []}',
        '{"tasks": [{"no_task_id": 1}]}',
        '{"tasks": [{"task_id": "t", "runs": "nope"}]}',
        '{"tasks": [{"task_id": "t", "runs": [{"wall_ms": "fast", "llm_calls": null}]}]}',
        '[{"task_id": "t", "runs": [{"attempt": 1, "wall_ms": 5}]}]',
        '{"tasks": [{"task_id": "t", "runs": [{"wall_ms": 1e400}]}]}',
    ],
)
def test_reading_metrics_never_raises_whatever_is_in_the_file(
    tmp_path: Path, contents: str
) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "m.json").write_text(contents, encoding="utf-8")
    reports, problems = read_eval_reports(tmp_path / "eval")
    assert isinstance(reports, list) and isinstance(problems, list)
    build_dashboard(tmp_path, tmp_path / "out.html")  # and the whole build survives it


def test_a_data_directory_of_empty_subdirectories_is_not_a_crash(tmp_path: Path) -> None:
    for name in ("skills", "graphs", "trajectories", "eval"):
        (tmp_path / name).mkdir()
    page = build_dashboard(tmp_path, tmp_path / "out.html").read_text(encoding="utf-8")
    assert page.count('class="empty"') == 4


# --------------------------------------------------------------------------------------
# The builder itself
# --------------------------------------------------------------------------------------


def test_build_dashboard_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = build_dashboard(FULL, tmp_path / "deep" / "deeper" / "index.html")
    assert out.is_file()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_the_same_data_at_the_same_moment_builds_the_same_page(tmp_path: Path) -> None:
    moment = datetime(2026, 9, 19, 12, tzinfo=UTC)
    first, second = collect(FULL, now=moment), collect(FULL, now=moment)
    assert first == second
    assert first.built_at == "2026-09-19 12:00 UTC"


def test_the_page_names_where_it_came_from(full_page: str) -> None:
    assert "dashboard_fixtures" in full_page
    assert "skillweaver.dashboard.build" in full_page


# --------------------------------------------------------------------------------------
# The small total conversions everything else is built on
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://sandbox.test/records", "sandbox.test/records"),
        ("http://sandbox.test/", "sandbox.test"),
        ("sandbox.test/records", "sandbox.test/records"),
        (None, ""),
        ("", ""),
        ("https://", "https://"),  # nothing left after stripping: keep the original
    ],
)
def test_strip_scheme(url: str | None, expected: str) -> None:
    assert strip_scheme(url) == expected


@pytest.mark.parametrize(
    ("part", "whole", "expected"),
    [(1, 2, 50.0), (0, 0, 0.0), (5, 0, 0.0), (-1, 4, 0.0), (9, 4, 100.0)],
)
def test_percent_is_always_a_drawable_width(part: float, whole: float, expected: float) -> None:
    assert percent(part, whole) == expected


def test_success_rate_distinguishes_never_tried_from_never_worked() -> None:
    assert success_rate(0, 0) is None
    assert success_rate(0, 4) == 0.0
    assert success_rate(3, 4) == 0.75


@pytest.mark.parametrize(
    ("ms", "expected"),
    [(0, "0ms"), (820, "820ms"), (6210, "6.2s"), (48120, "48.1s"), (130000, "2m 10s")],
)
def test_human_ms(ms: float, expected: str) -> None:
    assert human_ms(ms) == expected


def test_fit_shortens_only_what_does_not_fit() -> None:
    assert fit("inbox", 200, 7.0) == "inbox"
    shortened = fit("a very long screen label indeed", 60, 7.0)
    assert shortened.endswith("…")
    assert len(shortened) <= 60 // 7


def test_cold_is_the_lowest_attempt_however_the_runs_are_ordered() -> None:
    task = EvalTask(
        task_id="t",
        runs=(EvalRun(attempt=3, wall_ms=5.0), EvalRun(attempt=1, wall_ms=90.0)),
    )
    assert task.cold is not None
    assert task.cold.wall_ms == 90.0
    assert [run.attempt for run in task.warm] == [3]


def test_a_task_with_no_runs_has_no_cold_run() -> None:
    task = EvalTask(task_id="t")
    assert task.cold is None
    assert task.warm == ()
