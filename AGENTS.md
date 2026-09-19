# Project agent memory

Two rules the code cannot tell you, because this repository is built by many workers in parallel:

- **The shared surface is a coordination decision, not a local edit.**
  `src/skillweaver/contracts.py` and every package `__init__.py` under `src/skillweaver/` are imported by everyone.
  Do not change them as part of other work: report the change you need and let it be coordinated.
- **Every piece of work owns an exclusive file list.**
  Create and edit only the files your task names.
  If you need a change in a file you do not own, report it instead of editing it.

The command line is the way to drive all of this: `uv run python -m skillweaver.cli --help`.
There is no `skillweaver` console script - `pyproject.toml` has no `[project.scripts]`, and
that file is shared surface. `src/skillweaver/orchestrator.py` holds the cold-versus-warm
decision and `build_agent`, the one place the real agent is wired; build on that rather than
assembling a planner and an explorer by hand.

Test without a browser, a model or a network by using the doubles in `tests/fakes/` (fixtures in `tests/conftest.py`; `tests/fakes/scenario.py` is a small fake app to drive).

Two live-API facts that no test can teach you, both already paid for in lost runs.
`claude-opus-5` **refuses an assistant prefill** - a conversation ending on an assistant
turn is a 400 - so JSON is obtained by asking tolerantly and re-asking, not by prefilling;
and the Anthropic SDK **refuses a non-streaming request** whose implied duration passes
ten minutes, which a large `max_tokens` alone is enough to trigger. Both are written up
where they bite, at `_MAX_REPLY_TOKENS` in `src/skillweaver/skills/synthesize.py`.

Anything the agent learns must be RE-RUN before it is stored, so a task that changes
state cannot be learned unless something can change it back: pass `--reset-url`
(`learn --help`), and see `WorldReset` in `src/skillweaver/orchestrator.py`. The sandbox
site's `GET /__reset` is one instance of it.

A live page never fingerprints identically twice, so nothing that compares two screens
may ask it to: same-page-on-a-second-load bottoms out around 0.84 while a genuinely
different screen tops out around 0.28. Both cuts that judge it - `SAME_STATE_THRESHOLD`
in `perception/fingerprint.py` and `MIN_PRECONDITION_SIMILARITY` in
`skills/synthesize.py` - are 0.62, and each carries the measurements it was calibrated
from. An exact match is a property of the demo site alone, which is why a gate tuned
against the sandbox looks green until it meets a website.

Prefer **headless** against a REAL site anyway: two fresh browsers of opposite modes on
one Wikipedia page score 0.44. Block anything that renders only SOMETIMES, too:
Wikipedia's fundraising banner appears on some loads, pushes the article down, and a page
re-opened with one scores **0.04** against the same page recorded without one - one part
in twenty-five, the URL - so the library never grows and a retrieval measurement has
nothing to retrieve. Aborting `**/Special:BannerLoader*`, `**/Special:RecordImpression*`
and `**/geoiplookup*` on the Playwright context makes one URL fingerprint identically
every time. What else a live-site run needs is in `eval/wikipedia.yaml`.

Perception is dominated by OCR - 84-97% of every observation's time on real pages -
so the shipped fix is to not read the same pixels twice, and counts, not seconds, are how
that is judged (seconds move with machine load; this project has been burned by that
twice). `PerceptionCounters` and `CachingTextReader` in
`src/skillweaver/perception/ocr.py` hold the measured reasons the cache key is the exact
pixels and not "the same state", and `ComposedPerceiver` in `src/skillweaver/orchestrator.py`
holds the measured reason the text is NOT read lazily. Read both before trying either again.

`ultralytics` is a noisy import; three of its side effects have already cost time here.

- It installs its own top-level `tests` package into the venv, which shadows this repository's
  `tests/` in any plain `python` process. Pytest is unaffected; a script that needs the fakes
  must bind them first:
  `sys.modules["tests"] = types.ModuleType("tests"); sys.modules["tests"].__path__ = ["tests"]`.
- A relative `project=` given to `model.train()` resolves against ultralytics' GLOBAL
  `runs_dir` setting, which is per-user and can point at a different worktree entirely, so a
  training run silently writes into someone else's checkout. Pass an absolute path and read
  the checkpoint back from `model.trainer.best`; `scripts/train_detector.py` shows both.
- Importing it patches `PIL.Image.open`; `skillweaver.perception.detect_yolo._import_yolo`
  explains why that has to be undone and does it.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
