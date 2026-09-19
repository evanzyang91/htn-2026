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
