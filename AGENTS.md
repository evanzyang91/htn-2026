# Project agent memory

Two rules the code cannot tell you, because this repository is built by many workers in parallel:

- **The shared surface is a coordination decision, not a local edit.**
  `src/skillweaver/contracts.py` and every package `__init__.py` under `src/skillweaver/` are imported by everyone.
  Do not change them as part of other work: report the change you need and let it be coordinated.
- **Every piece of work owns an exclusive file list.**
  Create and edit only the files your task names.
  If you need a change in a file you do not own, report it instead of editing it.

Test without a browser, a model or a network by using the doubles in `tests/fakes/` (fixtures in `tests/conftest.py`; `tests/fakes/scenario.py` is a small fake app to drive).

`ultralytics` installs its own top-level `tests` package into the venv, which shadows this
repository's `tests/` in any plain `python` process. Pytest is unaffected; a script that needs
the fakes must bind them first:
`sys.modules["tests"] = types.ModuleType("tests"); sys.modules["tests"].__path__ = ["tests"]`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
