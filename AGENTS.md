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

"Where the last run of this task ended" is NOT where this run should end: a task that
takes an argument ends somewhere the argument decides, so a recalled end screen can
say yes and must never say no. `TieredCritic` has three check roles for that reason -
evidence, corroboration, veto - and `_warm_critic` in `src/skillweaver/orchestrator.py`
holds the rule that picks one. A read-only live task needs no reset and should say so
(`read_only`, `ResetOutcome` in the same file); a `403` from a real site means the
endpoint is not a reset hook, not that the world could not be put back.

An attempt's cost is read from the model client's own `total_usage` across it
(`Agent._charge_model`), never from what the path reports about itself: a call the
composer spent on a plan that was then discarded, and a critic escalation the planner
never charged itself for, both vanish otherwise - and a number that flatters us is the
one kind of bug this project cannot ship.

A live page never fingerprints identically twice, so nothing that compares two screens
may ask it to - including routing, which is why `find_route` settles "am I already
there?" by similarity and not by id. One number answers that question everywhere:
`SAME_STATE_THRESHOLD` in `perception/fingerprint.py`, which `MIN_PRECONDITION_SIMILARITY`
and `DEFAULT_MATCH_THRESHOLD` both defer to, and which carries the two corpora it was
calibrated on. Re-derive it, do not nudge it, and re-derive it again if the fingerprinter
changes what it puts in `parts` - a cut is only meaningful against the shape of the signal
it judges. An exact match is a property of the demo site alone, which is why a gate tuned
against the sandbox looks green until it meets a website.

What moves a real page is a notice arriving at the top - a fundraising appeal, a cookie
bar, an A/B strip - which pushes everything below it DOWN. A fingerprint part must
therefore never be named by WHERE it is; `StateFingerprinter` names each one by its
content, so the same page pushed down 200px scores 0.78 where a grid-anchored part scored
it 0.040. A full-screen takeover is a different matter and is refused on purpose:
Wikipedia's appeal displaces 555 of 800 pixels, and no identity can recover a screen that
is 69% gone.

So block anything that renders only SOMETIMES rather than relying on the identity to
absorb it - a measurement wants one page to be one screen. Aborting
`**/Special:BannerLoader*`, `**/Special:RecordImpression*` and `**/geoiplookup*` on the
Playwright context makes one Wikipedia URL fingerprint identically every time.

Use **headless** against a REAL site, on both sides of anything that will be compared.
Headed and headless are different screens and are meant to be: two fresh browsers of
opposite modes on one page score 0.13 (Wikipedia Main Page) and 0.42 (`json.html`), both
at or below the same-state cut, while headless against headless scores 1.000. Note that
`orchestrator._open_world` opens the CLI's browser HEADED, so a skill learned through the
CLI does not match one replayed headless. What else a live-site run needs is in
`eval/wikipedia.yaml`.

Perception is dominated by OCR - 84-97% of every observation's time on real pages -
so the shipped fix is to not read the same pixels twice, and counts, not seconds, are how
that is judged (seconds move with machine load; this project has been burned by that
twice). `PerceptionCounters` and `CachingTextReader` in
`src/skillweaver/perception/ocr.py` hold the measured reasons the cache key is the exact
pixels and not "the same state", and `ComposedPerceiver` in `src/skillweaver/orchestrator.py`
holds the measured reason the text is NOT read lazily. Read both before trying either again.
The SECOND cost is the one a cache hit still pays: `merge_elements` compares every
candidate against every cluster, so it grows quadratically with the element count while
the text read does not grow with it at all - the reader reads the whole frame and is
keyed on pixels, so nothing about how many elements are detected can change it. That is
why the detector's ceiling is where it is; `DEFAULT_MAX_DETECTIONS` in
`perception/detect_yolo.py` carries the measurement, including the fact that the cap
truncates NO real page - what loses elements on a dense page is recall, by a factor of
three, and not the cap.

**Every limit in this project is enforced from Python, so none of them bounds a native
call.** The skill clock is read by `charge_step`, by the runner and by a trace hook that
fires on Python frames; a thread inside ONNX Runtime executes no frames and is therefore
unbounded - once measured as 36 minutes at 98.8% CPU with no log line and no error. So
anything that calls into a native library bounds its OWN work and reports failing to:
`OcrWorker` in `perception/ocr.py` runs the read in a process it can kill, and its
`DEFAULT_OCR_THREADS` is explicit because ONNX Runtime otherwise sizes its pool from the
core count and spins (the measured table is in that module's docstring). A failed
observation is recorded on the ledger by `SkillAPI.observe`, and `sandbox.py` then keeps
that run out of the skill's statistics entirely: a skill whose eyes broke has not failed
its task, and demoting it for that is a defect this project has shipped once already.

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

What the element detector knows is a measured claim, not an assumption. Its training
set is the sandbox app AND live public pages, so a full `scripts/build_ui_dataset.py`
run needs the network (`--no-web` opts out) and takes minutes, not seconds. That same
script's `--bench` scores any set of weights on pages held out of training entirely,
and `tests/perception/fixtures/README.md` is where the detector's numbers and its
remaining blind spots are written down. Quote that file rather than guessing, and
remeasure with `--bench` rather than assuming a retrain helped.

The cheapest answer this architecture can give is a WRONG one, so no efficiency number
may be computed without ground truth beside it. A stored skill that does the wrong
thing runs in seconds and costs nothing, which improves every speedup, saving and
call-count the project reports. Two rules follow, and both are already load-bearing:
`_measured` in `src/skillweaver/eval/metrics.py` is the ONLY door a run reaches a
timing or a saving through, and the failures it excludes are named in the report
rather than dropped; and retrieval RANKS while the planner DECIDES - a ranking has a
winner even when nothing fits, so closest is not runnable until it has an account of
the whole request (`MIN_ACCOUNTED_FOR` in `src/skillweaver/agent/planner.py`, which
carries the measurements it was calibrated from).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
