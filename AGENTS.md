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
state cannot be learned unless something can change it back. `WorldReset` in
`src/skillweaver/orchestrator.py` is that undo and takes two written forms:
`--reset-url`, one GET that restores the application, which the sandbox site's
`GET /__reset` is the instance of; and `--reset-steps`, an undo PERFORMED on the
screen, which is the only kind a real site offers - nothing on DoorDash empties a
cart but emptying it. `src/skillweaver/reset_actions.py` holds that second form and
the three properties it has to keep, all bought in failed runs: the gate resets
BETWEEN its three re-runs, so an undo that works four times in five does not make a
flaky demo, it makes the gate REJECT the skill. Read its docstring before adding a
knob to it - in particular, a negative exit condition passes on every screen that
lacks the marker INCLUDING the wrong one, and `ElementIndex.best` is a ranking with a
winner even when nothing fits, which is the same distinction `MIN_ACCOUNTED_FOR` draws
for the planner.

A reset may read the DOM (`via="dom"`), and that is not the agent breaking the
pixels-only premise: a real site names its controls where no camera can read them -
DoorDash's quick-add and header cart are icon-only buttons whose only name is an
`aria-label`, and a search of that page's visible text finds nothing - so the
scaffolding that puts the world back is handed `BrowserGroundTruth` explicitly, at the
one call site in `_dom_of`. Nothing else may. That site is also Cloudflare-gated
against automated browsers and this project does not evade it; `apps/sandbox-site/README.md`
explains why the local Pantry Lane cart is the world to build an ordering task against.

"Where the last run of this task ended" is NOT where this run should end: a task that
takes an argument ends somewhere the argument decides, so a recalled end screen can
say yes and must never say no. `TieredCritic` has three check roles for that reason -
evidence, corroboration, veto - and `_warm_critic` in `src/skillweaver/orchestrator.py`
holds the rule that picks one. A read-only live task needs no reset and should say so
(`read_only`, `ResetOutcome` in the same file); a `403` from a real site means the
endpoint is not a reset hook, not that the world could not be put back.

A statistic that SUMS when two records are merged must be handed a DELTA, never the
whole record. The site graph loaded a domain, observed a few traversals and handed the
store back everything it had - including the counts the store had just supplied - so
every save doubled them: ten traversals stored as 1023, and 262144 on the live
Wikipedia graph. Worse than the number is what it freezes, because new evidence is
then outweighed by a history that doubles: an edge measured at 5 successes and then
failing 200 runs straight was still priced and still preferred. `InMemorySiteGraph.unsaved`
and `subtract_transitions` in `graph/model.py` carry the rule and its inverse-of-merge
arithmetic. A test that observes into a graph it never LOADED cannot see any of this,
which is how it survived a full suite; `explain_edge` in `graph/route.py` is how a
preferred edge is asked which counts chose it.

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
absorb it - a measurement wants one page to be one screen. `BrowserController` now does
it for every run, not just for measurement scripts: `SOMETIMES_ONLY_OVERLAYS` in
`controllers/browser.py` aborts the appeal's own endpoints on the context, and carries
what it cost to find out that those patterns have to be REGEXES. Wikipedia serves the
appeal from `index.php?title=Special:BannerLoader&...`, so the name is in the query
string, and the `**/Special:BannerLoader*` glob this file used to recommend matches
nothing and aborts nothing. Force the appeal with `?banner=<name>&force=1` to check any
of this again without waiting for it.

Use **headless** against a REAL site, on both sides of anything that will be compared.
Headed and headless are different screens and are meant to be, and HOW different is the
page's business, not a constant: two fresh browsers of opposite modes score 0.13
(Wikipedia Main Page) and 0.42 (`json.html`) - below the same-state cut - but 0.82 on the
sandbox app, above it. So the crossing is explained where it bites rather than refused up
front, which would break the small clean page to protect the real one;
`skillweaver.render_mode` carries the measurements and the rule. The mode is a setting
(`SKILLWEAVER_HEADLESS`, `--headless`/`--headed` before the subcommand) and stays HEADED
by default because the demo is watched; every shipped command gets it through the one
`BrowserController` in `orchestrator._open_world`. What else a live-site run needs is in
`eval/wikipedia.yaml`.

A skill learned against a LIVE site can stop matching its own start screen within hours,
so do not assume a stored one still replays: a Wikipedia skill learned at 10:14 failed at
17:00 with `no_route`, its Charles Babbage start screen fingerprinting differently in BOTH
render modes and differently from each other. Re-learning fixed it in one step. The cause
was not established and the mode crossing above is NOT it - that was the first guess and
it was wrong, which is the thing worth knowing, because `no_route` against a real site
reads like a routing bug and is usually the page having moved. Budget a learn run before
measuring anything that needs a warm hit.

A real site can refuse an automated browser outright, and what it is reading is WHO
STARTED THE BROWSER - not the binary, the profile, the IP or the debugging channel, all
of which the three shipped launch modes can share. Measured against live doordash.com on
2026-09-19, same machine and minutes apart: framework-launched Chromium refused;
framework-launched real Chrome (`--chrome-profile <dir>`, `SKILLWEAVER_CHROME_PROFILE`)
refused, 0 of 6 loads and 0 of 5 on a new profile; a PLAINLY-launched real Chrome
attached over its debugging port loaded 6 of 6. So `--chrome-attach` (with
`--chrome-profile`, `SKILLWEAVER_CHROME_ATTACH`) starts Chrome as an ordinary process and
attaches, and `PLAINLY_LAUNCHED` in `controllers/chrome_launch.py` carries that table and
the mechanism: `navigator.webdriver` is true in both framework-launched modes and false
here because `--enable-automation`, which the framework adds and this mode does not, is
never passed. Nothing rewrites that flag, and the line is exact - not passing a flag of
our own is allowed, contradicting the browser with a spoofed fingerprint or user agent
never is. `REAL_CHROME_CHANNEL` in `controllers/browser.py` still carries what the
profile alone is measured to fix.

Three rules go with all of it, every one already paid for: give every run its OWN
directory, because a profile is exclusive and concurrent runs sharing one spoil the state
that made it worth having; nothing in this project defeats a human-verification page - no
masking argument, no spoofed fingerprint, no retry-until-it-passes - so a challenge FAILS
the run and a person clears it by hand, once, in that profile; and access is not a
property you can retest your way into, because hammering a site to find out whether it is
still letting you in is what stops it. Say what this mode is, too: it makes us stop
announcing ourselves, which is smaller than making a site accept us. Prove it against
something that does not gate you - `apps/sandbox-site`, or `example.com`.

Chrome 153 does NOT write `DevToolsActivePort`, in either render mode, so a launcher that
waits for that file waits out its whole timeout beside a perfectly healthy browser. The
address comes from Chrome's own `DevTools listening on ws://...` line on the stderr of
the process we started, and that line is an IDENTITY as well as an address: its per-browser
UUID is checked against `/json/version` before anything attaches, because attaching to a
Chrome this run did not start would mean driving somebody's real session.

Perception is dominated by OCR - 84-97% of every observation's time on real pages -
so the shipped fix is to not read the same pixels twice, and counts, not seconds, are how
that is judged (seconds move with machine load; this project has been burned by that
twice). `PerceptionCounters` and `CachingTextReader` in
`src/skillweaver/perception/ocr.py` hold the measured reasons the cache key is the exact
pixels and not "the same state", and `ComposedPerceiver` in `src/skillweaver/orchestrator.py`
holds the measured reason the text is NOT read lazily. Read both before trying either again.
Inside one read the cost is RECOGNITION and not detection - 87% against 8% - so the lever
is how many text lines go into one ONNX Runtime call (`DEFAULT_REC_BATCH`, one, worth
~1.5x), and resolution, cropping to the detector's boxes and a per-line cache are all
refuted THERE rather than being open questions; that module's "Recognition is the read"
carries the numbers and the refutations, and the browser captures at `scale=1.0`, not 2.0.
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

A skill must never sleep for time the browser has already spent, which was the
largest single line item in warm replay. Every action is SETTLED before the controller
returns - `_settle` in `controllers/browser.py` pauses and then waits for the page's
load event - and `_deliver` returns EARLY on an explicit wait, so `ctx.ctl.wait(2000)`
after a `press` settles nothing and sleeps two seconds on top of a wait that already
happened. Measured on live Wikipedia against the stored
`search_and_open_wikipedia_article`, whose three fixed sleeps were 38% of an
eight-second run: with-sleeps median 6.42s / 6.39s against 3.26s / 3.17s without,
48/48 successes in BOTH arms, reproduced across two independent interleaved sets (load
4.4-5.9 and 4.0-5.5); and under slow-3G, with every navigation settle exhausting its
full 3s budget, both arms still passed 6/6. So the sleeps are waste, not insurance.
`strip_reflex_waits` in `skills/refactor.py` removes them BEFORE the admission gate,
which then proves the sleepless skill by running it - the speedup is never bought by
weakening the gate - and `reflex_waits` there is the detector alone. Removing the
reflex is not removing the capability: a wait for what no load event covers, an
animation or a debounce, survives when an adjacent `ctx.log` NAMES that thing
(`_ANNOUNCES_WAIT`), and a rejected attempt is TOLD which sleeps went (`_repair_brief`
in `skills/synthesize.py`), because a model that is not told writes the same wait
again forever. Skills stored BEFORE this are untouched on purpose - re-hardening one
would ship a rewrite the gate never ran - so they keep their sleeps until relearned.

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

A skill is FILED under a domain and a lookup HAPPENS in one, so the project's whole
claim rests on the two agreeing - and the two commands the README tells a new user to
type do not name the same one. `learn` is given a `--url`, so what it stores is filed
under that host; the repeat has no reason to pass one, because the point is that the
agent already knows how. That silence is resolved by searching every domain and
letting a skill that accounts for the request name its own, plus the page its start
screen was last seen at, in `resolve_domain` (`orchestrator.py`). Resolution is never
looser than the planner's own gate and deliberately cannot reach the composer, so a
verbatim repeat whose argument was never quoted is judged on the skill's bare text -
`bind_args` binds only from a quoted slot, which is why such a repeat is warm for ONE
composer call rather than zero. Drive the acceptance path WITHOUT `--domain`: a test
that passes it to both commands cannot see any of this, which is how the namespaces
were free to disagree for months.

A trajectory step is one ACTION, not one decision. A move that ran a code block is
several steps, and the agent's stated reason and the critic's verdict both sit on its
LAST one, so anything reading a recording has to regroup before it can say what the run
meant: `moves_of` in `trajectory/render.py`, which is also where the run is turned into
the text the synthesizer is shown. A rejected move there fell short of what it CLAIMED -
measured on one whose action the task still needed - so it is marked as suspect and
never dropped.

A better RANKING buys nothing while the two checks after it count WORDS. This is a
MEASURED NEGATIVE RESULT, not an unfinished feature: retrieval CAN rank by meaning -
a local MiniLM through onnxruntime, no new dependency and no network at run time
(`skills/embed.py`, `make embedder`) - and it is OFF because it was measured and did
not pay. Do not "finish" it by turning it on.

What the measurement said, over both libraries this repository carries
(`scripts/bench_retrieval.py`, 48 requests, re-runnable in a minute):

  * recall improved, which is what an embedder is for: top-1 18/24 -> 21/24.
  * RUNNABLE candidates did not move AT ALL: 3/24 both ways when a person types the
    request, 9/24 both ways when a suite supplies its values. That is the number that
    decides whether the fast path fires, and it is the only one worth reading.
  * precision got WORSE, and this matters as much as the recall gain: irrelevant
    requests answered went 1/6 -> 5/6, because a cosine is almost never zero. No
    cut-off separates the two populations either, so there is nothing to tune - which
    is the argument against ever switching this on without re-measuring.

WHY it does not pay: a candidate that ranks first still has to bind (`bind_args`) and
then account for the request (`MIN_ACCOUNTED_FOR`), and BOTH of those count words. A
request worded differently fails them for exactly the reason it ranked badly, so a
better score arrives at a door locked in the same language. The word-counting gates,
not the similarity score, decide whether a stored skill can run.

WHAT WOULD HAVE TO CHANGE for it to be worth enabling: those two gates, not this
module. When binding can take an argument out of a sentence that does not quote it,
or when the content gate judges coverage by something other than word overlap, re-run
the bench and read `runnable`. Turning the model on before then buys three ranking
positions and four extra wrong answers. It is reachable ONLY through
`SKILLWEAVER_EMBEDDER` (`DEFAULT_EMBEDDER_ENABLED` is False): fetched weights sitting
on disk do not enable it and are not meant to.

And the case that prompted all of this was never a retrieval miss. The ordering
suite's `add_dish_with_option` log line says `no embedder`, but its stage is
`unaccounted` - the gate never sees a retrieval score at all, and a cosine of 1.00
would have changed the order of the candidates and nothing else.
`test_the_ordering_miss_of_2026_09_19_was_the_gate_and_not_the_ranking` holds the
arithmetic.

And a fall-through is not a success. A warm attempt that missed and was rescued by
exploration reports the miss in its headline (`RunReport.warm_missed` and `rescued`,
`orchestrator.py`); `SOLVED by the cold path` on its own is the sentence that hid the
defect above, because it is what a demo and a measurement both quote.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
