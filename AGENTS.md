# Project agent memory

Two rules the code cannot tell you, because this repository is built by many workers in parallel:

- **The shared surface is a coordination decision, not a local edit.**
  `src/skillweaver/contracts.py` and every package `__init__.py` under `src/skillweaver/` are imported by everyone.
  Do not change them as part of other work: report the change you need and let it be coordinated.
- **Every piece of work owns an exclusive file list.**
  Create and edit only the files your task names.
  If you need a change in a file you do not own, report it instead of editing it.

**Work here is proven by a real end-to-end run, not by tests.** There is no test tree:
`tests/` and the whole static-test setup were removed on 2026-09-20, along with pytest and
`make test`. Do not reinstate them. Run the thing you built against the real target and
report what it did - the task, the steps, the wall time, the outcome. Change a default path that already worked and
you owe it one real run too. Report failures plainly: "it worked" with no run behind it is
worse than "I could not get it to run, here is where it stopped". The linter still runs.

**Perception is pixels-only EXCEPT for browser use, on purpose and behind a switch.**
`--perception dom` (`SKILLWEAVER_PERCEPTION`) reads the page's own controls through
`perception/dom.py` and runs no detector and no OCR; `--policy jev`
(`SKILLWEAVER_POLICY`) then lets TypeSafe's Jev choose each move through `llm/jev_.py`.
Both default OFF and the pixel path is untouched. Actions stay POINT-based through the
one `BrowserController` - there is no second action plane - and what the DOM path adds to
that class is a read-only `evaluate` and, for `Back`, one `page.go_back`. Computer use is
still pixels-only and no part of this reaches a desktop target. `BrowserGroundTruth` remains what it was:
an offline teacher, reachable only from `_dom_of`, and NOT what the DOM perceiver uses.
The two paths keep SEPARATE skill libraries, namespaced by domain in `perception_mode.py` -
read that module before assuming a fingerprint would have caught the crossing, because it
does not: `parts` come from the screenshot and the URL, which both paths share.

Jev's action space has one operation this project added: `BACK`, the browser's own
history, a `Back` in `contracts.py` that only a controller with a session history
supports. It is OFFERED from `DomSnapshot.can_go_back`, which is the Navigation API's
`canGoBack` and not `history.length` - length counts entries in both directions and would
offer a back with nothing behind it - and WITHHELD again by `BACK_SIGNATURE` in
`agent/jev_driver.py` once a back is on this screen's dead-end list. That is a second
line rather than a row in `DEAD_END_OPERATIONS` because `BACK` names no element, so
`_exclusions` cannot reach it twice over, and the policy will otherwise re-pick it until
the run gives up at `BLOCKED`. A skill CANNOT replay a back and that is deliberate: `ActionSurface` has no
`back()` any more than it has `navigate()`, and the sandbox namespace holds no action
class, so a stored skill can never pop a history stack it did not build. Two measured
facts to save a re-derivation: the operation head picks `BACK` only when the goal names
returning (0.41-0.90 there, 0.00-0.04 when a forward route exists), so on an ordinary task
it changes no step counts and is not meant to; and the criterion's WORDING moves that mass
far more than anything else - naming "the browser's Back button" went from 0.01 to 0.85 on
the same screen.

The command line is the way to drive all of this: `uv run python -m skillweaver.cli --help`.
There is no `skillweaver` console script - `pyproject.toml` has no `[project.scripts]`, and
that file is shared surface. `src/skillweaver/orchestrator.py` holds the cold-versus-warm
decision and `build_agent`, the one place the real agent is wired; build on that rather than
assembling a planner and an explorer by hand.

There are no test doubles to iterate against either - `tests/fakes/` went with the test
tree - so a change is exercised by running the real command against the real site. Budget
for that: it needs `.env`, a browser and money, and it is the only evidence this project
accepts.

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
against automated browsers and this project does not evade it. The local demo site that
used to stand in for it (Northwind Console / Pantry Lane, `apps/sandbox-site`) was removed
on 2026-09-20; an ordering task is now built against a real shop that does let us in, which
is what splitkb and Walmart are doing in the entries below.

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
arithmetic. Observing into a graph that was never LOADED cannot show any of this -
which is how the bug survived for months, and why the only way to see it is a real run
that loads a domain and then saves it; `explain_edge` in `graph/route.py` is how a
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
against the sandbox looks green until it meets a website. The place that rule was still
being broken is the one where it is hardest to see: a FAILED move usually repaints the
page without changing it, so a dead end filed under an exact fingerprint is filed under a
screen the very next step is not standing on - measured at six of seven lost on one
33-action run. Do not re-check that with a synthetic no-op: an inert click re-observes
IDENTICALLY on the sandbox, live Wikipedia, splitkb and bbc.com, which is how this was
first measured and reported the wrong way round. `FailureMemory.near` in `agent/explorer.py` carries that measurement and is
the reader an acting policy uses; `at` and `seen` beside it stay exact because the default
explorer's prompt and its repeat guard are calibrated against them.

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

**The default browser is now the person's OWN running Chrome, driven the way
`jev_ultrafast` drives it** (`--browser harness`, `SKILLWEAVER_BROWSER`,
`controllers/harness.py`): Browser Harness holds one connection to the Chrome already
open, the run gets a background tab of its own, and every action is a raw DevTools call -
no Playwright, no fresh profile. `--browser playwright` is everything described above and
is what `--headless` or `--chrome-profile` still select (`browser_backend`, `config.py`),
because both describe a browser this project starts. Three things to know. It is the real,
LOGGED-IN browser, so a task addressed to it can act on real accounts. Chrome must have
remote debugging allowed once, by hand, at `chrome://inspect/#remote-debugging` -
`uv run browser-harness --doctor` says whether it is. And to exercise it WITHOUT touching
that browser, start a `ChromeProcess` and point the harness at it with `BU_CDP_URL` plus a
`BU_NAME` of your own, which is how it was proven: Wikipedia search learned cold in 4
actions / 7 model calls / 40s wall, stored, and replayed warm in 5 actions / 0 model calls
/ 4.5s, `--perception dom --policy jev`, 2026-09-20. A synthetic Cmd+A selects nothing on
macOS unless the key event carries `commands=["selectAll"]` (`_EDIT_COMMANDS`).

Three rules go with all of it, every one already paid for: give every run its OWN
directory, because a profile is exclusive and concurrent runs sharing one spoil the state
that made it worth having; nothing in this project defeats a human-verification page - no
masking argument, no spoofed fingerprint, no retry-until-it-passes - so a challenge FAILS
the run and a person clears it by hand, once, in that profile; and access is not a
property you can retest your way into, because hammering a site to find out whether it is
still letting you in is what stops it. Say what this mode is, too: it makes us stop
announcing ourselves, which is smaller than making a site accept us. Prove it against
something that does not gate you - `example.com`, or any small page of your own.

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

- It installs its own top-level `tests` package into the venv. That no longer shadows
  anything here, but it means `import tests` in any script silently resolves to
  ultralytics' package rather than failing.
- A relative `project=` given to `model.train()` resolves against ultralytics' GLOBAL
  `runs_dir` setting, which is per-user and can point at a different worktree entirely, so a
  training run silently writes into someone else's checkout. Pass an absolute path and read
  the checkpoint back from `model.trainer.best`; `scripts/train_detector.py` shows both.
- Importing it patches `PIL.Image.open`; `skillweaver.perception.detect_yolo._import_yolo`
  explains why that has to be undone and does it.

**The trained detector ships and works; there is no in-repo way to rebuild it.**
`data/models/ui_detector.pt` is committed and every run uses it as before. What was
removed on 2026-09-20 is its producer: `scripts/build_ui_dataset.py` drove the local demo
site to harvest the training frames, and its `--bench`, which scored any set of weights on
held-out pages, went with it - as did `tests/perception/fixtures/README.md`, where the
detector's numbers and its remaining blind spots were written down. So the detector's
accuracy is now an undocumented property of a binary, retraining starts with writing a new
harvester against real pages, and `scripts/train_detector.py` still trains but no longer
has a dataset to be pointed at. Do not quote detector numbers from memory; there is
nothing in the repository left to quote.

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

What the measurement said, over the two libraries this repository carried at the time
(`scripts/bench_retrieval.py`, 48 requests):

  * recall improved, which is what an embedder is for: top-1 18/24 -> 21/24.
  * RUNNABLE candidates did not move AT ALL: 3/24 both ways when a person types the
    request, 9/24 both ways when a suite supplies its values. That is the number that
    decides whether the fast path fires, and it is the only one worth reading.
  * precision got WORSE, and this matters as much as the recall gain: irrelevant
    requests answered went 1/6 -> 5/6, because a cosine is almost never zero. No
    cut-off separates the two populations either, so there is nothing to tune - which
    is the argument against ever switching this on without re-measuring.

Half that corpus is gone. The `console` library was a fixture library under the removed
test tree, so the bench now runs the Wikipedia half alone: 24 requests, and re-measured
on 2026-09-20 it says the same thing - recall 8/9 both ways, runnable 2/9 (typed) and
4/9 (suite-supplied) both ways, false hits 0/3 keywords against 2/3 embedder. The
conclusion holds and the NEAR-NEIGHBOUR case does not survive: `open_records` /
`open_record_detail` lived only in that fixture library, and that is the shape which
breaks retrieval in a library that has grown. Re-establish it from a real run before
quoting this bench about a grown library.

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

And the case that prompted all of this was never a retrieval miss. The 2026-09-19
ordering run's `add_dish_with_option` log line says `no embedder`, but its stage is
`unaccounted` - the gate never sees a retrieval score at all, and a cosine of 1.00
would have changed the order of the candidates and nothing else. The arithmetic used to
be pinned by a test; read `MIN_ACCOUNTED_FOR` in `agent/planner.py` for it now.

Reuse is also keyed on what a skill DOES. The admission gate stores an action signature
(`TYPE_TEXT(text_field) -> CLICK(button) -> ...`, labels and values abstracted) and the
sentence-plus-arguments it proved, a `Precedent`, and ONLY for a skill that carries a
verifier; `skills/family.py` holds the signature, the family cut (`MAX_FAMILY_DISTANCE`,
with the twelve-signature table showing the two populations TOUCH at 0.50, so re-measure
with `scripts/measure_families.py`, do not nudge) and `same_intent`. Three things there
were each paid for in a live run. Test a gate against a skill the gate actually STORED: a
synthesized parameter often carries a schema `default`, which binds `{}` without reading
the sentence, so a check that lives in the text binder never runs - *Remove "X" from the
cart* accounted for 0.86 of itself on the add skill and was cleared to run until intent
became a gate every candidate passes (`asks_for`, `agent/planner.py`). The critic's
PER-MOVE verdict is wrong on pages that answer without repainting (a filled field, an
AJAX add-to-cart), so nothing may be keyed on it: borrowed roles and skeleton-following
both were, and both broke (`derive_signature`, `Explorer._follow`). And the cold-run
skeleton is followed 4 of 4 steps on splitkb yet bought NOTHING there - 7 actions and 11
calls with it and without, n=3 each - so do not quote it as a saving until it is measured
where the waste was (Walmart). An older checkout that touches a newer library silently
strips `action_signature` and `precedents` from `meta.json`; give each code version its
own data directory when comparing them.

And a fall-through is not a success. A warm attempt that missed and was rescued by
exploration reports the miss in its headline (`RunReport.warm_missed` and `rescued`,
`orchestrator.py`); `SOLVED by the cold path` on its own is the sentence that hid the
defect above, because it is what a demo and a measurement both quote.

A skill is recorded at MODEL speed and replayed at CODE speed, one to two orders of
magnitude faster, so a control the page answers WITHOUT navigating is a race the recording
never sees. `_settle` in `controllers/browser.py` waits for the load event, which such a
control fired long ago: measured on live splitkb.com, the click on "Add to cart" returns in
~130ms with the document complete and the old URL showing, while the cart commits at ~900ms
and the redirect lands at ~1050ms. `ComposedPerceiver.observe` CAPTURES FIRST and reads
after, so the frame a post-click `ctx.see` judges is the one taken while the click was still
being answered - waiting longer for the index cannot help, only re-capturing can. The wait
is therefore per-control and conditional: `ctx.wait_for_text` (`AWAIT_BUDGET_MS` in
`skills/api.py`) is `find_text` allowed to look again, it returns the instant the text is
there, and on a page that already answered its first look IS the observation the skill was
about to make - so it costs the fast path nothing. Wait for a THING, not for a TIME; that is
the whole difference from the `ctx.ctl.wait(ms)` reflex that `strip_reflex_waits` removes,
and the two passes sit next to each other in `skills/refactor.py` on purpose. It matches
strictly, never fuzzily, because a fuzzy `find_text("Your cart")` on splitkb's PRODUCT page
answers with "Add tocart" - a wait satisfied by the screen it was meant to outlast. The
hardening pass writes the rewrite in wherever a read is followed by a `ctx.expect` on it
or has its result handed to `ctx.ctl` (`awaited_reads`, `_acted_on_read`), which are the
shapes that prove the text is REQUIRED rather than merely asked about; a read that is only branched on is left alone, or every run pays the
budget for something it hoped was absent. `SETTLE_BUDGET_MS` in `reset_actions.py` is the
same lesson for a converging undo, which must wait for the screen to change AND go quiet
because such a page answers in two frames. Do NOT fix any of this by widening `_settle`: a
quiet WINDOW is the only signal that works, a zero-window check reads quiet before the
request has even started (measured: 25-45ms on both Wikipedia and splitkb), and a window
taxes every action on every site.

And a verifier that passes on the wrong screen is worse than none, because it is what turns
a skill that did nothing into a stored one, and then into a STATISTIC. A synthesized
verifier matching `"Keycaps"` - the word the top nav says on EVERY page of that shop -
passed 8 replays whose carts were empty all 8 times by `/cart.js` (`item_count=0`), and
`SkillStats` for it read **14 runs, 14 successes**. Nothing in the library can notice this:
`record_run` is told `ok` by the sandbox, and the sandbox is told `ok` by the verifier. The
warm CRITIC caught it every time and the report said `warm miss`, which is the system
working exactly as `RunReport.warm_missed` intends - so read the critic's verdict and the
site, never `SkillStats` and never the skill's own say-so, when you are asking whether a
replay did the job.

The gate now REFUSES that verifier rather than leaving the critic to catch it, and it does
so by the same move the gate already makes for the code: re-run it where the skill STARTED.
`_StartScreen` in `skills/synthesize.py` dresses the observation the precondition was
measured against as a frozen controller and perceiver, so the verifier is asked about the
start screen with no capture, no OCR and no second browser - and one that says yes there
stops at stage `discrimination`, because a check that was true before and after proves
nothing. A probe that cannot RUN admits, deliberately: a false accept still faces the
critic, a false reject destroys a correct skill. Measured on live splitkb.com against
`/cart.js`, empty cart -> filled cart: the verifier synthesized after this said False then
True, `find_text("Keycaps")` said True both times. Telling the model only "rejected" buys a
longer verifier over the same chrome words, so `_A_VERIFIER_MUST_BE_ABLE_TO_FAIL` and the
`verifier_code` section of `agent/prompts/synthesize.md` NAME the signals instead - a count
that moved, a row carrying the parameter's own value, a changed URL - and say that `verify`
sees only `ctx` and `result`, so a parameter reaches it by being RETURNED from `run`.

On live walmart.com a page is not the page when its load event fires, and three layers
read it then, measured 2026-09-19 through the DOM perceiver. (1) THE GATE, fixed:
`NavigatingEnvironment` resets the world and then does its OWN `Navigate(url)`, and
`_attempt` observed at once, so one recorded `/cart` start screen read 0.40, 1.00, 0.238,
0.238, 1.00, 0.40 against the 0.26 cut, and the end screen - reached as the skill reaches
it - read 0.011 in 4 of 4, still the search page. `_observe_at_rest` in
`skills/synthesize.py` waits for the fingerprint to hold EQUAL and is never told the
score; at rest those reads are 1.000 (6 of 6) and 0.901 (4 of 4), and the first Walmart
skill was stored with the cut and the critic untouched. Nothing a `--reset-steps` recipe
does at its tail can settle that screen, which cost one wrong fix here. (2) THE RECORDING,
not fixed: a run that goes on past its goal - Jev re-typed the product on the cart page at
confidence 0.30 before DONE - records an end screen no correct skill reaches (0.064 to
the cart it had just opened), and the admission critic demands that screen. (3) THE SKILL,
fixed: a result's TITLE paints before its Add button hydrates (0.61s later, measured), the
stored skill waited for the title and then looked for the button ONCE, and its
`ctx.see.best` fallback - a ranking with a winner when nothing fits - handed it the header
cart button, so it passed one gate attempt and then 0 of 3 replays with Walmart's cart
empty. `_acted_on_read` in `skills/refactor.py` now awaits a read whose result is PRESSED;
the same stored skill with that one call rewritten filled the cart. The assumption that
died is "one action is answered once" - a page answers in phases. (4) THE WARM CRITIC,
not fixed, and why a Walmart replay that WORKS is reported as a failure: the planner reads
the end screen the instant a code-speed skill returns, so six consecutive
`--library-only --no-learn` replays that each put exactly the right item in Walmart's cart
(3.2-4.4s of skill time, confirmed on the site) were all reported `ok=False,
stage=rejected`, each after ONE escalated model call (~$0.055) whose AFTER screenshot was
"empty with a loading spinner". `_observe_at_rest` is the fix's shape; it lives in the gate
only. So on this site the critic's verdict and the site DISAGREE in the safe direction, and
`SkillStats` (6/6) happens to be right - do not learn from that to trust it. Measure a screen by the
ROUTE and the MOMENT it is read at - `/cart` reloaded from `/cart` after 4s says 1.000 and
tells you nothing - and start a Walmart task at `/cart`, the only anchor that reaches
1.000. This machine also
geolocates to Canada, so walmart.com pins `fulfillment_method:Shipping` and anything
store-fulfilled (all of Great Value) returns "We couldn't find a match".

A run's model client is built with `computer_use=True`, which appends the computer tool to
EVERY request it makes, helpers included. A helper that wants text back can get a
`screenshot` tool call and no text instead: 1 reply in 12 for the Jev text writer, which
ended three cold runs at step 0 and read exactly like a truncated reply until the stop
reason was printed. `_TEXT_ASKS` in `llm/jev_.py` carries the fix and the measurement. The
critic's "(empty reply)" degradations are the same shape and are NOT fixed.

`skillweaver inspect` is a LIVE control surface, not `dashboard build`'s static report:
a loopback page that drives the real agent one move at a time (`src/skillweaver/inspector/`).
It is `Explorer`'s own loop split at its seam - `_ask`/`_ground`/`_refuse_repeat` is
"choose", `_make_move` is "execute" - so it calls PRIVATE explorer methods on purpose, from
`LiveSession._decide` and `_perform` only; rename one and `_check_explorer_seam` says so at
startup. Three facts each cost a live run. ONE thread owns the browser (Playwright's sync
driver is thread-affine and the HTTP server is threaded), so handlers post jobs to `_Worker`
and polls read a PUBLISHED snapshot. Anything that wraps a controller must pass through what
it does not record: `DomPerceiver` reads `controller.evaluate`, which is outside the
`Controller` protocol, and a wrapper without it killed every typed move at its second line.
And a stepped loop has no `return`, so every ending must call `_conclude` or the `Recorder`
refuses the next run. Binding `127.0.0.1:<port>` does NOT fail when another process holds
`*:<port>` (measured on macOS) - open the printed `127.0.0.1` URL, never `localhost`.
Every `/api` request needs the startup token, the frame included: it is a photograph of a
browser that may be logged in. Per-target odds are absent because `PolicyDecision` carries
only the chosen target's; wire them in `_decision_json` when it carries more.

`skillweaver search` is full-text search over the RECORDS - runs, steps with the text
that was on screen, graph nodes and edges, stored skill code - through Elasticsearch, and
it is observability, not retrieval: nothing on a run's path reaches it, and it does not
rank skills, because a better ranking was measured not to pay (above). The agent side is
read-only by construction, not by promise - `scripts/index_elastic.py` is the only writer
and nothing under `src/` calls a writing method; `dashboard/elastic.py` carries the five
indices, the `code` analyzer (the standard one keeps `ctx.see.find_text` as ONE token, so
`find_text` matched nothing across seven skills that all call it) and the `field:value`
syntax that asks about a count or a flag. `SKILLWEAVER_ELASTIC_URL` unset means the
subcommand says so and nothing else changes. Proven 2026-09-20 against a local
single-node 8.15: 41 runs / 777 steps / 136 states / 429 edges / 7 skills indexed in 4s,
queries 40-200ms.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
