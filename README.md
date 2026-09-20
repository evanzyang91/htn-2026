# Jevis

A self-growing skill library for computer-use agents (Hack the North 2026).

A computer-use agent does the same task the same slow way every time: look at the
screen, ask a model what to do, click, look again. Jevis does that once. It keeps what
worked as a reusable Python skill, and the next run of that task replays the skill with
no model in the action loop.

This is Voyager's "write your own skills as code" idea, pointed at a general
computer-use agent instead of a game. The agent sees through screenshots (a detector
for elements, OCR for text), and it keeps a graph per site of the screens it has met and
the actions that move between them. Browser first, with a real desktop controller
alongside.

## The two paths

**Cold.** `learn` explores a task by trial and error with a computer-use model (Claude
primary, Gemini alongside). When it reaches the goal it writes what worked as a skill,
then proves the skill by replaying it from the starting screen. A replay needs that
screen back, so a task that changes anything must say how to undo it, or nothing is
stored. A task that only reads says so with `-p read_only=true`.

**Warm.** `run` retrieves the stored skill and does the job directly. The site graph
routes over edges that have succeeded before, so an edge listed at 0/3 is one the agent
will not trust.

## Demo version

The `demo` branch holds the live browser agent: type a task in plain words, watch a real
Chrome window do it, and read what the run cost against what a frontier computer-use
model would have charged for the same work.

## Install

Requires [uv](https://docs.astral.sh/uv/) on macOS (Apple silicon is what is tested).
uv fetches Python 3.12 itself.

```sh
make install     # uv sync + playwright install chromium
```

Settings come from the environment or `.env`; `src/skillweaver/config.py` lists every
variable. An Anthropic API key is needed: every run drives a real model against a real
site.

## Use

The command line is the way to drive all of this. There is no console script, so it is
reached as a module:

```sh
uv run python -m skillweaver.cli --help
```

`--data-dir`, `--log-level`, `--headless` and `--chrome-profile` describe the whole
invocation, so they go **before** the subcommand and cover `learn`, `run` and `eval run`
alike. The browser is headed by default, because a browser you can watch is what makes
the agent legible.

```sh
# Teach it once. A task that changes something says how to undo it.
uv run python -m skillweaver.cli learn "empty the cart" \
    --url https://food.test/ --reset-steps undo/empty-cart.json

# Then do it from memory.
uv run python -m skillweaver.cli run "empty the cart" --url https://food.test/

# What does it remember?
uv run python -m skillweaver.cli skills ls
uv run python -m skillweaver.cli graph show food.test
uv run python -m skillweaver.cli dashboard build

# Cold against warm, over a suite.
uv run python -m skillweaver.cli --headless eval run --suite eval/wikipedia.yaml --repeat 3
```

## Perception and policy

Perception is pixels only, except for browser use, on purpose and behind a switch.
`--perception dom` reads the page's own controls and runs no detector and no OCR;
`--policy jev` then lets TypeSafe's Jev choose each move. Both default off, the pixel
path is untouched, and the two paths keep separate skill libraries. Actions stay
point-based through the one browser controller: there is no second action plane.

## Lint

```sh
make lint        # ruff check + format check
make fmt         # auto-fix
```

There is no test tree, and reinstating one is not wanted. Work here is proven by a real
end-to-end run: the task, the steps, the wall time, the outcome. "It worked" with no run
behind it is worth less than "I could not get it to run, here is where it stopped".

## Layout

- `src/skillweaver/contracts.py` - every shared type and Protocol. Start here.
- `src/skillweaver/orchestrator.py` - the cold-versus-warm decision, and the one place
  the real agent is wired. Build on this rather than assembling a planner by hand.
- `src/skillweaver/<package>/` - implementations, one package per concern.
- `eval/wikipedia.yaml` - the live-site evaluation suite.
- `AGENTS.md` - the rules the code cannot tell you. Read it before editing shared files.
