"""Checks for what confirming a ``DONE`` claim costs, and that it still refuses a wrong one.

Pure logic by default - no browser, no model, no network:

    uv run python scripts/check_h_done.py

``--judge <data-dir>`` additionally asks the REAL judge about screens a recorded run left
behind (``<data-dir>/trajectories``; it needs ``.env`` and costs a few cents a call), and
is the honest negative: the same goal against the run's screen BEFORE anything was added,
and the run's real end screen against a goal naming a product that is not in the cart.
Both must be refused. ``--n`` repeats the timed positive ask per arm, interleaved, with
the computer tool on the request (what the run's client sends) and without it (what the
critic sends now).

    uv run python scripts/check_h_done.py --judge <data-dir> --n 3

``--jev-pages`` asks the REAL Jev judge (``JevPolicy.judge_done``, DOM text, no screenshot)
about live pages in a PRIVATE headless Chrome, and prints every distribution - which is
what ``MIN_SATISFIED`` is calibrated from. It adds one item to a cart it then abandons
with the browser profile: splitkb's product page BEFORE the add (must be refused), the
cart AFTER it (must be accepted), the same cart text with the product name swapped (must
be refused), and a Wikipedia article against the right and the wrong title.

    uv run python scripts/check_h_done.py --jev-pages

What none of this shows is a whole cold run's clock; those numbers are in the commit.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from skillweaver.agent.critic import REASK_MAX_TOKENS, TieredCritic, text_only
from skillweaver.agent.critic_final import MIN_SATISFIED, JevDoneCritic, quoted_arguments
from skillweaver.agent.explorer import Explorer, Move
from skillweaver.contracts import (
    Fingerprint,
    LLMResponse,
    Observation,
    Screenshot,
    ToolCall,
    Usage,
    Wait,
)
from skillweaver.errors import ProviderError
from skillweaver.llm.usage import UsageMeter

_failed: list[str] = []

YES = (
    '{"ok": true, "evidence": "the cart lists the keycaps with quantity 1", '
    '"reason": "the product is in the cart", "confidence": 0.9}'
)
NO = (
    '{"ok": false, "evidence": "the cart is empty", '
    '"reason": "nothing was added", "confidence": 0.9}'
)
TOOL_CALL = LLMResponse(
    text="", tool_calls=(ToolCall(name="computer", args={}, id="t1"),), stop_reason="tool_use"
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail else ''}")
    if not ok:
        _failed.append(name)


class _Scripted:
    """An LLMClient that answers from a script and remembers how it was asked."""

    def __init__(self, *replies: LLMResponse | str) -> None:
        self._replies = list(replies)
        self.asked: list[int] = []

    def complete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any) -> Any:
        del messages, system, tools
        self.asked.append(int(kw["max_tokens"]))
        reply = self._replies.pop(0)
        return reply if isinstance(reply, LLMResponse) else LLMResponse(text=reply)

    def total_usage(self) -> Usage:
        return Usage()

    def name(self) -> str:
        return "scripted"


def _obs(value: str) -> Observation:
    now = datetime.now(UTC)
    return Observation(
        screenshot=Screenshot(png=b"", width=1, height=1, scale=1.0, captured_at=now),
        elements=(),
        index=SimpleNamespace(),  # type: ignore[arg-type]
        fingerprint=Fingerprint(value, {"part": value}),
        url="https://shop.example/",
        taken_at=now,
    )


def the_reask() -> None:
    before, after = _obs("home"), _obs("cart")

    llm = _Scripted(TOOL_CALL, YES)
    verdict = TieredCritic(llm).judge("add it", before, after)
    check("an empty tool-call reply is asked for once more", len(llm.asked) == 2, str(llm.asked))
    check("and the second answer is the verdict", verdict.ok and verdict.policy == "model")
    check(
        "the re-ask is given room to think",
        llm.asked == [512, REASK_MAX_TOKENS],
        str(llm.asked),
    )

    llm = _Scripted(TOOL_CALL, TOOL_CALL, YES)
    verdict = TieredCritic(llm).judge("add it", before, after)
    check("ONCE: two bad replies degrade and stop", len(llm.asked) == 2 and not verdict.ok)
    check(
        "degraded is still 'I do not know'",
        verdict.confidence == 0.0 and verdict.policy == "model-unparseable",
        verdict.policy,
    )
    check("and names why the reply was empty", "stop_reason=tool_use" in verdict.reason)

    llm = _Scripted(NO, YES)
    verdict = TieredCritic(llm).judge("add it", before, after)
    check("a NO is a verdict: never re-asked into a yes", len(llm.asked) == 1 and not verdict.ok)

    llm = _Scripted('{"ok": true, "evidence": "yes", "reason": "done"}', NO)
    verdict = TieredCritic(llm).judge("add it", before, after)
    check(
        "an unevidenced yes is re-asked, and the re-ask may refuse",
        len(llm.asked) == 2 and not verdict.ok and verdict.policy == "model",
        verdict.policy,
    )

    llm = _Scripted(YES)
    verdict = TieredCritic(llm).judge("add it", before, before)
    check("an unchanged screen is still refused for free", not llm.asked and not verdict.ok)


class _ToolClient:
    """The shape of ``AnthropicClient`` that matters here: a flag, a meter, a transport."""

    def __init__(self) -> None:
        self._computer_use = True
        self._meter = UsageMeter()
        self._client = object()


def the_tool_free_judge() -> None:
    run_client = _ToolClient()
    judge = text_only(run_client)  # type: ignore[arg-type]
    check("the judge's client sends no computer tool", judge._computer_use is False)  # type: ignore[attr-defined]  # noqa: SLF001
    check("the run's own client still does", run_client._computer_use is True)  # noqa: SLF001
    check(
        "they share ONE meter, so the judge is charged to the run",
        judge._meter is run_client._meter and judge._client is run_client._client,  # type: ignore[attr-defined]  # noqa: SLF001
    )
    check("text_only is idempotent", text_only(judge) is judge)
    plain = _Scripted()
    check("a client with no such flag is left alone", text_only(plain) is plain)  # type: ignore[arg-type]
    critic = TieredCritic(run_client)  # type: ignore[arg-type]
    check("TieredCritic judges through the tool-free one", critic._llm._computer_use is False)  # type: ignore[union-attr]  # noqa: SLF001
    check("and so does a per-step clone", critic.expecting()._llm is critic._llm)  # noqa: SLF001


class _Policy:
    def propose(self, *args: Any, **kw: Any) -> str:
        raise AssertionError("not asked here")

    def name(self) -> str:
        return "policy"


def _wait(done: bool = False) -> Move:
    return Move("t", "the page settles", done, Wait(150), None, "wait 150ms", "wait:150")


def the_wait_move() -> None:
    llm = _Scripted(YES, YES, YES)
    run = SimpleNamespace(usage_mark=Usage(), spend=SimpleNamespace(add_usage=lambda usage: None))
    driven = Explorer(llm, SimpleNamespace(), policy=_Policy())  # type: ignore[arg-type]
    before, after = _obs("product"), _obs("cart")

    verdict = driven._judge("x", before, after, _wait(), run, per_move=True)  # noqa: SLF001
    check(
        "a policy's wait over a changed screen is ok with NO model call",
        verdict.ok and not llm.asked and not verdict.escalated,  # type: ignore[attr-defined]
    )
    verdict = driven._judge("x", before, before, _wait(), run, per_move=True)  # noqa: SLF001
    check(
        "over an unchanged screen it fails, for free, as before", not verdict.ok and not llm.asked
    )

    driven._judge("the task", before, after, _wait(done=True), run)  # noqa: SLF001
    check("the DONE claim still goes to the full critic", len(llm.asked) == 1, str(llm.asked))

    default = Explorer(llm, SimpleNamespace())  # type: ignore[arg-type]
    default._judge("x", before, after, _wait(), run, per_move=True)  # noqa: SLF001
    check("the default explorer's wait is judged as it always was", len(llm.asked) == 2)


# --------------------------------------------------------------------------------------
# the Jev done critic, offline: a scripted judge and hand-built snapshots
# --------------------------------------------------------------------------------------

GOAL = 'Add the "LPF Glow Keycaps" to the cart'


def _snap(url: str, text: str, title: str = "") -> Any:
    return SimpleNamespace(url=url, title=title, text=text, digest=f"{url}|{text}", controls=())


class _Eyes:
    def __init__(self) -> None:
        self.last: Any = None


class _Judge:
    """``JevPolicy.judge_done`` with the answer written in advance."""

    def __init__(self, choice: str, p: float, *, fails: bool = False) -> None:
        self.choice, self.p, self.fails = choice, p, fails
        self.asked: list[dict[str, Any]] = []

    def judge_done(self, goal: str, end: Any, **kw: Any) -> Any:
        self.asked.append({"goal": goal, "end": end, **kw})
        if self.fails:
            raise ProviderError("HTTP 503")
        rest = (1.0 - self.p) / 2
        odds = {"satisfied": rest, "not_satisfied": rest, "cannot_tell": rest, self.choice: self.p}
        return SimpleNamespace(choice=self.choice, probability=self.p, probabilities=odds, ms=300.0)


class _NeverAsked:
    def complete(self, *args: Any, **kw: Any) -> Any:
        raise AssertionError("Claude was asked to verify done-ness on the Jev path")

    def total_usage(self) -> Usage:
        return Usage()

    def name(self) -> str:
        return "claude"


def _claim(judge: _Judge, start: Any, end: Any, **kw: Any) -> Any:
    eyes = _Eyes()
    critic = JevDoneCritic(judge, eyes, **kw)
    first, last = _obs("home"), _obs("cart")
    eyes.last = start
    critic.open_run(first)
    critic.brief_final(("1. click [a] button 'Add to cart'",))
    eyes.last = end
    return critic.judge(GOAL, first, last)


def the_jev_done_critic() -> None:
    home = _snap("https://shop.example/", "Welcome. Keyboards. Keycaps.", "Shop")
    cart = _snap("https://shop.example/cart", "Your cart. LPF Glow Keycaps x1. Subtotal", "Cart")

    judge = _Judge("satisfied", 0.93)
    verdict = _claim(judge, home, cart)
    check("satisfied above the floor is the only yes", verdict.ok, verdict.policy)
    check("the verdict says WHO decided", "Jev judged" in verdict.reason, verdict.reason[:90])
    check("and what became of the quoted argument", "new on the end page" in verdict.reason)
    asked = judge.asked[0]
    check(
        "the judge is shown the start page and the literal actions",
        asked["start_url"] == home.url and asked["start_title"] == "Shop" and asked["actions"],
    )

    for choice, p, policy in (
        ("satisfied", MIN_SATISFIED - 0.05, "jev-weak"),
        ("not_satisfied", 0.97, "jev-not-satisfied"),
        ("cannot_tell", 0.6, "jev-cannot-tell"),
    ):
        verdict = _claim(_Judge(choice, p), home, cart)
        check(f"{choice} p={p:.2f} REFUSES the claim", not verdict.ok, verdict.policy)
        check(f"  ...as {policy}", verdict.policy == policy)

    verdict = _claim(_Judge("satisfied", 0.99, fails=True), home, cart)
    check("a provider error refuses; nothing falls back", not verdict.ok, verdict.policy)

    judge = _Judge("satisfied", 0.99)
    verdict = _claim(judge, home, _snap(home.url, home.text, "Shop"))
    check(
        "a page literally unchanged since the start is refused with NO call",
        not verdict.ok and not judge.asked and verdict.policy == "literal-unchanged",
    )

    # The warm path: TieredCritic's roles with no model behind them decide first.
    judge = _Judge("not_satisfied", 0.99)
    free = TieredCritic(None, corroborating_state=_obs("cart").fingerprint)
    verdict = _claim(judge, home, cart, free=free)
    check("a corroborated warm end screen is a free yes", verdict.ok and not judge.asked)
    judge = _Judge("satisfied", 0.95)
    free = TieredCritic(None, corroborating_state=_obs("elsewhere").fingerprint)
    verdict = _claim(judge, home, cart, free=free)
    check("a missed corroboration goes to Jev", verdict.ok and len(judge.asked) == 1)
    free = TieredCritic(None, expected_state=_obs("elsewhere").fingerprint)
    verdict = _claim(judge, home, cart, free=free)
    check("an unverified skill's recalled screen keeps its veto", not verdict.ok, verdict.policy)

    # At rest: only when the instant read did not earn a free yes.
    eyes, reads = _Eyes(), []

    def observe(controller: Any) -> Observation:
        reads.append(controller)
        eyes.last = cart
        return _obs("cart")

    rest = (SimpleNamespace(observe=observe), SimpleNamespace(quiesce=lambda quiet, cap: 10.0))
    judge = _Judge("not_satisfied", 0.99)
    free = TieredCritic(None, corroborating_state=_obs("cart").fingerprint)
    critic = JevDoneCritic(judge, eyes, free=free, rest=rest)  # type: ignore[arg-type]
    verdict = critic.judge(GOAL, _obs("home"), _obs("cart"))
    check("an instant free yes costs no rest and no model", verdict.ok and not reads)
    verdict = critic.judge(GOAL, _obs("home"), _obs("spinner"))
    check(
        "a spinner is re-read AT REST and then corroborates, still with no model",
        verdict.ok and len(reads) == 1 and not judge.asked,
        verdict.policy,
    )

    check("quoted arguments are read verbatim", quoted_arguments(GOAL) == ("LPF Glow Keycaps",))

    # And the explorer hands the claim to it, with Claude wired to explode if touched.
    eyes = _Eyes()
    judge = _Judge("satisfied", 0.9)
    critic = JevDoneCritic(judge, eyes)
    explorer = Explorer(_NeverAsked(), SimpleNamespace(), policy=_Policy(), critic=critic)  # type: ignore[arg-type]
    run = SimpleNamespace(usage_mark=Usage(), spend=SimpleNamespace(add_usage=lambda usage: None))
    eyes.last = cart
    verdict = explorer._judge(GOAL, _obs("home"), _obs("cart"), _wait(done=True), run)  # noqa: SLF001
    check("the explorer's done claim reaches Jev, never Claude", verdict.ok and judge.asked)


# --------------------------------------------------------------------------------------
# --judge: the real model, on screens a real run recorded
# --------------------------------------------------------------------------------------


class _AsBuilt:
    """The run's client exactly as ``orchestrator`` builds it, hidden from ``text_only``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def complete(self, *args: Any, **kw: Any) -> Any:
        return self._inner.complete(*args, **kw)

    def total_usage(self) -> Usage:
        return self._inner.total_usage()

    def name(self) -> str:
        return self._inner.name()


def _timed(critic: TieredCritic, llm: Any, goal: str, before: Any, after: Any) -> tuple[Any, str]:
    mark, began = llm.total_usage(), time.perf_counter()
    verdict = critic.judge(goal, before, after)
    seconds = time.perf_counter() - began
    spent = llm.total_usage()
    return verdict, (
        f"{seconds:5.2f}s calls={spent.calls - mark.calls} "
        f"in={spent.input_tokens - mark.input_tokens} "
        f"out={spent.output_tokens - mark.output_tokens} policy={verdict.policy}"
    )


def the_real_judge(data_dir: Path, n: int) -> None:
    from skillweaver.llm.anthropic_ import AnthropicClient
    from skillweaver.trajectory.store import TrajectoryFileStore

    store = TrajectoryFileStore(data_dir / "trajectories")
    summary = next(s for s in store.summaries() if s.ok)
    run = store.load(summary.run_id)
    first, end = run.steps[0].before, run.steps[-1].after
    unadded = next(s for s in run.steps if "/products/" in (s.before.url or "")).before
    print(f"run {run.run_id}: {run.task!r}")
    print(f"  first={first.url}\n  unadded={unadded.url}\n  end={end.url}")

    llm = AnthropicClient(computer_use=True)
    with_tool, without = TieredCritic(_AsBuilt(llm)), TieredCritic(llm)  # type: ignore[arg-type]
    for round_ in range(n):
        for label, critic in (("tool on request ", with_tool), ("tool-free (now) ", without)):
            verdict, line = _timed(critic, llm, run.task, first, end)
            print(f"  positive #{round_} {label} {line} ok={verdict.ok}")
            check(f"the real end screen is accepted [{label.strip()} #{round_}]", verdict.ok)

    verdict, line = _timed(without, llm, run.task, first, unadded)
    print(f"  negative A (product page, nothing added) {line}\n    {verdict.reason[:200]}")
    check("a DONE claimed BEFORE the add is refused", not verdict.ok)

    wrong = 'Add the "Aurora Sweep" keyboard kit to the cart'
    verdict, line = _timed(without, llm, wrong, first, end)
    print(f"  negative B (right screen, wrong product) {line}\n    {verdict.reason[:200]}")
    check("a cart holding a DIFFERENT product is refused", not verdict.ok)


# --------------------------------------------------------------------------------------
# --jev-pages: the real Jev judge on live pages, for calibration
# --------------------------------------------------------------------------------------

PRODUCT = "LPF Glow Legended Low Profile MX Keycaps"
PRODUCT_URL = "https://splitkb.com/products/lpf-glow-legended-mx-keycaps"


def _ask_jev(policy: Any, label: str, goal: str, end: Any, want: bool, **kw: Any) -> None:
    answer = policy.judge_done(goal, end, **kw)
    odds = " ".join(f"{k}={v:.3f}" for k, v in sorted(answer.probabilities.items()))
    accepted = answer.choice == "satisfied" and answer.probability >= MIN_SATISFIED
    print(f"  {label}\n    {answer.choice} p={answer.probability:.3f} {answer.ms:.0f}ms [{odds}]")
    check(f"{label}: {'accepted' if want else 'refused'}", accepted is want, answer.choice)


def the_live_pages() -> None:
    import dataclasses
    import os
    import subprocess
    import tempfile

    from skillweaver.config import settings
    from skillweaver.contracts import Click, Navigate
    from skillweaver.controllers.chrome_launch import ChromeProcess
    from skillweaver.llm.jev_ import JevPolicy
    from skillweaver.perception.dom import DomPerceiver

    policy = JevPolicy(None, api_key=settings().typesafe_api_key)  # type: ignore[arg-type]
    goal = f'Add the "{PRODUCT}" to the cart'
    with ChromeProcess(user_data_dir=tempfile.mkdtemp(prefix="check-h-"), headless=True) as chrome:
        os.environ["BU_CDP_URL"] = chrome.endpoint
        os.environ["BU_NAME"] = f"checkh{os.getpid()}"
        from skillweaver.controllers.harness import HarnessBrowserController

        try:
            with HarnessBrowserController(start_url=PRODUCT_URL) as ctl:
                eyes = DomPerceiver()

                def look() -> Any:
                    ctl.quiesce(500, 4000)
                    eyes.observe(ctl)
                    return eyes.last

                page = look()
                print(f"splitkb product page: {page.url} ({len(page.text)} chars)")
                start = {"start_url": "https://splitkb.com/", "start_title": "splitkb.com"}
                searched = ["1. type_text 'LPF Glow'", "2. click button 'Search'", "3. click link"]
                _ask_jev(policy, "NOT done: product page, nothing added", goal, page, False,
                         actions=searched, **start)  # fmt: skip
                claimed = [*searched, "4. click [x] button 'Add to cart'"]
                _ask_jev(policy, "NOT done: same page, but the actions CLAIM the click", goal,
                         page, False, actions=claimed, **start)  # fmt: skip

                add = next(c for c in page.controls if c.label.strip().lower() == "add to cart")
                ctl.perform(Click(add.box.center))
                time.sleep(1.5)
                cart = look()
                print(f"after the add: {cart.url} ({len(cart.text)} chars)")
                _ask_jev(policy, "DONE: the cart holds the product", goal, cart, True,
                         actions=claimed, **start)  # fmt: skip
                swapped = dataclasses.replace(
                    cart, text=cart.text.replace(PRODUCT, "MBK Legend Blank Keycaps")
                )
                _ask_jev(policy, "NOT done: the cart holds a DIFFERENT product", goal, swapped,
                         False, actions=claimed, **start)  # fmt: skip

                wiki = {"start_url": "https://en.wikipedia.org/wiki/Main_Page",
                        "start_title": "Wikipedia, the free encyclopedia"}  # fmt: skip
                ctl.perform(Navigate("https://en.wikipedia.org/wiki/Alan_Turing"))
                article = look()
                did = ["1. type_text 'Alan Turing'", "2. click button 'Search'"]
                _ask_jev(policy, "DONE: the Alan Turing article is open",
                         'Open the Wikipedia article on "Alan Turing"', article, True,
                         actions=did, **wiki)  # fmt: skip
                _ask_jev(policy, "NOT done: the article open is a different one",
                         'Open the Wikipedia article on "Grace Hopper"', article, False,
                         actions=did, **wiki)  # fmt: skip
        finally:
            subprocess.run(["pkill", "-f", os.environ["BU_NAME"]], check=False)
    spent = policy._meter.total()  # noqa: SLF001 - total_usage() adds a writer this has none of
    print(f"  jev calls={spent.calls} in={spent.input_tokens} out={spent.output_tokens}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", type=Path, default=None)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--jev-pages", action="store_true")
    args = parser.parse_args()
    the_reask()
    the_tool_free_judge()
    the_wait_move()
    the_jev_done_critic()
    if args.jev_pages:
        the_live_pages()
    if args.judge is not None:
        the_real_judge(args.judge, args.n)
    print(f"\n{len(_failed)} failed" if _failed else "\nall passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
