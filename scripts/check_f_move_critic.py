"""Checks for the fast cold path's per-move judge, and for the seam it is reached through.

Pure logic: no browser, no model, no network.

    uv run python scripts/check_f_move_critic.py

One line per check; exits non-zero if any failed. What these CANNOT show is whether a run
judged this way still learns a skill that replays - that is a live run, and the numbers
are at ``DEFAULT_FAST_MOVES`` in ``src/skillweaver/config.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from skillweaver.agent.explorer import Explorer
from skillweaver.agent.move_critic import LiteralMoveCritic
from skillweaver.config import Settings
from skillweaver.contracts import Usage, Verdict
from skillweaver.orchestrator import _open_move_critic
from skillweaver.perception.dom import DomPerceiver

_failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail else ''}")
    if not ok:
        _failed.append(name)


def _obs(text: str = "") -> Any:
    elements = (SimpleNamespace(text=text),) if text else ()
    return SimpleNamespace(elements=elements, url="https://shop.example/")


class _Source:
    def __init__(self) -> None:
        self.last: Any = None

    def show(self, digest: str, url: str = "https://shop.example/") -> None:
        self.last = SimpleNamespace(digest=digest, url=url)


def _judged(before_digest: str, after_digest: str, *, after_url: str = "", text: str = "") -> Any:
    source = _Source()
    critic = LiteralMoveCritic(source)
    before, after = _obs(), _obs(text)
    source.show(before_digest)
    critic.open_move(before)
    source.show(after_digest, after_url or "https://shop.example/")
    return critic.judge("anything", before, after, "anything")


def literal_verdicts() -> None:
    changed = _judged("aaaa", "bbbb")
    check("an in-place change is ok, free and programmatic", changed.ok and not changed.escalated)
    check("and says how little it judged", "NOT judged" in changed.reason, changed.policy)

    same = _judged("aaaa", "aaaa")
    check("an unchanged page is a failed move", not same.ok and same.policy == "literal-unchanged")
    check("decisively, so it may be filed as a dead end", same.confidence == 1.0)

    moved = _judged("aaaa", "aaaa", after_url="https://shop.example/cart")
    check("a new address is a change even at an equal digest", moved.ok, moved.policy)

    broken = _judged("aaaa", "bbbb", text="404 page not found")
    check("an error page that 'changed' is still a failure", not broken.ok, broken.reason[:60])


def unread_is_unknown() -> None:
    source = _Source()
    critic = LiteralMoveCritic(source)
    source.show("aaaa")
    verdict = critic.judge("g", _obs(), _obs())
    check("no open_move: not ok, confidence 0", not verdict.ok and verdict.confidence == 0.0)

    before, other = _obs(), _obs()
    critic.open_move(other)
    source.show("bbbb")
    verdict = critic.judge("g", before, _obs())
    check("open_move for ANOTHER screen is not used", verdict.policy == "literal-unread")

    critic.open_move(before)
    critic.judge("g", before, _obs())
    verdict = critic.judge("g", before, _obs())
    check("one open_move serves one judgement", verdict.policy == "literal-unread")


class _Recording:
    def __init__(self, name: str, log: list[str]) -> None:
        self._name, self._log = name, log

    def judge(self, goal: str, before: Any, after: Any, expectation: str | None = None) -> Verdict:
        self._log.append(self._name)
        return Verdict(True, self._name)


def the_done_claim_keeps_the_full_critic() -> None:
    calls: list[str] = []
    llm = SimpleNamespace(total_usage=lambda: Usage(), name=lambda: "stub")
    recorder = SimpleNamespace()
    explorer = Explorer(
        llm,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        critic=_Recording("full", calls),
        move_critic=_Recording("literal", calls),
        recorder=recorder,  # type: ignore[arg-type]
    )
    run = SimpleNamespace(usage_mark=Usage(), spend=SimpleNamespace(add_usage=lambda usage: None))
    move = SimpleNamespace(expect="", summary="s")
    explorer._judge("g", _obs(), _obs(), move, run, per_move=True)  # type: ignore[arg-type]
    explorer._judge("g", _obs(), _obs(), move, run)  # type: ignore[arg-type]
    check("a move goes to move_critic, a done claim to critic", calls == ["literal", "full"])

    calls.clear()
    default = Explorer(
        llm,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        critic=_Recording("full", calls),
        recorder=recorder,  # type: ignore[arg-type]
    )
    default._judge("g", _obs(), _obs(), move, run, per_move=True)  # type: ignore[arg-type]
    check("with no move_critic every judgement is the full critic's", calls == ["full"])


def only_the_jev_path_gets_it() -> None:
    dom = DomPerceiver()
    on = _open_move_critic(Settings(policy="jev", perception="dom", fast_moves=True), dom)
    check("jev + dom + fast_moves: literal", isinstance(on, LiteralMoveCritic))
    off = _open_move_critic(Settings(policy="jev", perception="dom", fast_moves=False), dom)
    check("fast_moves off: the cold critic", off is None)
    default = _open_move_critic(Settings(fast_moves=True), dom)
    check("the default policy never gets it, even switched on", default is None)
    pixels = _open_move_critic(Settings(policy="jev", fast_moves=True), SimpleNamespace())  # type: ignore[arg-type]
    check("nor does a perceiver with no page snapshot", pixels is None)


def main() -> int:
    literal_verdicts()
    unread_is_unknown()
    the_done_claim_keeps_the_full_critic()
    only_the_jev_path_gets_it()
    print(f"\n{len(_failed)} failed" if _failed else "\nall passed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
