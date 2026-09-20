"""Checks for the loop rules ``JevDriver`` took from upstream ``jev-ultrafast`` ``1489129``.

    uv run python scripts/check_c_driver.py

No network, no browser, no key: the policy and the perceiver are stubs that replay a
script, and the rules under test are pure functions of the step list. One line per check,
and a non-zero exit if any failed. This is not the proof that the driver works - that is a
real run - it is what keeps upstream's own cases (``tests/test_agent.py`` there) true here.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

from skillweaver.agent import jev_driver as jd
from skillweaver.agent.explorer import Attempt, _resolve
from skillweaver.contracts import Box, PressKey, Wait
from skillweaver.llm.jev_ import PolicyDecision
from skillweaver.perception.dom import DomControl, DomSnapshot

_failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else f'  <- {detail!r}'}")
    if not ok:
        _failures.append(name)


def steps(labels: list[str], kind: str = "CLICK") -> list[dict[str, Any]]:
    """Upstream's ``steps`` helper: every step a click that CHANGED the page."""
    return [{"label": label, "kind": kind, "page_changed": True} for label in labels]


def miss(kind: str, state: str = "s", label: str | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label or kind,
        "state": state,
        "page_changed": False,
        "element_id": None,
    }


def control(index: int, name: str, *, editable: bool = False) -> DomControl:
    return DomControl(
        index=index,
        element_id=f"id{index}",
        role="textbox" if editable else "button",
        name=name,
        kind="text_field" if editable else "button",
        box=Box(0, index * 30, 100, 20),
        editable=editable,
    )


def snapshot(*controls: DomControl, text: str = "page", back: bool = False) -> DomSnapshot:
    return DomSnapshot(
        url="https://example.test/",
        title="t",
        text=text,
        controls=tuple(controls),
        can_go_back=back,
        by_element_id={c.element_id: c for c in controls},
    )


class ScriptedPolicy:
    """Answers from a list, and keeps what it was asked with."""

    def __init__(self, *decisions: PolicyDecision) -> None:
        self._decisions = list(decisions)
        self.asked: list[tuple[DomSnapshot, dict[str, set[str]], int]] = []

    def name(self) -> str:
        return "scripted"

    def total_usage(self) -> Any:
        return None

    def decide(self, goal: str, snap: DomSnapshot, history: Any, exclude: Any) -> PolicyDecision:
        self.asked.append((snap, {k: set(v) for k, v in (exclude or {}).items()}, len(history)))
        return self._decisions.pop(0)


class StubPerceiver:
    site_ms = 0.0

    def __init__(self, snap: DomSnapshot) -> None:
        self.last = snap
        self.armed: list[tuple[int, bool]] = []

    def rest_after(self, actions: int, basis: DomSnapshot, *, waited: bool = False) -> None:
        self.armed.append((actions, waited))


class StubController:
    def supports(self, kind: str) -> bool:
        return True

    def viewport(self) -> Box:
        return Box(0, 0, 1280, 800)

    def describe(self) -> str:
        return "stub"


def propose(driver: jd.JevDriver, snap: DomSnapshot, history: list[str], **kw: Any) -> dict:
    observation = SimpleNamespace(url=snap.url)
    task = SimpleNamespace(text="do the thing")
    catalog = set(snap.by_element_id)
    answer = driver.propose(
        task,  # type: ignore[arg-type]
        observation,  # type: ignore[arg-type]
        catalog,  # type: ignore[arg-type]
        history,
        kw.get("dead_ends", ()),
        kw.get("rejection"),
    )
    return json.loads(answer)


def check_cycling() -> None:
    # Upstream's own cases. Every step changes the page, so no page_changed or
    # fingerprint rule can see these; the Amazon loop doubles back, so strict
    # alternation must not be required.
    amazon = ["Open Search", "Go", "Open Search", "Open Search", "Go", "Open Search"]
    check("cycling: the Amazon open/go churn", jd.cycling(steps(amazon)) == (set(), set(amazon)))
    three = ["Search", "Open Search", "Clear"] * 3
    check("cycling: three labels filling the window", jd.cycling(steps(three))[1] == set(three))
    check(
        "cycling: one label repeating is progress",
        jd.cycling(steps(["Increase quantity by 1"] * 8)) == (set(), set()),
    )
    varied = ["Search", "Go", "Item A", "Add to cart", "Search", "Go", "Item B"]
    check("cycling: a varied multi-item flow", jd.cycling(steps(varied)) == (set(), set()))
    check(
        "cycling: too short to conclude", jd.cycling(steps(["Open Search", "Go"])) == (set(), set())
    )
    # This project's half: a target-less move is named by its operation and comes back
    # under the other key; DONE claims are not moves and must not pad a short history.
    scrolls = steps(["SCROLL_DOWN", "SCROLL_UP"] * 3, kind="")
    for step in scrolls:
        step["kind"] = step["label"]
    check(
        "cycling: scroll churn is CONTROLS",
        jd.cycling(scrolls) == ({"SCROLL_DOWN", "SCROLL_UP"}, set()),
    )
    mixed = steps(["Open", "Open", "Open"]) + [miss("WAIT"), miss("WAIT"), miss("WAIT")]
    check("cycling: a label and a control split", jd.cycling(mixed) == ({"WAIT"}, {"Open"}))
    padded = steps(["Open Search", "Go"] * 2) + [miss("DONE")] * 4
    check("cycling: DONE claims do not count as moves", jd.cycling(padded) == (set(), set()))
    labelled_wait = steps(["WAIT", "Go"] * 3)
    check(
        "cycling: a button LABELLED 'WAIT' stays a label",
        jd.cycling(labelled_wait) == (set(), {"WAIT", "Go"}),
    )


def check_controls() -> None:
    check("controls: nothing on a first miss", jd.spent_controls([miss("WAIT")], "s") == set())
    check(
        "controls: withheld on the second", jd.spent_controls([miss("WAIT")] * 2, "s") == {"WAIT"}
    )
    through = [miss("SCROLL_DOWN"), miss("WAIT"), miss("SCROLL_DOWN")]
    check("controls: counted THROUGH a wait", jd.spent_controls(through, "s") == {"SCROLL_DOWN"})
    changed = [miss("ENTER"), {**miss("CLICK"), "page_changed": True}, miss("ENTER")]
    check("controls: a change ends the streak", jd.spent_controls(changed, "s") == set())
    check(
        "controls: another page state does not count",
        jd.spent_controls([miss("BACK", "x")] * 2, "s") == set(),
    )
    check(
        "controls: an unclosed step does not count",
        jd.spent_controls([miss("WAIT"), {**miss("WAIT"), "page_changed": None}], "s") == set(),
    )
    check("controls: a click is never one", jd.spent_controls([miss("CLICK")] * 3, "s") == set())
    check("controls: DONE is never one", jd.spent_controls([miss("DONE")] * 3, "s") == set())

    # Targets keep this project's rule: spent on a first miss, released by a WAIT.
    snap = snapshot(control(1, "Go"), control(2, "Other"))
    driver = jd.JevDriver(ScriptedPolicy(), StubPerceiver(snap))  # type: ignore[arg-type]
    state = snap.digest
    driver._steps = [{**miss("CLICK", state), "element_id": "id1"}]
    check("targets: spent on the first miss", driver._spent(state) == {"CLICK": {"id1"}})
    driver._steps.append(miss("WAIT", state))
    check("targets: a WAIT puts them back", driver._spent(state) == {})


def check_exclusions() -> None:
    snap = snapshot(control(1, "Open Search"), control(2, "Go"), control(3, "Cart"), back=True)
    exclude, restored = jd._exclusions(snap, (), None, controls={"WAIT", "DONE"}, labels={"Go"})
    check(
        "exclude: reserved keys, DONE never withheld",
        exclude == {"CONTROLS": {"WAIT"}, "LABELS": {"Go"}} and not restored,
        exclude,
    )
    check("exclude: reserved keys are not counted as targets", jd._withheld(exclude) == 0)
    note = jd._note(exclude, restored)
    check(
        "exclude: the note names them",
        "WAIT" in note and "'Go'" in note and "target" not in note,
        note,
    )
    everything = {"Open Search", "Go", "Cart"}
    exclude, restored = jd._exclusions(snap, (), None, labels=everything)
    check(
        "exclude: labels that empty a head are put back",
        exclude == {} and restored == {"LABELS"},
        (exclude, restored),
    )
    check("exclude: and the note says so", "still offered" in jd._note(exclude, restored))
    dead = [Attempt("fp", "click:id1:left:1", "click", "no")]
    exclude, restored = jd._exclusions(snap, dead, None, labels={"Go", "Cart"})
    check(
        "exclude: ids and labels together empty the head",
        restored == {"LABELS"} and exclude == {"CLICK": {"id1"}},
        (exclude, restored),
    )
    check("exclude: nothing to withhold is nothing", jd._exclusions(snap, ()) == ({}, set()))
    check(
        "without: BACK in CONTROLS turns the flag off",
        jd._without(snap, {"BACK"}).can_go_back is False,
    )
    check(
        "without: ENTER tolerates a snapshot with no such field",
        jd._without(snap, {"ENTER"}) is snap,
    )


def check_done() -> None:
    check("done: first claim is a look", jd.fresh_look("DONE", False) == (True, True))
    check("done: second claim stands", jd.fresh_look("DONE", True) == (False, True))
    check("done: an action resets it", jd.fresh_look("CLICK", True) == (False, False))
    check("done: a WAIT is an action", jd.fresh_look("WAIT", True) == (False, False))
    check("done: BLOCKED leaves it", jd.fresh_look("BLOCKED", True) == (False, True))

    snap = snapshot(control(1, "Go"))
    policy = ScriptedPolicy(
        PolicyDecision("CLICK", element_id="id1", why="CLICK [1] 'Go' (0.9)"),
        PolicyDecision("DONE"),
        PolicyDecision("DONE"),
        PolicyDecision("DONE"),
        PolicyDecision("CLICK", element_id="id1"),
        PolicyDecision("DONE"),
    )
    perceiver = StubPerceiver(snap)
    driver = jd.JevDriver(policy, perceiver)  # type: ignore[arg-type]
    history: list[str] = []

    first = propose(driver, snap, history)
    check(
        "sequence: the click is a click", first["action"] == {"kind": "click", "element_id": "id1"}
    )
    check(
        "sequence: its step keeps a clean label beside the why",
        driver._steps[-1]["label"] == "Go" and driver._steps[-1]["action"].startswith("CLICK [1]"),
    )
    history.append("1. click -> ok")

    # The click's effect lands only during the look: the page reads the same at first.
    look = propose(driver, snap, history)
    check(
        "sequence: first DONE is a wait, not a claim",
        look["done"] is False and look["action"] == {"kind": "wait", "ms": jd.DONE_LOOK_MS},
        look,
    )
    check("sequence: the look is armed as NOT a wait", perceiver.armed[-1] == (1, False))
    check(
        "sequence: the look is not a step",
        len(driver._steps) == 1 and driver._steps[-1]["page_changed"] is False,
    )
    history.append("2. wait -> FAILED")

    landed = snapshot(control(1, "Go"), text="page, cart (1)")
    perceiver.last = landed
    claim = propose(driver, landed, history, rejection="your last move (wait 150ms) did not work")
    check(
        "sequence: second consecutive DONE is the claim",
        claim["done"] is True and "action" not in claim,
        claim,
    )
    check(
        "sequence: what landed during the look closes the step",
        driver._steps[0]["page_changed"] is True,
    )
    check("sequence: the look's rejection is relayed to nobody", "refused" not in driver._steps[0])
    check("sequence: the claim IS a step", [s["kind"] for s in driver._steps] == ["CLICK", "DONE"])

    again = propose(driver, landed, history, rejection="the critic disagreed")
    check("sequence: a refused claim is not looked at again", again["done"] is True)
    check(
        "sequence: so four claims are still four DONE steps",
        [s["kind"] for s in driver._steps][1:] == ["DONE", "DONE"],
    )

    propose(driver, landed, history)
    after = propose(driver, landed, history)
    check(
        "sequence: a real action makes the next DONE a look again",
        after["done"] is False and after["action"]["kind"] == "wait",
        after,
    )
    check(
        "sequence: a wait is never counted by NO_CHANGE_LIMIT's streak",
        all(s["kind"] != "WAIT" for s in driver._steps),
    )


def check_driver_emits() -> None:
    snap = snapshot(control(1, "Open Search"), control(2, "Go"), control(3, "Cart"), back=True)
    policy = ScriptedPolicy(PolicyDecision("CLICK", element_id="id3"))
    driver = jd.JevDriver(policy, StubPerceiver(snap))  # type: ignore[arg-type]
    churn = ["Open Search", "Go", "Open Search", "Open Search", "Go", "Open Search"]
    driver._steps = [
        {**s, "state": f"s{i}", "element_id": "x", "model_ms": 0.0, "site_mark": 0.0}
        for i, s in enumerate(steps(churn))
    ]
    back_dead = [Attempt("fp", jd.BACK_SIGNATURE, "go back", "no")]
    answer = propose(driver, snap, ["6. x"], dead_ends=back_dead)
    asked, exclude, _ = policy.asked[0]
    check(
        "driver: churn labels reach the policy as LABELS",
        exclude.get("LABELS") == {"Open Search", "Go"},
        exclude,
    )
    check(
        "driver: a dead back is CONTROLS and the flag",
        exclude.get("CONTROLS") == {"BACK"} and asked.can_go_back is False,
        exclude,
    )
    check(
        "driver: the thought says what was withheld",
        "churning" in answer["thought"] and "BACK" in answer["thought"],
        answer["thought"],
    )


def check_enter() -> None:
    snap = snapshot(control(1, "Search", editable=True))
    answer = jd._answer(PolicyDecision("ENTER", confidence=0.9, probability=0.8), set(), snap)  # type: ignore[arg-type]
    check(
        "enter: one press_key action, not done",
        answer["action"] == {"kind": "press_key", "keys": ["Enter"]} and answer["done"] is False,
        answer,
    )
    check(
        "enter: a truthful expect",
        "submitted" in answer["expect"] and "focused field" in answer["expect"],
        answer["expect"],
    )
    action, signature, _ = _resolve(answer["action"], None, StubController())  # type: ignore[arg-type]
    check(
        "enter: the explorer grounds it to PressKey(Enter)",
        action == PressKey(("Enter",)) and signature == "press_key:Enter",
        (action, signature),
    )
    check("enter: it is in _ACTIONS_IN as one action", jd._ACTIONS_IN.get("ENTER") == 1)
    check(
        "enter: it is a control",
        "ENTER" in jd.CONTROL_OPERATIONS and "DONE" not in jd.CONTROL_OPERATIONS,
    )
    named = SimpleNamespace(enter_label="Search repositories", by_element_id={})
    told = jd._answer(PolicyDecision("ENTER"), set(), named)  # type: ignore[arg-type]
    check(
        "enter: the other worker's enter_label is named when present",
        "'Search repositories'" in told["thought"],
        told["thought"],
    )
    look = jd._look_again(PolicyDecision("DONE"))
    action, signature, _ = _resolve(look["action"], None, StubController())  # type: ignore[arg-type]
    check(
        "look: grounds to a Wait with its own signature",
        action == Wait(jd.DONE_LOOK_MS) and signature != f"wait:{jd.WAIT_MS}",
        signature,
    )


def main() -> int:
    check_cycling()
    check_controls()
    check_exclusions()
    check_done()
    check_driver_emits()
    check_enter()
    print(f"\n{'FAILED: ' + ', '.join(_failures) if _failures else 'all checks passed'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
