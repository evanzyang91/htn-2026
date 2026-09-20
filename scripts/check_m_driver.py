"""Checks for what ``JevDriver`` took from upstream ``jev-ultrafast`` ``cbf517a``..``0da4053``.

    uv run python scripts/check_m_driver.py

Names (a label plus the context around it), the run-wide ``inert`` memory, a scroll judged
by what it reveals, and the operation handed to ``DomPerceiver.rest_after``. No network, no
browser, no key: the policy and the perceiver are stubs, and most of what is under test is
a pure function. One line per check and a non-zero exit if any failed. Not the proof that
the driver works - that is a real run - but what keeps upstream's own cases true here,
``test_scroll_that_reveals_nothing_counts_against_itself`` among them.
``scripts/check_c_driver.py`` holds the previous port's cases and must still pass too.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace
from typing import Any

from skillweaver.agent import jev_driver as jd
from skillweaver.contracts import Box
from skillweaver.llm.jev_ import PolicyDecision
from skillweaver.perception.dom import DomControl, DomSnapshot

_failures: list[str] = []

if "context" in {f.name for f in dataclasses.fields(DomControl)}:
    Control: Any = DomControl
else:
    # ``context`` is another worker's field on DomControl; until it lands, a subclass
    # carries it, which is exactly what the driver's getattr is tolerant of.
    @dataclasses.dataclass(frozen=True, slots=True)
    class Control(DomControl):  # type: ignore[no-redef]
        context: str | None = None


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else f'  <- {detail!r}'}")
    if not ok:
        _failures.append(name)


def control(index: int, name: str, context: str | None = None, *, editable: bool = False) -> Any:
    return Control(
        index=index,
        element_id=f"id{index}",
        role="textbox" if editable else "button",
        name=name,
        kind="text_field" if editable else "button",
        box=Box(0, index * 30, 100, 20),
        editable=editable,
        context=context,
    )


def snapshot(*controls: Any, text: str = "page", scroll_y: float = 0.0) -> DomSnapshot:
    return DomSnapshot(
        url="https://example.test/",
        title="t",
        text=text,
        controls=tuple(controls),
        scroll_y=scroll_y,
        page_height=10_000.0,
        viewport=Box(0, 0, 1280, 800),
        by_element_id={c.element_id: c for c in controls},
    )


def decision(operation: str, element_id: str | None = None, text: str = "") -> PolicyDecision:
    fields = {f.name for f in dataclasses.fields(PolicyDecision)}
    given = {
        "operation": operation,
        "element_id": element_id,
        "text": text,
        "why": f"{operation} [{element_id}] p=0.5",
        "confidence": 0.5,
        "probability": 0.5,
        "policy_ms": 0.0,
        "latency_ms": 0.0,
    }
    return PolicyDecision(**{k: v for k, v in given.items() if k in fields})


class ScriptedPolicy:
    def __init__(self, *decisions: PolicyDecision) -> None:
        self._decisions = list(decisions)
        self.asked: list[dict[str, set[str]]] = []

    def name(self) -> str:
        return "scripted"

    def total_usage(self) -> Any:
        return None

    def decide(self, goal: str, snap: DomSnapshot, history: Any, exclude: Any) -> PolicyDecision:
        self.asked.append({k: set(v) for k, v in (exclude or {}).items()})
        return self._decisions.pop(0)


class OldPerceiver:
    """Today's ``rest_after``: no ``operation`` keyword."""

    site_ms = 0.0

    def __init__(self, snap: DomSnapshot) -> None:
        self.last = snap
        self.armed: list[tuple[Any, ...]] = []

    def rest_after(self, actions: int, basis: DomSnapshot, *, waited: bool = False) -> None:
        self.armed.append((actions, waited))


class NewPerceiver(OldPerceiver):
    """The other worker's: the operation is named."""

    def rest_after(  # type: ignore[override]
        self, actions: int, basis: DomSnapshot, *, waited: bool = False, operation: str = ""
    ) -> None:
        self.armed.append((actions, waited, operation))


def propose(driver: jd.JevDriver, snap: DomSnapshot, history: list[str]) -> dict[str, Any]:
    answer = driver.propose(
        SimpleNamespace(text="do the thing"),  # type: ignore[arg-type]
        SimpleNamespace(url=snap.url),  # type: ignore[arg-type]
        set(snap.by_element_id),  # type: ignore[arg-type]
        history,
        (),
        None,
    )
    return json.loads(answer)


ITEMS = ["Sourdough", "Rye", "Baguette", "Focaccia", "Ciabatta", "Brioche", "Challah"]


def seven() -> list[Any]:
    return [control(i + 1, "Add item to cart", item) for i, item in enumerate(ITEMS)]


def check_names() -> None:
    adds = seven()
    names = [jd.control_name(c) for c in adds]
    check("names: seven shared labels are seven names", len(set(names)) == 7, names)
    check("names: upstream's spelling", names[0] == "Add item to cart — Sourdough", names[0])
    check("names: no context is the label", jd.control_name(control(9, "Go")) == "Go")
    check(
        "names: a control without the field at all",
        jd.control_name(SimpleNamespace(label="Go")) == "Go",
    )
    snap = snapshot(*adds)
    # Pruned ONE AT A TIME: each inert name takes one control, never the label's seven.
    for count in range(1, 7):
        gone = set(names[:count])
        exclude, restored = jd._exclusions(snap, (), None, inert=gone)
        left = [c for c in adds if jd.control_name(c) not in exclude.get("LABELS", ())]
        check(
            f"names: {count} pruned leaves {7 - count}",
            exclude.get("LABELS") == gone and len(left) == 7 - count and not restored,
            (exclude, restored),
        )
    exclude, restored = jd._exclusions(snap, (), None, inert=set(names))
    check(
        "put-back: all seven inert are offered again, not a false BLOCKED",
        "LABELS" not in exclude and restored == {"INERT"},
        (exclude, restored),
    )
    check(
        "put-back: and the note says so",
        "still offered" in jd._note(exclude, restored, set(names)),
    )
    # The two sets are tried apart: churn that would empty the head does not take the
    # inert name's withholding down with it.
    exclude, restored = jd._exclusions(snap, (), None, inert={names[0]}, labels=set(names[1:]))
    check(
        "put-back: churn put back, the inert name still withheld",
        exclude.get("LABELS") == {names[0]} and restored == {"LABELS"},
        (exclude, restored),
    )
    note = jd._note(exclude, restored, {names[0]})
    check("note: an inert name is not called churn", "changed nothing 3 times" in note, note)
    # The bare label no longer matches anything when the controls carry context.
    exclude, restored = jd._exclusions(snap, (), None, labels={"Add item to cart"})
    check(
        "names: the bare shared label withholds none of the seven",
        not any(jd.control_name(c) in exclude.get("LABELS", ()) for c in adds),
    )


def step(kind: str, label: str, changed: bool, **more: Any) -> dict[str, Any]:
    return {"kind": kind, "label": label, "page_changed": changed, **more}


def check_inert() -> None:
    inert: dict[str, int] = {}
    name = "Make 2 required selections"
    for expected in (1, 2):
        jd.tally(inert, step("CLICK", name, False))
        check(f"inert: miss {expected} is not withheld", jd.withheld_inert(inert) == (set(), set()))
    jd.tally(inert, step("CLICK", name, False))
    check("inert: the third miss withholds the name", jd.withheld_inert(inert) == (set(), {name}))
    jd.tally(inert, step("CLICK", name, True))
    check("inert: one change resets it to zero", inert[name] == 0, inert)
    check("inert: and it is offered again", jd.withheld_inert(inert) == (set(), set()))
    jd.tally(inert, step("TYPE_TEXT", "Search", False))
    check("inert: typing counts", inert["Search"] == 1, inert)
    for kind in ("WAIT", "BACK", "ENTER", "DONE"):
        jd.tally(inert, step(kind, kind, False))
    check("inert: other operations are spent_controls' business", set(inert) == {name, "Search"})
    a, b = (jd.control_name(c) for c in seven()[:2])
    inert = {}
    for _ in range(3):
        jd.tally(inert, step("CLICK", a, False))
    jd.tally(inert, step("CLICK", b, False))
    check("inert: counted per NAME, not per label", inert == {a: 3, b: 1}, inert)


def check_scroll() -> None:
    # Upstream's test_scroll_that_reveals_nothing_counts_against_itself, through the driver:
    # the offset moves every time, so page_changed is TRUE every time, and nothing new shows.
    adds = seven()[:3]
    frames = [snapshot(*adds, scroll_y=560.0 * i) for i in range(5)]
    policy = ScriptedPolicy(*[decision("SCROLL_DOWN") for _ in range(3)], decision("CLICK", "id1"))
    perceiver = NewPerceiver(frames[0])
    driver = jd.JevDriver(policy, perceiver)  # type: ignore[arg-type]
    history: list[str] = []
    for frame in frames[:4]:
        perceiver.last = frame
        propose(driver, frame, history)
        history.append("moved")
    scrolls = driver._steps[:3]
    check(
        "scroll: the offset moved, so page_changed is true", all(s["page_changed"] for s in scrolls)
    )
    check("scroll: each step revealed == 0", [s.get("revealed") for s in scrolls] == [0, 0, 0])
    check("scroll: inert count is 3", driver._inert.get("SCROLL_DOWN") == 3, driver._inert)
    check("scroll: no snapshot names left on a closed step", not any("seen" in s for s in scrolls))
    check(
        "scroll: withheld under CONTROLS on the fourth ask, not before",
        [a.get("CONTROLS") for a in policy.asked] == [None, None, None, {"SCROLL_DOWN"}],
        policy.asked,
    )
    check(
        "scroll: spent_controls cannot see it (every state differs)",
        jd.spent_controls(driver._steps, frames[3].digest) == set(),
    )
    check(
        "rest_after: the operation is named to a perceiver that takes it",
        [a[2] for a in perceiver.armed] == ["SCROLL_DOWN"] * 3 + ["CLICK"],
        perceiver.armed,
    )

    # A revealing scroll resets it.
    inert = {"SCROLL_DOWN": 2}
    jd.tally(inert, step("SCROLL_DOWN", "SCROLL_DOWN", True, revealed=4))
    check("scroll: one that reveals resets the count", inert == {"SCROLL_DOWN": 0}, inert)
    before, after = snapshot(*seven()[:3]), snapshot(*seven()[2:6])
    check("scroll: revealed counts new NAMES", jd.count_revealed(jd.names_on(before), after) == 3)
    same_label = snapshot(*[control(i, "Add item to cart") for i in range(1, 4)])
    check(
        "scroll: more of one unnamed label reveals nothing",
        jd.count_revealed(jd.names_on(same_label), same_label) == 0,
    )
    up = {"SCROLL_UP": 3, "SCROLL_DOWN": 1, "Go": 3}
    check("scroll: each direction on its own", jd.withheld_inert(up) == ({"SCROLL_UP"}, {"Go"}))


def check_driver_names() -> None:
    adds = seven()
    a, b = jd.control_name(adds[0]), jd.control_name(adds[1])
    # Three misses on the first button, each on a DIFFERENT page state (the text differs
    # per lap), so no identity rule can carry them over: this is the reopen case.
    frames = [snapshot(*adds, text=f"lap {i}") for i in range(4)]
    policy = ScriptedPolicy(*[decision("CLICK", "id1") for _ in range(3)], decision("CLICK", "id2"))
    perceiver = OldPerceiver(frames[0])
    driver = jd.JevDriver(policy, perceiver)  # type: ignore[arg-type]
    history: list[str] = []
    answers = []
    for frame in frames:
        perceiver.last = frame
        # Stands in for the reopen between laps: the open step is re-filed under the state
        # it is about to be closed against, so it closes as a miss made on THAT state.
        if driver._steps:
            driver._steps[-1]["state"] = frame.digest
        answers.append(propose(driver, frame, history))
        history.append("moved")
    check("driver: the step's label key holds the NAME", driver._steps[0]["label"] == a)
    check("driver: the thought names which button", a in answers[0]["thought"], answers[0])
    check(
        "driver: withheld by name across page states at the third miss",
        [x.get("LABELS") for x in policy.asked] == [None, None, None, {a}],
        policy.asked,
    )
    check("driver: the sibling is untouched", driver._inert.get(b) is None, driver._inert)
    check(
        "rest_after: a perceiver without the keyword still works",
        perceiver.armed == [(1, False)] * 4,
        perceiver.armed,
    )


def check_cycling_with_names() -> None:
    def named(labels: list[str]) -> list[dict[str, Any]]:
        return [step("CLICK", label, True) for label in labels]

    amazon = ["Open Search", "Go", "Open Search", "Open Search", "Go", "Open Search"]
    check(
        "cycling: upstream's Amazon churn still holds", jd.cycling(named(amazon))[1] == set(amazon)
    )
    names = [jd.control_name(c) for c in seven()]
    check(
        "cycling: seven adds by name are progress, not churn",
        jd.cycling(named(names)) == (set(), set()),
    )
    pair = [names[0], names[1]] * 3
    check(
        "cycling: two NAMES churning are withheld as names", jd.cycling(named(pair))[1] == set(pair)
    )
    varied = ["Search", "Go", "Item A", "Add to cart", "Search", "Go", "Item B"]
    check("cycling: a varied flow", jd.cycling(named(varied)) == (set(), set()))
    check("cycling: too short", jd.cycling(named(["Open Search", "Go"])) == (set(), set()))


def main() -> int:
    check_names()
    check_inert()
    check_scroll()
    check_driver_names()
    check_cycling_with_names()
    print(f"\n{len(_failures)} failed: {_failures}" if _failures else "\nall checks passed")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
