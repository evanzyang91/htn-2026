"""Pure-logic checks of the Jev policy's request building, offer, retry and speculation.

Run with ``uv run python scripts/check_b_policy.py``; prints one line per check and exits
non-zero if any failed. It needs NO network and no credential: the policy's ``_post`` is
replaced by a scripted one and the text writer by a stub, so nothing here says the policy
CHOOSES well - only that what it is asked, offered and charged is what the code claims. A
live run is still the proof this project accepts.

``section``/``opens`` and ``can_press_enter``/``enter_label`` are another module's half
of this port (``perception/dom.py``). The hand-built snapshots below carry them through a
subclass when the dataclasses do not have them yet, which is also a check that
``llm/jev_.py`` reads them tolerantly.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from skillweaver.contracts import Box, ElementKind, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm import jev_
from skillweaver.llm.jev_ import JevPolicy, NoFieldValue, progress, targets_of
from skillweaver.llm.openai_ import _PAGE_TEXT_SHOWN
from skillweaver.llm.usage import UsageMeter
from skillweaver.perception.dom import DomControl, DomSnapshot

if "section" in DomControl.__dataclass_fields__:
    Control = DomControl
else:

    @dataclasses.dataclass(frozen=True)
    class Control(DomControl):  # type: ignore[no-redef]
        section: str | None = None
        opens: str | None = None


if "can_press_enter" in DomSnapshot.__dataclass_fields__:
    Snapshot = DomSnapshot
else:

    @dataclasses.dataclass(frozen=True)
    class Snapshot(DomSnapshot):  # type: ignore[no-redef]
        can_press_enter: bool = False
        enter_label: str = ""


def control(index: int, name: str, *, editable: bool = False, **extra: Any) -> DomControl:
    return Control(
        index=index,
        element_id=f"e{index}",
        role="textbox" if editable else "button",
        name=name,
        kind=ElementKind.text_field if editable else ElementKind.button,
        box=Box(0, index * 10, 100, 10),
        editable=editable,
        **extra,
    )


def snapshot(controls: Sequence[DomControl], **extra: Any) -> DomSnapshot:
    fields = {"page_height": 2000.0, "viewport": Box(0, 0, 800, 600), "scroll_y": 300.0}
    return Snapshot(
        url="https://example.test/",
        title="t",
        text="x" * 5000,
        controls=tuple(controls),
        **{**fields, **extra},
    )


class StubWriter:
    """A text writer that counts, optionally dawdles, and charges like a real one."""

    def __init__(self, value: str | Exception = "typed", delay: float = 0.0) -> None:
        self.value, self.delay = value, delay
        self.calls: list[str] = []
        self.threads: list[str] = []
        self.meter = UsageMeter()

    def write(self, goal: str, field: DomControl, snap: DomSnapshot, history: Any) -> str:
        self.calls.append(field.element_id)
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        self.meter.add(Usage(10, 2, 1, 0.0))
        if isinstance(self.value, Exception):
            raise self.value
        return self.value

    def total_usage(self) -> Usage:
        return self.meter.total()


def answer(choice: str, keys: Sequence[str]) -> dict[str, Any]:
    rest = (1.0 - 0.9) / max(len(keys) - 1, 1)
    probabilities = {key: (0.9 if key == choice else rest) for key in keys}
    if len(keys) == 1:
        probabilities = {choice: 1.0}
    return {"choice": choice, "confidence": 0.9, "probabilities": probabilities}


class ScriptedPolicy(JevPolicy):
    """``JevPolicy`` with the wire replaced: ``script`` is called with each POST body and
    returns the operation to answer, or raises what ``_post`` would raise."""

    __slots__ = ("bodies", "script")

    def __init__(self, writer: Any, script: Callable[[Mapping[str, Any]], str]) -> None:
        super().__init__(writer, api_key="check-script-key")
        self.script = script
        self.bodies: list[Mapping[str, Any]] = []

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.bodies.append(body)
        chosen = self.script(body)
        questions = body["questions"]
        answers = {"operation": answer(chosen, list(questions["operation"]["criteria"]))}
        for key, question in questions.items():
            if key != "operation":
                keys = list(question["criteria"])
                answers[key] = answer(keys[0], keys)
        return {"answers": answers, "usage": {"input_tokens": 100, "output_tokens": 5}}


FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail and not ok else ''}")
    if not ok:
        FAILED.append(name)


def offer_and_exclusions() -> None:
    controls = [
        control(1, "Search", editable=True),
        control(2, "Go"),
        control(3, "Help", section="footer", opens="menu"),
    ]
    plain = snapshot(controls, can_go_back=True)
    offered = jev_._offered(plain, targets_of(plain))
    check("ENTER is not offered without can_press_enter", "ENTER" not in offered)
    check("ENTER is in OPERATIONS", "ENTER" in jev_.OPERATIONS)

    focused = snapshot(controls, can_go_back=True, can_press_enter=True, enter_label="Search")
    offered = jev_._offered(focused, targets_of(focused))
    check(
        "ENTER is offered with can_press_enter, worded from enter_label",
        offered.get("ENTER") == "Press Enter to submit Search",
        str(offered.get("ENTER")),
    )
    check("ENTER has no target head", "ENTER" not in targets_of(focused))
    bare = jev_._offered(snapshot(controls, can_press_enter=True), {})
    check("ENTER with no label still reads", bare["ENTER"].endswith("the focused field"))

    exclude = {"CONTROLS": {"SCROLL_DOWN", "WAIT", "BACK", "ENTER", "DONE", "BLOCKED"}}
    targets = targets_of(focused, exclude)
    check("CONTROLS is not an operation name", set(targets) == {"CLICK", "TYPE_TEXT"})
    check("CONTROLS withholds no target", len(targets["CLICK"]) == 3)
    offered = jev_._offered(focused, targets, exclude["CONTROLS"])
    check(
        "CONTROLS names are withheld from the offer",
        not {"SCROLL_DOWN", "WAIT", "BACK", "ENTER"} & set(offered),
        str(sorted(offered)),
    )
    check("an unnamed control stays offered", "SCROLL_UP" in offered)
    check("DONE and BLOCKED are never withheld", {"DONE", "BLOCKED"} <= set(offered))

    targets = targets_of(focused, {"LABELS": {"Search"}, "CLICK": {"e2"}})
    check("LABELS is not an operation name", "LABELS" not in targets)
    check(
        "LABELS withholds a label from EVERY targeted operation",
        "TYPE_TEXT" not in targets and set(targets["CLICK"]) == {"3"},
        str({op: sorted(group) for op, group in targets.items()}),
    )
    targets = targets_of(focused, {"CLICK": {"e1"}})
    check(
        "a per-operation exclusion is still per operation",
        "1" not in targets["CLICK"] and "1" in targets["TYPE_TEXT"],
    )


def request_body() -> None:
    controls = [
        control(1, "Search", editable=True, section="search"),
        control(2, "Account", section="header", opens="menu"),
        control(3, "Plain"),
    ]
    snap = snapshot(controls)
    targets = targets_of(snap)
    body = jev_._request("m", "goal", snap, [], targets, jev_._offered(snap, targets))
    elements = {e["index"]: e for e in body["state"]["elements"]}
    check(
        "section and opens are in the elements list",
        elements["2"].get("section") == "header" and elements["2"].get("opens") == "menu",
    )
    check(
        "an absent section or opens is not sent",
        "section" not in elements["3"] and "opens" not in elements["1"],
    )
    criteria = body["questions"]["click_target"]["criteria"]
    check(
        "section and opens are in each target criterion",
        criteria["2"].get("section") == "header" and criteria["2"].get("opens") == "menu",
    )
    typed = body["questions"]["type_text_target"]["criteria"]["1"]
    check("the TYPE_TEXT criterion carries section too", typed.get("section") == "search")
    check("a plain DomControl is still described", "section" not in criteria["3"])
    check(
        "the whole page text is sent when not narrowed", len(body["state"]["page"]["text"]) == 5000
    )
    rules = body["questions"]["operation"]["instructions"]["rules"]
    for phrase in (
        "once per item",
        "choose ENTER when neither exists",
        "set a quantity to that number in one action",
        "BACK undoes work",
        "the browser's Back button",
        "sits in the footer or nav",
        "Recent WAIT\nactions are not evidence of loading",
        "NEVER complete a purchase",
    ):
        check(f"rules say {phrase.splitlines()[0]!r}", phrase in rules)
    check("rules never name SELECT, which is not offered", "SELECT" not in rules)


def progress_limit() -> None:
    history = [{"action": f"a{i}", "kind": "CLICK", "page_changed": True} for i in range(100)]
    kept = progress(history)
    check("progress stops at 60 by default", len(kept) == 60, str(len(kept)))
    check("the OLDEST completed work is what goes", kept[0]["action"] == "a40")
    check("the newest step survives", kept[-1]["action"] == "a99")
    check("limit is a parameter", len(progress(history, limit=20)) == 20)
    quiet = [{"action": f"a{i}", "kind": "WAIT", "page_changed": False} for i in range(100)]
    check("unchanged older steps are still dropped first", len(progress(quiet)) == 20)
    check("a short history is sent whole", len(progress(history[:5])) == 5)


def oversized_retry() -> None:
    controls = [control(i, f"c{i}") for i in range(1, 201)]
    snap = snapshot(controls)
    history = [{"action": f"a{i}", "kind": "CLICK", "page_changed": True} for i in range(50)]
    sizes: list[int] = []

    def refuse_until(limit: int, marker: str) -> Callable[[Mapping[str, Any]], str]:
        def script(body: Mapping[str, Any]) -> str:
            sizes.append(len(body["state"]["elements"]))
            if sizes[-1] > limit:
                raise jev_._Oversized(f"Jev returned HTTP 400: {marker}")
            return "DONE"

        return script

    policy = ScriptedPolicy(StubWriter(), refuse_until(50, "max_tokens_exceeded"))
    decision = policy.decide("goal", snap, history)
    check("an oversized page halves until it is answered", sizes == [200, 100, 50], str(sizes))
    check("the narrowed question is answered", decision.operation == "DONE")
    last = policy.bodies[-1]
    check("a narrowed question cuts page text to 1500", len(last["state"]["page"]["text"]) == 1500)
    check("a narrowed question cuts progress to 20", len(last["state"]["recent_actions"]) == 20)
    check(
        "a narrowed question offers only the controls it shows",
        set(last["questions"]["click_target"]["criteria"]) == {str(i) for i in range(1, 51)},
    )
    check(
        "the first question was not narrowed",
        len(policy.bodies[0]["state"]["recent_actions"]) == 50,
    )
    check("a refused ask is not charged, an answered one is", policy.total_usage().calls == 1)
    policy.close()

    sizes.clear()
    policy = ScriptedPolicy(StubWriter(), refuse_until(0, "Too many choices"))
    try:
        policy.decide("goal", snap, history)
        raised = False
    except ProviderError:
        raised = True
    check("the retry stops at the floor of 20 and raises", raised and sizes == [200, 100, 50, 25])
    policy.close()

    sizes.clear()

    def other_failure(body: Mapping[str, Any]) -> str:
        sizes.append(len(body["state"]["elements"]))
        raise ProviderError("Jev returned HTTP 401; no action was performed: bad key")

    policy = ScriptedPolicy(StubWriter(), other_failure)
    try:
        policy.decide("goal", snap, history)
        raised = False
    except ProviderError:
        raised = True
    check("any other refusal is raised at once, not narrowed", raised and sizes == [200])
    policy.close()

    class Reply:
        status_code, is_error, text = 400, True, '{"error": "x' + "y" * 400 + ' Too many choices"}'

    class Session:
        def post(self, *args: Any, **kwargs: Any) -> Reply:
            return Reply()

    real = JevPolicy(StubWriter(), api_key="check-script-key")
    real._session = Session()  # type: ignore[assignment]
    try:
        real._post({})
        kind = None
    except ProviderError as exc:
        kind = type(exc)
    check("_post reads the marker past the 240 shown characters", kind is jev_._Oversized)
    Reply.text = '{"error": "invalid api key"}'
    try:
        real._post({})
        kind = None
    except ProviderError as exc:
        kind = type(exc)
    check("_post leaves every other error a plain ProviderError", kind is ProviderError)
    real._session = None


def speculation() -> None:
    one_field = snapshot([control(1, "Search", editable=True), control(2, "Go")])
    two_fields = snapshot([control(1, "From", editable=True), control(2, "To", editable=True)])

    writer = StubWriter("laptops", delay=0.2)
    policy = ScriptedPolicy(writer, lambda body: "TYPE_TEXT")
    started = time.perf_counter()
    decision = policy.decide("goal", one_field, [])
    check(
        "a matching speculation is reused: ONE writer call",
        writer.calls == ["e1"],
        str(writer.calls),
    )
    check("and it ran in the worker thread", writer.threads[0].startswith("jev-text"))
    check("its value is what is typed", decision.text == "laptops" and decision.element_id == "e1")
    check(
        "latency is wall time, not the two added up",
        decision.latency_ms < 380 and decision.policy_ms < decision.latency_ms,
        f"latency_ms={decision.latency_ms:.0f} policy_ms={decision.policy_ms:.0f}",
    )
    check("decide did not take two writer delays", time.perf_counter() - started < 0.38)
    usage = policy.total_usage()
    check(
        "policy and writer are both charged, once each",
        usage.calls == 2 and usage.input_tokens == 110,
    )
    policy.close()

    writer = StubWriter("laptops", delay=0.2)
    policy = ScriptedPolicy(writer, lambda body: "CLICK")
    decision = policy.decide("goal", one_field, [])
    check("a mismatch discards it: CLICK carries no text", decision.text is None)
    check("a discarded speculation was still started", writer.calls == ["e1"])
    policy.close()
    check(
        "a discarded speculation is still CHARGED once it returns, in tokens and not as a call",
        policy.total_usage().calls == 1
        and policy.total_usage().input_tokens == 110
        and policy.discarded_calls() == 1,
        str(policy.total_usage()),
    )
    check("close() shuts the worker down", policy._workers is None)
    policy.close()

    writer = StubWriter("x")
    policy = ScriptedPolicy(writer, lambda body: "TYPE_TEXT")
    policy.decide("goal", two_fields, [])
    check("two fields: nothing speculated, one direct call", writer.threads == ["MainThread"])
    policy.decide(
        "goal", one_field, [{"kind": "TYPE_TEXT", "action": "typed", "page_changed": True}]
    )
    check("right after a TYPE_TEXT: nothing speculated", writer.threads == ["MainThread"] * 2)
    policy.decide("goal", one_field, [{"kind": "CLICK", "action": "clicked", "page_changed": True}])
    check("after any other step it is", writer.threads[-1].startswith("jev-text"))
    policy.close()

    decline = NoFieldValue("declined")
    writer = StubWriter(decline)
    seen: list[list[str]] = []

    def script(body: Mapping[str, Any]) -> str:
        seen.append(sorted(body["questions"]))
        return "TYPE_TEXT" if "type_text_target" in body["questions"] else "CLICK"

    policy = ScriptedPolicy(writer, script)
    decision = policy.decide("goal", one_field, [])
    check(
        "a decline raised in the thread withholds the field and asks again",
        decision.operation == "CLICK" and "type_text_target" not in seen[-1] and len(seen) == 2,
        str(seen),
    )
    check("the decline was given the field's id", decline.element_id == "e1")
    check("the declined call is charged", policy.total_usage().calls == 3)
    policy.close()

    writer = StubWriter(ProviderError("no usable value"))
    policy = ScriptedPolicy(writer, lambda body: "TYPE_TEXT")
    try:
        policy.decide("goal", one_field, [])
        raised = False
    except NoFieldValue:
        raised = False
    except ProviderError:
        raised = True
    check("a writer failure raised in the thread still fails the step", raised)
    policy.close()


def writer_context() -> None:
    check("the text writer is shown 2000 characters of page text", _PAGE_TEXT_SHOWN == 2000)


def main() -> int:
    for group in (
        offer_and_exclusions,
        request_body,
        progress_limit,
        oversized_retry,
        speculation,
        writer_context,
    ):
        group()
    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
