"""Pure-logic checks of three upstream fixes in the Jev policy (``cbf517a``..``0da4053``).

Run with ``uv run python scripts/check_l_policy.py``; prints one line per check and exits
non-zero if any failed. NO network and no credential: the policy's wire is scripted, the
HTTP session is a stub that raises what a dropped connection raises, and the text writer
is a stub that dawdles and charges like a real one. So nothing here says the policy
CHOOSES well or that a real connection drops this way - only that what is sent, withheld,
retried, cancelled and charged is what the code claims. A live run is still the proof.

``context`` is another module's half of this port (``perception/dom.py``). The controls
below carry it through a subclass when ``DomControl`` does not have it yet, which is also
the check that ``llm/jev_.py`` and ``llm/openai_.py`` read it tolerantly.
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from skillweaver.contracts import Box, ElementKind, Usage
from skillweaver.errors import ProviderError
from skillweaver.llm import jev_, openai_
from skillweaver.llm.jev_ import JevPolicy, name_of, targets_of
from skillweaver.llm.openai_ import OpenAITextWriter
from skillweaver.llm.usage import UsageMeter
from skillweaver.perception.dom import DomControl, DomSnapshot

if "context" in DomControl.__dataclass_fields__:
    Control = DomControl
else:

    @dataclasses.dataclass(frozen=True)
    class Control(DomControl):  # type: ignore[no-redef]
        context: str | None = None


FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail and not ok else ''}")
    if not ok:
        FAILED.append(name)


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


def snapshot(controls: Sequence[DomControl]) -> DomSnapshot:
    return DomSnapshot(
        url="https://example.test/",
        title="t",
        text="page text",
        controls=tuple(controls),
        viewport=Box(0, 0, 800, 600),
    )


def answer(choice: str, keys: Sequence[str]) -> dict[str, Any]:
    rest = 0.1 / max(len(keys) - 1, 1)
    probabilities = {key: (0.9 if key == choice else rest) for key in keys}
    if len(keys) == 1:
        probabilities = {choice: 1.0}
    return {"choice": choice, "confidence": 0.9, "probabilities": probabilities}


class StubWriter:
    """A text writer whose successive calls take ``delays`` seconds (the last repeats)."""

    def __init__(self, *delays: float, charge: bool = True) -> None:
        self.delays = list(delays) or [0.0]
        self.charge = charge
        self.calls: list[str] = []
        self.threads: list[str] = []
        self.meter = UsageMeter()
        self.lock = threading.Lock()

    def write(self, goal: str, field: DomControl, snap: DomSnapshot, history: Any) -> str:
        with self.lock:
            delay = self.delays[min(len(self.calls), len(self.delays) - 1)]
            self.calls.append(field.element_id)
            self.threads.append(threading.current_thread().name)
        time.sleep(delay)
        if self.charge:
            self.meter.add(Usage(10, 2, 1, 0.001))
        return "typed"

    def total_usage(self) -> Usage:
        return self.meter.total()


class ScriptedPolicy(JevPolicy):
    """``JevPolicy`` with the wire replaced: ``script`` names the operation to answer,
    and each round trip takes ``delay`` seconds."""

    __slots__ = ("bodies", "delay", "script")

    def __init__(
        self, writer: Any, script: Callable[[Mapping[str, Any]], str], delay: float = 0.0
    ) -> None:
        super().__init__(writer, api_key="check-script-key")
        self.script, self.delay = script, delay
        self.bodies: list[Mapping[str, Any]] = []

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.bodies.append(body)
        time.sleep(self.delay)
        questions = body["questions"]
        answers = {"operation": answer(self.script(body), list(questions["operation"]["criteria"]))}
        for key, question in questions.items():
            if key != "operation":
                keys = list(question["criteria"])
                answers[key] = answer(keys[0], keys)
        return {"answers": answers, "usage": {"input_tokens": 100, "output_tokens": 5}}


# -- fix 1: context on the wire, and LABELS matched by name -------------------------------

ITEMS = ("Bread", "Milk", "Eggs", "Rice", "Tea", "Salt", "Oats")


def context_wire() -> None:
    adds = [control(i + 1, "Add item to cart", context=item) for i, item in enumerate(ITEMS)]
    plain = control(8, "Checkout")
    snap = snapshot([*adds, plain])

    check(
        "a shared label is named label - context, em dash with spaces",
        name_of(adds[1]) == "Add item to cart — Milk",
        name_of(adds[1]),
    )
    check("a control with no context is named by its label alone", name_of(plain) == "Checkout")
    bare = DomControl(
        index=9,
        element_id="e9",
        role="button",
        name="Bare",
        kind=ElementKind.button,
        box=Box(0, 0, 1, 1),
    )
    check("a control that has no context attribute at all is read tolerantly", bool(name_of(bare)))

    policy = ScriptedPolicy(StubWriter(), lambda body: "CLICK")
    decision = policy.decide("add milk", snap, [])
    body = policy.bodies[0]
    elements = body["state"]["elements"]
    check(
        "context is in the elements list, for the controls that have one",
        [e.get("context") for e in elements] == [*ITEMS, None] and "context" not in elements[-1],
        str([e.get("context") for e in elements]),
    )
    criteria = body["questions"]["click_target"]["criteria"]
    check(
        "context is in every target criterion that has one",
        [criteria[str(i + 1)].get("context") for i in range(7)] == list(ITEMS)
        and "context" not in criteria["8"],
    )
    check(
        "the element's label stays the bare label",
        elements[0]["label"] == "Add item to cart"
        and criteria["1"]["element"] == "[1] Add item to cart",
    )
    check(
        "the decision's why - the history's action - quotes the NAME",
        "Add item to cart — Bread" in decision.why,
        decision.why,
    )
    policy.close()

    pruned = targets_of(snap, {"LABELS": {name_of(adds[2])}})
    check(
        "LABELS by name prunes ONE of seven same-label controls",
        sorted(pruned["CLICK"]) == ["1", "2", "4", "5", "6", "7", "8"],
        str(sorted(pruned["CLICK"])),
    )
    check(
        "the bare label no longer prunes controls that carry a context",
        len(targets_of(snap, {"LABELS": {"Add item to cart"}})["CLICK"]) == 8,
    )
    check(
        "and still prunes a control whose name IS its label",
        "8" not in targets_of(snap, {"LABELS": {"Checkout"}})["CLICK"],
    )

    seen: list[Any] = []

    class Capturing(OpenAITextWriter):
        __slots__ = ()

        def _complete(self, system: str, content: str, model: str) -> str | None:
            seen.append(content)
            return '{"text": "2"}'

    writer = Capturing(api_key="check-script-key")
    writer.write("goal", control(1, "Quantity", editable=True, context="Milk"), snap, [])
    writer.write("goal", control(2, "Search", editable=True), snap, [])
    check(
        "the text writer is shown a field's context, and only when there is one",
        '"context": "Milk"' in seen[0] and "context" not in seen[1],
    )


# -- fix 3: a dropped connection is retried twice ------------------------------------------


class DroppingSession:
    """An ``httpx.Client`` stand-in whose first ``failures`` posts raise ``error``."""

    def __init__(self, failures: int, error: Exception, reply: Mapping[str, Any]) -> None:
        self.failures, self.error, self.reply, self.posts = failures, error, reply, 0

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.posts += 1
        if self.posts <= self.failures:
            raise self.error
        return httpx.Response(200, json=dict(self.reply), request=httpx.Request("POST", url))

    def close(self) -> None:
        pass


def transport_retry() -> None:
    naps: list[float] = []
    real_sleep = time.sleep
    jev_.time.sleep = naps.append  # type: ignore[assignment]  # same module object as time
    try:
        writer = StubWriter()
        policy = JevPolicy(writer, api_key="check-script-key")
        policy._session = DroppingSession(99, httpx.RemoteProtocolError("server disconnected"), {})
        try:
            policy.decide("goal", snapshot([control(1, "Go")]), [])
            message = ""
        except ProviderError as exc:
            message = str(exc)
        check("a transport error is tried three times in all", policy._session.posts == 3)
        check("with a 0.25 * 2^n backoff", naps == [0.25, 0.5], str(naps))
        check(
            "then raised, naming the exception type and that nothing executed",
            "RemoteProtocolError" in message and "no action was performed" in message,
            message,
        )
        check("and nothing was charged for it", policy.total_usage().calls == 0)

        for error in (
            httpx.ConnectError("refused"),
            httpx.ReadTimeout("slow"),
            httpx.PoolTimeout("pool"),
        ):
            naps.clear()
            reply = {
                "answers": {"operation": answer("DONE", ["CLICK", "WAIT", "DONE", "BLOCKED"])},
                "usage": {},
            }
            policy._session = DroppingSession(2, error, reply)
            decision = policy.decide("goal", snapshot([control(1, "Go")]), [])
            check(
                f"{type(error).__name__} twice, then an answer: the step stands",
                decision.operation == "DONE" and policy._session.posts == 3 and len(naps) == 2,
            )
        policy._session = None
        policy.close()

        naps.clear()
        text = OpenAITextWriter(api_key="check-script-key")
        text._session = DroppingSession(99, httpx.ReadError("reset"), {})
        try:
            text.write("goal", control(1, "Search", editable=True), snapshot([]), [])
            message = ""
        except ProviderError as exc:
            message = str(exc)
        check(
            "the text writer retries the same way and says nothing was typed",
            text._session.posts == 3
            and naps == [0.25, 0.5]
            and "ReadError" in message
            and "nothing was typed" in message,
            f"{naps} {message}",
        )
        check("the two modules share one time module", openai_.time is jev_.time)
    finally:
        jev_.time.sleep = real_sleep  # type: ignore[assignment]


# -- fix 5: speculation ---------------------------------------------------------------------

ONE_FIELD = [control(1, "Search", editable=True), control(2, "Go")]
CLICKED = {"kind": "CLICK", "action": "clicked", "page_changed": True}


def speculation() -> None:
    snap = snapshot(ONE_FIELD)

    # A guess that is already RUNNING cannot be cancelled. The next step needs a value:
    # it must be written BESIDE the policy call, not after it and not behind the guess.
    writer = StubWriter(1.0, 0.2)
    policy = ScriptedPolicy(
        writer, lambda body: "CLICK" if len(policy.bodies) == 1 else "TYPE_TEXT", delay=0.15
    )
    policy.decide("goal", snap, [])
    started = time.perf_counter()
    decision = policy.decide("goal", snap, [CLICKED])
    took = time.perf_counter() - started
    check("a needed value is typed", decision.text == "typed")
    check(
        "it is not blocked by a running guess: wall time is about the writer alone",
        took < 0.3,
        f"{took:.2f}s (one worker: 0.15 + 0.2 written after the policy; queued: 1.0+)",
    )
    check(
        "because the second worker ran it beside the policy call",
        writer.threads[1].startswith("jev-text") and writer.threads[0] != "MainThread",
        str(writer.threads),
    )
    check(
        "a guess still running is not yet counted as discarded, so calls never dip",
        policy.discarded_calls() == 0 and policy.total_usage().calls == 3,
        str(policy.total_usage()),
    )
    policy.close()
    usage = policy.total_usage()
    check(
        "close() waits the running guess out: its tokens and dollars are charged",
        usage.input_tokens == 220 and abs(usage.cost_usd - 0.002) < 1e-9,
        str(usage),
    )
    check(
        "but it is not a CALL against the run's budget",
        usage.calls == 3 and policy.discarded_calls() == 1,
        f"{usage} discarded={policy.discarded_calls()}",
    )
    check("close() shut the workers down and is idempotent", policy._workers is None)
    policy.close()

    # Both workers busy with abandoned guesses: the third is QUEUED, and is cancelled the
    # moment its move is chosen - never run, never paid for.
    writer = StubWriter(0.5)
    policy = ScriptedPolicy(writer, lambda body: "CLICK")
    for _ in range(3):
        policy.decide("goal", snap, [CLICKED])
    check("the outstanding guess is dropped when the move is chosen", policy._speculation is None)
    policy.close()
    check(
        "an obsolete guess still queued is cancelled: two writer calls, not three",
        len(writer.calls) == 2,
        str(writer.calls),
    )
    usage = policy.total_usage()
    check(
        "usage under two workers: 3 policy calls, 2 discarded guesses charged in tokens only",
        usage.calls == 3 and usage.input_tokens == 320 and policy.discarded_calls() == 2,
        f"{usage} discarded={policy.discarded_calls()}",
    )

    # Both workers busy AND a value needed: the queued guess is cancelled and the value is
    # written directly, on this thread.
    writer = StubWriter(0.6, 0.6, 0.05)
    policy = ScriptedPolicy(
        writer, lambda body: "TYPE_TEXT" if len(policy.bodies) == 3 else "CLICK"
    )
    policy.decide("goal", snap, [CLICKED])
    policy.decide("goal", snap, [CLICKED])
    started = time.perf_counter()
    decision = policy.decide("goal", snap, [CLICKED])
    took = time.perf_counter() - started
    check(
        "with both workers busy a needed value is written directly, not queued",
        decision.text == "typed" and writer.threads[-1] == "MainThread" and took < 0.3,
        f"{took:.2f}s {writer.threads}",
    )
    policy.close()
    check(
        "and that direct call IS a call; the two guesses are not",
        policy.total_usage().calls == 4 and policy.discarded_calls() == 2,
        str(policy.total_usage()),
    )

    # A reused guess is a real call, charged as one.
    writer = StubWriter(0.1)
    policy = ScriptedPolicy(writer, lambda body: "TYPE_TEXT")
    policy.decide("goal", snap, [])
    policy.close()
    check(
        "a guess that was READ counts as a call",
        policy.total_usage().calls == 2 and policy.discarded_calls() == 0,
    )

    # A writer metered elsewhere reports nothing; its guesses must not eat policy calls.
    writer = StubWriter(0.0, charge=False)
    policy = ScriptedPolicy(writer, lambda body: "CLICK")
    policy.decide("goal", snap, [])
    policy.decide("goal", snap, [CLICKED])
    policy.close()
    check(
        "a writer that reports nothing cannot take the policy's own calls off the count",
        policy.total_usage().calls == 2,
        str(policy.total_usage()),
    )


def main() -> int:
    for group in (context_wire, transport_retry, speculation):
        group()
    print(f"FAILED: {', '.join(FAILED)}" if FAILED else "all checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
