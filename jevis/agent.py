"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .browser import Browser, CoveredTarget, StalePage
from .model import NoFieldValue, action_name, action_space, choose, field_context, field_text
from .questions import MAX_STEPS


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.speculation = None
        self.workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-text")
        self.stale_streak, self.stale_at, self.stale_known = 0, None, 0
        self.covered = {}
        self.done_seen = False
        # Controls whose label promises nothing this run: a label like "Make 2 required selections"
        # can never advance the goal, and it is re-offered after any reopen because the page
        # fingerprint resets. The label changes once the control becomes usable, so it returns then.
        self.inert = {}
        # Which action was already executed from each observed page state. Returning to an identical
        # state proves that action did not advance the goal, even though it changed the page: it is
        # what turns open/close into an endless oscillation. Re-offering it repeats the cycle.
        self.taken = {}
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
            # A bot check or splash can be the first complete document and replace itself seconds
            # later. Without this the run's first and only decision is made on an empty page.
            for _ in range(4):
                if not self.blank(page):
                    break
                self.browser.settle(quiet_ms=200, cap_ms=1200)
                page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            model_ms=0,
            load_ms=0,
            frame_ms=0,
            initial_load_ms=self.browser.load_ms,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            if page.get("screenshot"):
                (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def speculate(self, state):
        """Start the only typeable field's value beside the work that follows.

        The helper is the long pole on a typing step (~1.3 s against ~0.3 s for the policy), and its
        value does not depend on which operation wins. Starting it as soon as the page settles
        overlaps it with the frame capture and the next decision instead of following them.
        """
        self.speculation = None
        fields = [a for a in state["page"]["actions"] if a["kind"] == "fill"]
        # Speculate whenever exactly one field can be typed into, even one holding a value: a search
        # box keeps the previous query, and a multi-item run types over it for every item. Gating on
        # an empty field measured 1.4-5.4 s of unhidden helper latency per item against ~0.5 s here.
        if len(fields) != 1 or state["status"] in {"done", "blocked"}:
            return
        if state["history"] and state["history"][-1]["kind"] == "fill":
            return  # A query was just typed; the next action submits it rather than typing again.
        context = field_context(state["goal"], fields[0], state["page"], state["history"])
        self.speculation = (context, self.workers.submit(field_text, context))

    @staticmethod
    def cycling(history, window=8, repeats=6):
        """Labels caught in a repeating cycle, judged on the action sequence alone.

        The fingerprint-keyed memory cannot see these. A page that varies between laps — search
        suggestions, timestamps, a re-rendered overlay — looks new every time, so two or three
        actions can alternate until the budget runs out even though each one "changes the page".
        Only cycles of two or more are counted: a single action repeated (Scroll down, Next page,
        Increase quantity) is usually real progress, and a dead one is already handled elsewhere.
        """
        labels = [h["action"] for h in history[-window:]]
        if len(labels) < repeats:
            return set()
        distinct = set(labels)
        # Churn, not a strict alternation: a real loop doubles back on itself (open, go, open, open,
        # go), so requiring an exact repeating pattern misses it. A handful of labels filling the
        # whole window is the signal. Two or more, because one action repeating — "Increase quantity
        # by 1", "Load more" — is usually progress.
        return distinct if 2 <= len(distinct) <= 3 else set()

    @staticmethod
    def blank(page):
        """An interstitial with nothing to act on: a bot check or challenge shell that self-replaces.

        Deciding here wastes the decision — the only offered operations are WAIT and BLOCKED, and a
        BLOCKED ends the run on a page that was about to become the real site.
        """
        return not page["text"].strip() and not any(
            a["kind"] in {"click", "fill", "select"} for a in page["actions"]
        )

    def observed(self, screenshot=None):
        """Observe outside the act path, attributing the wait to the site rather than the models."""
        started = time.perf_counter()
        browser = self.state["browser"]
        shot = self.screenshots if screenshot is None else screenshot
        page = browser.observe(screenshot=shot)
        for _ in range(4):
            if not self.blank(page):
                break
            browser.settle(quiet_ms=200, cap_ms=1200)
            page = browser.observe(screenshot=shot)
        self.state["load_ms"] += round((time.perf_counter() - started) * 1000)
        return page

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                # A decision discarded against an unchanged page cannot become executable by
                # repeating: report it instead of spending the whole budget on the same page.
                # Each newly rejected target shrinks the action space, so the next decision is a
                # genuinely different one. Only count failures that taught the run nothing.
                fingerprint = state["page"]["fingerprint"]
                learned = len(self.covered.get(fingerprint, ()))
                stuck = fingerprint == self.stale_at and learned <= self.stale_known
                self.stale_streak = self.stale_streak + 1 if stuck else 1
                self.stale_at, self.stale_known = fingerprint, learned
                if self.stale_streak >= 8:
                    state["status"] = "blocked"
                    raise ValueError("Stopped: the page keeps rejecting decisions without changing.")
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = self.observed()
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = self.observed()
                # The speculation was built on the page we just replaced, so its input no longer
                # matches and it could not be reused. Start one for the page we will decide on.
                self.speculate(state)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            ask = (
                state["page"],
                state["goal"],
                state["history"],
                self.covered.get(state["page"]["fingerprint"], set())
                | self.taken.get(state["page"]["fingerprint"], set()),
                {label for label, misses in self.inert.items() if misses >= 3}
                | self.cycling(state["history"]),
            )
            if self.speculation is None:
                self.speculate(state)
            try:
                state["decision"] = choose(*ask)
            except ValueError as invalid:
                # A rejected answer executed nothing, so asking again is not a mutation retry.
                # Large, repetitive action spaces provoke this often enough to end runs.
                if "Invalid TypeSafe" not in str(invalid):
                    raise
                state["decision"] = choose(*ask)
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                if selected == "DONE" and not self.done_seen:
                    # DONE is judged on the snapshot the last action produced, which is often taken
                    # before the effect lands: a cart badge, a navigation, an availability notice.
                    # Require the claim to survive one fresh look, so success is confirmed against
                    # the settled page rather than the optimistic one.
                    self.done_seen = True
                    state["browser"].settle(quiet_ms=150, cap_ms=1200)
                    state["page"] = self.observed()
                    state["status"] = "ready"
                    state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                    return self.snapshot()
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper, overlapped = None, None, False
            if action["kind"] == "fill":
                # Check the target, not the whole document: unrelated churn elsewhere on a live
                # page must not veto typing into a field that is still the same field.
                if not state["browser"].fresh(page, action):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    try:
                        # Reuse the speculative value only when its entire input is identical,
                        # exactly as a stale retry does.
                        if self.speculation and self.speculation[0] == context:
                            text, helper = self.speculation[1].result()
                            overlapped = True
                        else:
                            text, helper = field_text(context)
                    except NoFieldValue as decline:
                        # A declined field is unusable on this page, not a reason to end the run:
                        # drop it from the next decision and let the policy choose differently.
                        self.covered.setdefault(page["fingerprint"], set()).add(selected)
                        raise StalePage(str(decline)) from None
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Re-entering a field's existing value cannot change the page. Execute nothing: the
            # recorded no-change entry drops this target from the next decision's action space.
            noop = action["kind"] == "fill" and text == action.get("value")
            # Browser.act checks freshness immediately before input, including after text generation.
            try:
                if not noop:
                    state["browser"].act(action, page, text=text)
            except CoveredTarget:
                # Nothing executed. Remember the refusal against this exact page so the next
                # decision cannot choose it again, then let the caller re-decide.
                self.covered.setdefault(page["fingerprint"], set()).add(selected)
                raise
            except StalePage:
                # Freshness can fail on churn the observation cannot see. If a new observation is
                # identical to the decision's basis, the decision still applies: retry once against
                # the target's own guard. Nothing was executed, so this is not a mutation retry.
                current = self.observed(screenshot=False)
                if action["kind"] not in {"click", "select"} or current["fingerprint"] != page["fingerprint"]:
                    raise
                state["browser"].act(action, current, text=text, page_level=False)
            self.pending_text = None
            self.speculation = None  # Its page is gone; the settled page below gets a fresh one.
            self.stale_streak = 0
            self.done_seen = False  # A new action means the next DONE claim is about a new page.
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    # The distinguishing name, so a history of seven "Add item to cart" entries
                    # says which item each one was — and so the loop guards below can tell them
                    # apart instead of banning every control that shares a label.
                    "action": action_name(action),
                    "kind": action["kind"],
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "page_fingerprint": page["fingerprint"],
                    # Time the models owned, against time the site owned. Filled after observing.
                    # A speculative helper call ran beside the decision, so counting it would make
                    # the parts exceed the whole.
                    "model_ms": decision["latency_ms"] + (0 if overlapped or not helper else helper["latency_ms"]),
                    "load_ms": None,
                    "frame_ms": None,
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                }
            )
            # Observe without the screenshot first: the frame is the expensive part on heavy pages,
            # and a mutation can land asynchronously (cart badges, toasts). Give the page a bounded
            # grace period to show the action's effect before recording "no change observed".
            load_started = time.perf_counter()
            after = page if noop else state["browser"].observe(screenshot=False)
            if not noop and action["kind"] != "wait" and after["fingerprint"] == page["fingerprint"]:
                # The effect may land asynchronously (cart badges, toasts): wait for the page to
                # go quiet, then look once more before recording "no change observed".
                state["browser"].settle(quiet_ms=120, cap_ms=1000)
                after = state["browser"].observe(screenshot=False)
            elif after["url"] != page["url"]:
                # A navigation may still be materializing content (skeleton results pages).
                # Decide on a quiet page, not on whatever happened to be rendered first.
                state["browser"].settle(quiet_ms=150, cap_ms=1500)
                after = state["browser"].observe(screenshot=False)
            elif not noop and action["kind"] != "wait":
                # The page changed in place: an overlay, dropdown or in-page result swap. Its first
                # frame is half-built — a menu measured 3 elements and no text at once, then 14
                # elements and its options a moment later — so let it finish before deciding.
                state["browser"].settle(quiet_ms=100, cap_ms=700)
                after = state["browser"].observe(screenshot=False)
            state["page"] = after
            # Record the outcome BEFORE speculating. The helper's input includes the action history,
            # so speculating while this entry still reads page_changed=None guarantees a context
            # that cannot match the one act() builds later — every speculation was discarded.
            state["history"][-1]["page_changed"] = after["fingerprint"] != page["fingerprint"]
            # Start the next value before the frame, not after it: the capture is for the inspector
            # and can cost hundreds of milliseconds that the helper can be working through.
            self.speculate(state)
            frame_ms = 0
            if self.screenshots:
                # Unchanged page, unchanged picture. Otherwise take only the frame: re-reading the
                # page would cost a second full read and invalidate the speculation above.
                frame_started = time.perf_counter()
                after["screenshot"] = (
                    page.get("screenshot")
                    if after["fingerprint"] == page["fingerprint"]
                    else state["browser"].capture() or page.get("screenshot")
                )
                frame_ms = round((time.perf_counter() - frame_started) * 1000)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Keep the inspector's frame out of "load": it is viewer overhead, not the site
            # responding, and reporting them together makes a scroll look like a page load.
            load_ms = round((time.perf_counter() - load_started) * 1000) - frame_ms
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
                load_ms=load_ms,
                frame_ms=frame_ms,
            )
            state["model_ms"] += state["history"][-1]["model_ms"]
            state["load_ms"] += load_ms
            state["frame_ms"] += frame_ms
            if action["kind"] in {"click", "select", "fill"}:
                changed = state["history"][-1]["page_changed"]
                name = action_name(action)
                self.inert[name] = 0 if changed else self.inert.get(name, 0) + 1
            elif action["kind"] == "scroll":
                # A scroll always moves the offset, so it always "changes the page" and no guard
                # built on page_changed can ever see a pointless one. Judge it by what it revealed:
                # controls that were not on screen before. Scrolling a settled panel reveals none.
                seen = {action_name(a) for a in page["actions"]}
                revealed = sum(1 for a in after["actions"] if action_name(a) not in seen)
                state["history"][-1]["revealed"] = revealed
                name = action["label"]
                self.inert[name] = 0 if revealed else self.inert.get(name, 0) + 1
            self.taken.setdefault(page["fingerprint"], set()).add(selected)
            if state["record"] and state["page"].get("screenshot"):
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            # Give up only on a page that is genuinely inert. A navigation that commits after the
            # observation window records a false "no change", so differing URLs across the streak
            # are proof of progress even when each step looked idle.
            repeated = state["history"][-4:]
            state["status"] = (
                "blocked"
                if len(repeated) == 4
                and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
                and len({h["url"] for h in repeated}) == 1
                else "ready"
            )
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.workers.shutdown(wait=False, cancel_futures=True)
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
