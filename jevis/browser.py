"""Observed actions through Browser Harness; one CDP session, no per-step subprocess."""

import hashlib
import json
import sys
import time
from pathlib import Path

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"

class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class CoveredTarget(StalePage):
    """The target is present but something is drawn over it, so the click was refused.

    Distinct from ordinary staleness: re-deciding on the same page selects the same covered
    element again, so the caller must stop offering it instead of observing again.
    """


# Event-driven quiesce: DOM mutations and completed network fetches both reset the quiet window.
# Resolves once the page has been still for quiet ms, capped at cap ms. Animations are ignored
# (no attribute observation), matching "animation alone does not force another prediction".
SETTLE = """(([quiet, cap]) => new Promise(resolve => {
  const started = performance.now();
  let last = performance.now();
  const observer = new MutationObserver(() => { last = performance.now(); });
  observer.observe(document.documentElement, {subtree: true, childList: true, characterData: true});
  const lastResource = () => {
    const entries = performance.getEntriesByType('resource');
    return entries.length ? entries[entries.length - 1].responseEnd : 0;
  };
  const check = () => {
    const now = performance.now();
    if (now - Math.max(last, lastResource()) >= quiet || now - started >= cap) {
      observer.disconnect();
      resolve(Math.round(now - started));
    } else setTimeout(check, 40);
  };
  setTimeout(check, 40);
}))(%s)"""


class Browser:
    def __init__(self, url):
        ensure_daemon()
        self.target = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
        self.session = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
        # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        self.frame_misses, self.frame_skip = 0, 0
        started = time.perf_counter()
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        # The tab still holds about:blank for a moment after Page.navigate, and that document is
        # already "complete" — polling readiness alone returns instantly and the first decision is
        # made on a blank page. Require the new document to have replaced it.
        ready = "(() => location.href !== 'about:blank' && document.readyState === 'complete')()"
        while time.monotonic() < deadline:
            if self.evaluate(ready):
                break
            time.sleep(0.02)
        # The first load happens before the run clock starts, so record it or it is invisible.
        self.load_ms = round((time.perf_counter() - started) * 1000)

    def call(self, method, **params):
        return cdp(method, session_id=self.session, **params)

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def capture(self):
        """A frame for the inspector, or None. Never re-reads the page.

        Some pages block the compositor for long stretches, so a frame costs the full timeout and
        returns nothing. Back off after repeated misses and retry periodically: the agent must not
        pay for frames it cannot get.
        """
        if self.frame_skip > 0:
            self.frame_skip -= 1
            return None
        data = capture_frame(self.session)
        self.frame_misses = 0 if data else self.frame_misses + 1
        if self.frame_misses >= 3:
            self.frame_misses, self.frame_skip = 0, 5
        return data

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass
        for attempt in range(10):
            try:
                return browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot}
                )
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(0.02)
        raise StalePage("Page did not settle")

    def fresh(self, page, action=None, page_level=True):
        kind = action["kind"] if action else None
        targeted = kind in {"click", "select", "fill"}
        if not page_level:
            # Retry mode: the caller re-observed and proved the page is identical to the decision's
            # basis, so only the target's own identity and context must still hold. A signal wider
            # than that observation must not veto a decision the observation still supports.
            # Geometry and occlusion are re-checked atomically at input regardless.
            if not targeted:
                return True
            node = action["node"]
            if type(node) is not int:
                return False
            guard = self.evaluate(
                f"(() => {{ const c=window.__jevFast; return c ? c.guard(c.nodes.get({node})) : null; }})()"
            )
            return guard is not None and guard == page["guards"].get(str(node))
        if targeted:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None, page_level=True):
        if not self.fresh(page, action, page_level=page_level):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text})
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def settle(self, quiet_ms=150, cap_ms=1500):
        try:
            self.call(
                "Runtime.evaluate",
                expression=SETTLE % json.dumps([quiet_ms, cap_ms]),
                awaitPromise=True,
                returnByValue=True,
            )
        except (RuntimeError, TimeoutError):
            # A navigating document interrupts evaluation, and a frozen main thread (long task or
            # native dialog) outlives the transport timeout. Settling is advisory: never fail a
            # step that already executed. observe() retries its own reads.
            pass

    def show(self):
        """Raise the agent's background tab so the user can watch the real window."""
        try:
            cdp("Target.activateTarget", targetId=self.target)
        except (RuntimeError, OSError):
            pass  # Focus is a convenience; a tab that cannot be raised must not end the run.

    def close(self):
        if self.target:
            target, self.target = self.target, None
            try:
                cdp("Target.closeTarget", targetId=target)
            except (RuntimeError, OSError):
                pass  # The tab or browser is already gone; reset must still proceed.


def capture_frame(session):
    """The frame dominates observation cost on heavy pages: the default encoder measured ~1,000 ms
    against ~40 ms for the speed-optimised one, and a healthy capture 28-56 ms. A blocked compositor
    never delivers at all, so bound the wait tightly and let a lost frame cost nothing."""
    try:
        return cdp(
            "Page.captureScreenshot",
            session_id=session,
            _response_timeout=0.5,
            format="jpeg",
            quality=72,
            optimizeForSpeed=True,
        )["data"]
    except (RuntimeError, TimeoutError):
        return None


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]

    def call(method, **params):
        return cdp(method, session_id=session, **params)

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            # Aim the wheel at the region the snapshot found scrollable, not a fixed point.
            call(
                "Input.dispatchMouseEvent",
                type="mouseWheel",
                x=action.get("x", 550),
                y=action.get("y", 650),
                deltaX=0,
                deltaY=action["delta"],
            )
        elif kind == "back":
            # A fixed, code-owned navigation. The model selects the operation, never its content.
            call("Runtime.evaluate", expression="history.back()")
        elif kind == "enter":
            # A fixed key, sent to whatever holds focus. The model chooses the operation, not the key.
            for event, extra in (("keyDown", {"text": "\r"}), ("keyUp", {})):
                call(
                    "Input.dispatchKeyEvent",
                    type=event,
                    key="Enter",
                    code="Enter",
                    windowsVirtualKeyCode=13,
                    nativeVirtualKeyCode=13,
                    **extra,
                )
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise CoveredTarget("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = capture_frame(session)
    return info
