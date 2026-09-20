// The inspector page. Polls /api/state, posts /api/<command>, and draws what comes back.
// Everything shown is server state: this file decides nothing about the run.

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="inspector-token"]').content;
const headers = { "Content-Type": "application/json", "X-Inspector-Token": token };

let state = null;
let busy = false;
let stale = false;
let stateAt = performance.now();
let shownFrame = -1;
let frameUrl = null;
let hovered = null;
const opened = new Set();

const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
const percent = (value) => `${(value * 100).toFixed(value < 0.01 ? 1 : 0)}%`;
const seconds = (ms) => `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
const TERMINAL = ["done", "blocked", "stopped"];

// sessionStorage can throw (blocked storage, some private modes). Nothing here needs it
// to work: without it the page shows the reload banner instead of reloading itself.
const stored = {
  get(key) {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      sessionStorage.setItem(key, value);
    } catch {
      /* see above */
    }
  },
  drop(key) {
    try {
      sessionStorage.removeItem(key);
    } catch {
      /* see above */
    }
  },
};

// A 403 is nearly always this tab outliving the server it was loaded from: the token in
// its meta tag belongs to a process that has exited. The server says which check failed;
// a stale token is cured by loading the page again, so do that ONCE, keeping what was
// typed, and if that was tried seconds ago and it is still refused, stop and say so
// rather than reloading in a loop.
async function refused(response) {
  if (response.status !== 403) return false;
  let data = {};
  try {
    data = await response.clone().json();
  } catch {
    /* a bare 403 is still a refusal */
  }
  if (stale) return true;
  stale = true;
  const again = Date.now() - Number(stored.get("inspector-reloaded") || 0) < 15000;
  if (data.code === "stale-token" && !again) {
    stored.set("inspector-reloaded", String(Date.now()));
    stored.set("inspector-draft", JSON.stringify({ url: $("url").value, goal: $("goal").value }));
    location.reload();
    return true;
  }
  $("restarted-text").textContent =
    data.error || "This inspector was restarted. Reload the page to reconnect to it.";
  $("restarted").hidden = false;
  $("server-dot").classList.add("down");
  $("status").textContent = "Disconnected · reload to reconnect";
  for (const el of document.querySelectorAll("main button, main input, main select, main textarea"))
    if (el.id !== "reload") el.disabled = true;
  return true;
}

async function call(name, body = {}) {
  const response = await fetch(`/api/${name}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (await refused(response)) throw Error("This inspector was restarted - reload the page.");
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "Request failed");
  accept(data);
  return data;
}

async function poll() {
  if (stale) return;
  try {
    const response = await fetch("/api/state", { headers });
    if (await refused(response)) return;
    if (!response.ok) throw Error("state");
    accept(await response.json());
    $("server-dot").classList.remove("down");
  } catch {
    $("server-dot").classList.add("down");
    if (!state) $("status").textContent = "Cannot reach the inspector server";
  }
}

// A poll that changed nothing must not touch the DOM: every panel here is drawn with
// innerHTML, so rebuilding it once a second would reset the scroll position of the element
// list, wipe a text selection in the log and restart every hover under the pointer.
// The clock is excluded because it always moves; renderElapsed() draws it on its own timer.
let shownKey = "";
function changed(data) {
  const { elapsed_ms, spend, ...rest } = data;
  const { seconds: _seconds, ...spent } = spend || {};
  // The half-minute bucket is what keeps "recorded 17s ago" honest on an idle page.
  const key = JSON.stringify([rest, spent, Math.floor(Date.now() / 30000)]);
  if (key === shownKey) return false;
  shownKey = key;
  return true;
}

function accept(data) {
  const redraw = changed(data);
  state = data;
  stateAt = performance.now();
  if (redraw) render();
  refreshFrame();
}

// The screenshot is fetched with the token in a header and shown from a blob, so the
// secret never appears in a URL and a page on another origin cannot load the picture.
async function refreshFrame() {
  if (!state || state.frame === shownFrame || !state.frame) return;
  const wanted = state.frame;
  try {
    const response = await fetch("/api/frame", { headers });
    if (await refused(response)) return;
    if (!response.ok) {
      // A closed session keeps its frame COUNTER and has no picture: without this the
      // page asked again on every poll - 230 logged 404s on one idle tab.
      if (response.status === 404) shownFrame = wanted;
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    if (frameUrl) URL.revokeObjectURL(frameUrl);
    frameUrl = url;
    shownFrame = wanted;
    $("screenshot").src = url;
    $("screenshot").hidden = false;
    $("empty").hidden = true;
  } catch {
    /* The next poll tries again. */
  }
}

function renderElapsed() {
  const active = state && state.status !== "idle";
  $("elapsed").hidden = $("spend").hidden = !active;
  if (!active) return;
  const running = !TERMINAL.includes(state.status);
  const ms = (state.elapsed_ms || 0) + (running ? performance.now() - stateAt : 0);
  $("elapsed").textContent = seconds(ms);
  const s = state.spend || {};
  $("spend").textContent =
    `${s.steps ?? 0}/${s.max_steps ?? "—"} steps · ${s.calls ?? 0}/${s.max_llm_calls ?? "—"} calls · ` +
    `$${(s.usd ?? 0).toFixed(3)}/$${(s.max_usd ?? 0).toFixed(2)}`;
}
setInterval(renderElapsed, 100);

const isLive = () => state && !["idle", ...TERMINAL].includes(state.status);
// An automatic run is the SERVER's loop (see server.py); the page only flips the switch
// and draws what it is told. `running` is that loop with something left to do.
const running = () => Boolean(state?.auto && isLive());

function controls() {
  if (stale) return;
  const working = busy || state?.busy || running();
  const can = state?.can || {};
  const live = isLive();
  for (const id of ["start", "url", "goal", "reset-url", "reset-steps", "read-only", "max-steps", "max-usd"])
    $(id).disabled = working;
  const adjustable = Boolean(state?.options?.adjustable);
  $("refine").disabled = $("text-model").disabled = $("text-effort").disabled = working || !adjustable;
  $("refine-model").disabled = working || !adjustable || !$("refine").checked;
  $("choose").disabled = working || !can.predict;
  $("execute").disabled = working || !can.act;
  $("step").disabled = working || !live;
  // Never disabled: switching it OFF in the middle of a move is the whole point of it.
  $("auto").checked = Boolean(state?.auto);
  $("reset-run").disabled = working || !can.reset_run;
  $("reset-browser").disabled = working || !can.reset_browser;
  $("reset-site").disabled = working || !can.reset_run || !$("recipe").value;
  $("download").disabled = !state?.history?.length;
}

async function perform(fn, label) {
  if (busy) return;
  busy = true;
  $("error").hidden = true;
  controls();
  $("status").textContent = label;
  try {
    await fn();
  } catch (error) {
    await poll();
    $("error").textContent = error.message;
    $("error").hidden = false;
  } finally {
    busy = false;
    shownKey = "";
    render();
  }
}

const STATUS = {
  idle: "Ready for a task",
  ready: "Screen observed · ready for a decision",
  predicted: "Move chosen · inspect it, or execute",
  done: "The critic agrees the task is complete",
  blocked: "Blocked · no supported next move",
  stopped: "Stopped · a budget limit was reached",
};
const DOING = {
  start: "Opening a browser and observing the page…",
  predict: "Choosing the next move…",
  act: "Executing, then judging the result…",
  tick: "Choosing, executing and judging…",
  auto: "Running automatically · choosing, executing and judging…",
  reset_run: "Resetting the run…",
  reset_browser: "Opening a fresh browser…",
  reset_site: "Putting the site back…",
};

function chosenId() {
  const d = state?.decision;
  if (!d?.target) return null;
  return d.target.split(" (")[0];
}

function renderDecision() {
  const d = state.decision;
  const last = state.history?.at(-1);
  $("choice-title").textContent = d
    ? d.headline
    : state.status === "done"
      ? "Task complete"
      : state.status === "idle"
        ? "Waiting for a page"
        : "Choose a move";
  $("thought").textContent = d ? d.thought : TERMINAL.includes(state.status) ? state.note : "";
  $("thought").hidden = !$("thought").textContent;
  $("latency").textContent = d ? `${d.latency_ms} ms` : "—";
  $("confidence").textContent = d?.confidence != null ? percent(d.confidence) : "—";
  $("confidence-label").textContent =
    d?.probability != null ? `Confidence · target ${percent(d.probability)}` : "Confidence";
  $("operation").textContent = d ? d.operation : last && !state.decision ? `last: ${last.operation}` : "—";
  $("ranking-note").textContent = d
    ? `Aimed by ${d.decided_by}`
    : "Choose next to aim";
  const held = [
    d?.withheld_controls?.length ? `Not offered this step, having changed nothing twice: ${d.withheld_controls.join(", ")}.` : "",
    d?.withheld_labels?.length ? `Withheld from every operation, because the run was going round between them: ${d.withheld_labels.map((l) => `“${l}”`).join(", ")}.` : "",
    d?.fresh_look ? "The policy said DONE. That claim has to survive one fresh look at the page - a cart badge or a navigation may not have landed yet - so this move is a short wait and the policy is asked again." : "",
  ]
    .filter(Boolean)
    .join(" ");
  $("aside-note").textContent = d?.ranked
    ? `${held ? `${held} ` : ""}The chosen target is highlighted and dead ends are marked withheld. Per-target odds are not shown: the policy reports only the chosen target's probability.`
    : state.mode?.policy === "jev"
      ? "Jev is the acting policy. Choose next to see its operation, target and confidence."
      : "The prompted model is the acting policy here, so there is no confidence to show - it states a reason and an expectation instead.";
}

function renderElements() {
  const chosen = chosenId();
  const elements = [...(state.elements || [])];
    // On the DOM path the page's visible text rides along as plain `text` elements beside
  // the controls; the target table is the controls, so a run of text is listed only when
  // it is what was chosen. The pixel path has no control table and lists everything.
  const indexed = elements.some((e) => e.index != null);
  const actionable = elements.filter(
    (e) => e.id && (!indexed || e.index != null || e.id === chosen),
  );
  $("action-count").textContent = indexed
    ? `${actionable.length} controls · ${elements.length} elements`
    : `${elements.length} elements`;
  actionable.sort((a, b) => (b.id === chosen) - (a.id === chosen));
  $("choices").innerHTML = actionable.length
    ? actionable
        .slice(0, 160)
        .map((e) => {
          const meta = [e.role || e.kind, e.value ? `value “${e.value}”` : "", e.withheld?.length ? `withheld: ${e.withheld.join(", ")}` : ""]
            .filter(Boolean)
            .join(" · ");
          return `<div class="choice ${e.id === chosen ? "best" : ""} ${e.withheld?.length ? "withheld" : ""}" data-element="${escape(e.id)}"><span class="choice-id">${e.index != null ? `[${e.index}]` : "·"}</span><div class="choice-label">${escape(e.text || e.id)}<small>${escape(meta)}</small></div><span class="probability">${e.id === chosen && state.decision?.probability != null ? percent(state.decision.probability) : ""}</span></div>`;
        })
        .join("")
    : '<p class="muted">The controls the agent can see will appear here.</p>';

  const [w, h] = state.viewport || [0, 0];
  $("targets").innerHTML =
    w && h
      ? actionable
          .filter((e) => e.kind !== "text" || e.id === chosen)
          .slice(0, 220)
          .map((e) => {
            const [x, y, bw, bh] = e.box;
            const selected = e.id === chosen || e.id === hovered;
            return `<div class="target ${selected ? "selected" : ""}" data-element="${escape(e.id)}" style="left:${(100 * x) / w}%;top:${(100 * y) / h}%;width:${(100 * bw) / w}%;height:${(100 * bh) / h}%">${e.index != null ? `<span>${e.index}</span>` : ""}</div>`;
          })
          .join("")
      : "";
  $("targets").hidden = !$("overlays").checked;
}

function verdictOf(h) {
  if (h.refused) return ["refused", "Answer refused"];
  if (h.verdict_ok === true) return ["ok", `Verdict: worked · ${h.verdict_source}`];
  if (h.verdict_ok === false) return ["bad", `Verdict: did not work · ${h.verdict_source}`];
  return h.actions.length
    ? ["bad", "Stopped partway · not judged"]
    : ["none", "Nothing reached the screen"];
}

function renderHistory() {
  const history = state.history || [];
  $("history").innerHTML = history.length
    ? history
        .map((h, i) => {
          const [tone, label] = verdictOf(h);
          const open = opened.has(i);
          const odds = h.confidence != null ? ` · ${percent(h.confidence)}` : "";
          return `<div class="trace-row ${open ? "open" : ""}" data-row="${i}">
  <span class="number">${String(h.number).padStart(2, "0")}</span>
  <div class="trace-main">${escape(h.headline)}${h.claimed_done ? ' <em class="claim">claims done</em>' : ""}<small>${escape(h.thought)}</small></div>
  <span class="time">${h.model_ms ?? h.decide_ms} ms model · ${h.site_ms ?? "—"} ms site · ${h.frame_ms ?? "—"} ms frame${odds}</span>
  <span class="effect ${tone}">${escape(label)}</span>
  <div class="trace-detail">
    <dl>
      <dt>Decided</dt><dd>${escape(h.summary)} <code>${escape(h.signature)}</code></dd>
      <dt>Expected</dt><dd>${escape(h.expect || "—")}</dd>
      <dt>Executed</dt><dd>${h.actions.length ? h.actions.map((a) => `<code>${escape(a)}</code>`).join("<br>") : "nothing reached the browser"}</dd>
      <dt>Came back</dt><dd>${h.screen_changed ? "a different screen" : h.similarity < 1 ? "the same screen, repainted" : "no change observed"} <small>(similarity ${h.similarity.toFixed(2)} to the screen before)</small>${h.url_after !== h.url_before ? ` · now at <code>${escape(h.url_after)}</code>` : ""}${h.error ? ` · <span class="bad">${escape(h.error)}</span>` : ""}</dd>
      <dt>Took</dt><dd>${seconds(h.wall_ms ?? h.decide_ms + h.act_ms)} <small>(${h.decide_ms} ms choosing, ${h.act_ms} ms executing and judging)</small> · model ${h.model_ms ?? "—"} ms · site ${h.site_ms ?? "—"} ms · frame ${h.frame_ms ?? "—"} ms · judging and recording ${h.other_ms ?? "—"} ms</dd>
      <dt>Verdict</dt><dd>${escape(h.verdict_reason || label)}${h.verdict_source ? ` <small>(${escape(h.verdict_source)}, confidence ${percent(h.verdict_confidence)})</small>` : ""}</dd>
    </dl>
  </div>
</div>`;
        })
        .join("")
    : '<p class="muted">Each move leaves what was decided, what was executed, what came back, and the verdict.</p>';
  const s = state.spend || {};
  const t = state.timing || {};
  // `stepping` is the agent's own time; the clock beside the status also counts the time
  // a person spent reading between presses, which is nobody's latency.
  $("step-count").textContent =
    `${state.moves || 0} moves · ${state.steps || 0} actions · ${seconds(t.wall_ms || 0)} stepping · ` +
    `${seconds(t.model_ms || 0)} model · ${seconds(t.site_ms || 0)} site · ${seconds(t.frame_ms || 0)} frames` +
    (t.other_ms ? ` · ${seconds(t.other_ms)} judging` : "") +
    (t.first_load_ms ? ` · ${seconds(t.first_load_ms)} first load` : "") +
    (s.calls != null ? ` · ${s.calls} model calls · $${(s.usd || 0).toFixed(3)}` : "");
  $("step-count").title = t.site_clock
    ? "model: the policy's own calls · site: performing, settling, observing, resting · frames: the screenshots, taken out of the site's share · judging: the critic and the recording"
    : "This perceiver keeps no site clock, so site is what the actions reported and perception is counted under judging.";
}

function ago(iso) {
  const then = Date.parse(iso);
  if (!then) return "";
  const s = Math.max(0, (Date.now() - then) / 1000);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 129600) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function renderLibrary() {
  const lib = state.library || { skills: [] };
  const query = $("library-filter").value.trim().toLowerCase();
  const here = $("library-here").checked ? state.domain : "";
  const skills = lib.skills.filter(
    (k) =>
      (!here || k.domain === here) &&
      (!query || `${k.name} ${k.domain} ${k.task_text} ${k.summary}`.toLowerCase().includes(query)),
  );
  $("library-count").textContent =
    `${lib.count || 0} skills · ${(lib.domains || []).length} sites` +
    (skills.length !== lib.count ? ` · showing ${skills.length}` : "");
  $("library-root").textContent = lib.root ? `Read through the skill store at ${lib.root} · re-read when it changes on disk` : "";
  $("library-here").parentElement.hidden = !state.domain;
  $("library").innerHTML = skills.length
    ? `<div class="skill-head"><span>Skill</span><span>Site</span><span>Ver</span><span>Recorded</span><span>Verifier</span><span>Runs</span></div>` +
      skills
        .map((k) => {
          const fresh = Date.now() - Date.parse(k.created_at) < 600000;
          const key = `skill:${k.namespace}/${k.name}`;
          const open = opened.has(key);
          const st = k.stats;
          return `<div class="skill ${open ? "open" : ""} ${k.demoted_reason ? "demoted" : ""}" data-skill="${escape(key)}">
  <div class="skill-name">${escape(k.name)}${fresh ? ' <em class="fresh">new</em>' : ""}<small>${escape(k.summary)}</small></div>
  <span class="skill-site">${escape(k.domain)}<small>${escape(k.path)}${k.render_mode ? ` · ${escape(k.render_mode)}` : ""}</small></span>
  <span class="mono">v${k.version}</span>
  <span class="mono" title="${escape(k.created_at)}">${escape(ago(k.created_at))}</span>
  <span class="badge ${k.verifier ? "yes" : "no"}">${k.verifier ? "verifier" : "none"}</span>
  <span class="mono">${st.runs ? `${st.successes}/${st.runs}` : "—"}</span>
  <div class="skill-detail">
    <dl>
      <dt>Learned from</dt><dd>${escape(k.task_text)}</dd>
      <dt>Does</dt><dd>${k.action_signature.length ? k.action_signature.map((a) => `<code>${escape(a)}</code>`).join(" → ") : "no signature earned yet"}</dd>
      <dt>Takes</dt><dd>${k.params.length ? k.params.map((p) => `<code>${escape(p)}</code>`).join(" ") : "no parameters"}</dd>
      <dt>Proven for</dt><dd>${k.precedents.length ? k.precedents.map(escape).join("<br>") : "no precedent recorded"}</dd>
      <dt>Starts on</dt><dd>${k.precondition ? `<code>${escape(k.precondition.slice(0, 16))}</code>` : "anywhere"}</dd>
      <dt>Written by</dt><dd>${escape(k.model)} · run <code>${escape(k.trajectory_id)}</code> · ${k.lines} lines · versions ${k.versions.join(", ")}</dd>
      <dt>Self-reported</dt><dd>${st.runs ? `${st.successes}/${st.runs} runs, mean ${st.mean_ms} ms` : "never run"} <small>— counted by the skill's own verifier, so read the critic, not this</small></dd>
      ${k.demoted_reason ? `<dt>Demoted</dt><dd class="bad">${escape(k.demoted_reason)}</dd>` : ""}
    </dl>
  </div>
</div>`;
        })
        .join("")
    : `<p class="muted">${lib.count ? "No skill matches that filter." : "Nothing learned yet. Run <code>learn</code> on a task and it appears here without a restart."}</p>`;
}

function renderRecipes() {
  const names = state.server?.recipes || [];
  const select = $("recipe");
  if (select.dataset.filled === names.join()) return;
  select.dataset.filled = names.join();
  select.replaceChildren(
    ...names.map((n) => {
      const option = document.createElement("option");
      option.value = n;
      option.textContent = n === "task" ? "the task's own undo" : n;
      return option;
    }),
  );
}

function fill(select, values, label, current) {
  if (select.dataset.filled === values.join()) return;
  select.dataset.filled = values.join();
  select.replaceChildren(
    ...values.map((v) => {
      const option = document.createElement("option");
      option.value = v;
      option.textContent = label(v);
      return option;
    }),
  );
  select.value = current;
}

// Filled ONCE from what the server is configured with and then left alone: these are the
// next run's settings and belong to whoever is about to press Start, so a poll must not
// put them back.
function renderModels() {
  const o = state.options;
  if (!o) return;
  const c = o.configured || {};
  fill($("text-model"), o.text_models || [], (v) => v, c.text_model);
  fill($("refine-model"), o.text_models || [], (v) => v, c.refine_model || c.text_model);
  fill($("text-effort"), o.efforts || [""], (v) => v || "default", c.text_effort || "");
  if (!$("refine").dataset.filled) {
    $("refine").dataset.filled = "1";
    $("refine").checked = Boolean(c.refine_goal);
  }
  const m = state.mode || {};
  $("model-summary").textContent = !o.adjustable
    ? "· only the Jev policy has them"
    : state.status === "idle"
      ? ""
      : `· this run types with ${m.text_model}${m.text_effort ? ` (${m.text_effort})` : ""}, refinement ${m.refine_goal ? `on, by ${m.refine_model || m.text_model}` : "off"}`;
  $("model-note").textContent =
    "Applied when a run is started: the text writer is built once per run, so a change here takes effect at the next Start run, not in the middle of this one." +
    (m.goal_shown ? ` The policy is being shown: “${m.goal_shown}”` : "");
}

function render() {
  if (!state || stale) return;
  renderElapsed();
  renderRecipes();
  renderModels();
  const working = busy || state.busy || running();
  $("status").textContent = working
    ? DOING[state.doing] || (running() ? DOING.auto : $("status").textContent)
    : STATUS[state.status] || state.status;
  $("status-dot").className = `dot ${working ? "working" : state.status}`;
  // A finished run's note is already in the decision panel; the last reset is reported
  // in every state, because "did the undo work?" is asked most often AFTER a run ends.
  const reset = state.resets?.at(-1);
  $("note").textContent =
    state.auto_note ||
    (!TERMINAL.includes(state.status) && state.note) ||
    (reset ? `Last reset · ${reset.what} · ${reset.ok ? "restored" : "FAILED"} · ${reset.detail}` : "");
  const m = state.mode || {};
  $("mode-line").textContent = `${m.perception || "—"} perception · ${m.policy || "—"} policy · ${m.render || "—"}${m.chrome_attach ? " · attached Chrome" : m.chrome_profile ? " · real Chrome profile" : ""}`;
  $("model-tag").innerHTML = m.model ? `${escape(m.model)} <span>decides</span>` : "no run yet";
  $("address").textContent = state.url || "No browser open";
  $("live").textContent = working ? "WORKING" : state.status === "idle" ? "IDLE" : "LIVE";
  $("browser-line").textContent = m.browser || "—";
  $("pipeline").textContent =
    m.perception === "dom" ? "DOM controls → typed move → point action" : "pixels → detector + OCR → typed move → point action";
  $("footer-right").textContent = state.server ? `data ${state.server.data_dir}` : "";
  if (state.status === "idle") {
    $("screenshot").hidden = true;
    $("empty").hidden = false;
    shownFrame = -1;
  }
  renderDecision();
  renderElements();
  renderHistory();
  renderLibrary();
  const { library, elements, ...rest } = state;
  $("model-state").textContent = JSON.stringify(
    { ...rest, elements: `${(elements || []).length} elements`, library: `${library?.count || 0} skills` },
    null,
    2,
  );
  controls();
}

$("task-form").addEventListener("submit", (event) => {
  event.preventDefault();
  opened.clear();
  perform(
    () =>
      call("start", {
        url: $("url").value,
        goal: $("goal").value,
        reset_url: $("reset-url").value,
        reset_steps: $("reset-steps").value,
        read_only: $("read-only").checked,
        max_steps: $("max-steps").value,
        max_usd: $("max-usd").value,
        ...(state?.options?.adjustable
          ? {
              refine_goal: $("refine").checked,
              refine_model: $("refine").checked ? $("refine-model").value : "",
              text_model: $("text-model").value,
              text_effort: $("text-effort").value,
            }
          : {}),
      }),
    DOING.start,
  );
});
$("choose").addEventListener("click", () => perform(() => call("predict"), DOING.predict));
$("execute").addEventListener("click", () => perform(() => call("act"), DOING.act));
$("step").addEventListener("click", () => perform(() => call("tick"), DOING.tick));
// Not through perform(): that refuses while a step is running, and "off" has to get
// through exactly then. The server takes this one without queueing it, for the same reason.
$("auto").addEventListener("change", async () => {
  const on = $("auto").checked;
  $("error").hidden = true;
  try {
    await call("auto", { on });
    if (!on && state?.busy) $("status").textContent = "Pausing after the current move…";
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
});
$("refine").addEventListener("change", controls);
$("reload").addEventListener("click", () => location.reload());
$("reset-run").addEventListener("click", () => {
  opened.clear();
  perform(() => call("reset_run"), DOING.reset_run);
});
$("reset-browser").addEventListener("click", () => {
  opened.clear();
  perform(() => call("reset_browser"), DOING.reset_browser);
});
$("reset-site").addEventListener("click", () => {
  perform(() => call("reset_site", { recipe: $("recipe").value }), DOING.reset_site);
});
$("recipe").addEventListener("change", controls);
$("overlays").addEventListener("change", () => {
  $("targets").hidden = !$("overlays").checked;
});
$("library-filter").addEventListener("input", renderLibrary);
$("library-here").addEventListener("change", renderLibrary);

$("choices").addEventListener("pointerover", (event) => {
  hovered = event.target.closest("[data-element]")?.dataset.element ?? null;
  document
    .querySelectorAll(".target")
    .forEach((t) => t.classList.toggle("selected", t.dataset.element === hovered || t.dataset.element === chosenId()));
});
$("choices").addEventListener("pointerleave", () => {
  hovered = null;
  document
    .querySelectorAll(".target")
    .forEach((t) => t.classList.toggle("selected", t.dataset.element === chosenId()));
});
$("history").addEventListener("click", (event) => {
  const row = event.target.closest("[data-row]");
  if (!row) return;
  const key = Number(row.dataset.row);
  opened.has(key) ? opened.delete(key) : opened.add(key);
  renderHistory();
});
$("library").addEventListener("click", (event) => {
  const row = event.target.closest("[data-skill]");
  if (!row) return;
  const key = row.dataset.skill;
  opened.has(key) ? opened.delete(key) : opened.add(key);
  renderLibrary();
});
$("download").addEventListener("click", () => {
  const { library, ...rest } = state;
  const blob = new Blob([JSON.stringify(rest, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "skillweaver-inspector-trace.json";
  a.click();
  URL.revokeObjectURL(url);
});

// What was typed before a self-reload (see refused()), put back once.
try {
  const draft = JSON.parse(stored.get("inspector-draft") || "null");
  if (draft) {
    $("url").value = draft.url ?? $("url").value;
    $("goal").value = draft.goal ?? $("goal").value;
  }
} catch {
  /* a draft that does not parse is not worth a message */
}
stored.drop("inspector-draft");

// Faster while the server is running by itself: nothing on this page is awaiting those
// moves, so the poll is the only thing that shows them.
async function loop() {
  if (!busy) await poll();
  if (!stale) setTimeout(loop, running() ? 400 : 1000);
}
loop();
