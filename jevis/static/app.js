const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="demo-token"]').content;
let state = null,
  busy = false,
  automatic = false;
const goals = {
  flights: 'Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. Stop when matching flight options are visible. Do not select or book a flight.',
  custom: "Add all the ingredients needed to bake a cake to my cart",
  travel: 'Find a Design stay in Lisbon with Free cancellation and open Casa Flora.',
  research:
    "Open the article about using finite choices to control browser agents.",
};
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const percent = (value) => `${(value * 100).toFixed(value < 0.01 ? 1 : 0)}%`;
let stateAt = performance.now();
async function call(name, body = {}) {
  const response = await fetch(`/api/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Demo-Token": token },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "Request failed");
  state = data;
  stateAt = performance.now();
  render();
  return data;
}
function renderElapsed() {
  const active = state?.started_at != null;
  $("elapsed").hidden = $("model-elapsed").hidden = !active;
  if (!active) return;
  const running = !["done", "blocked"].includes(state.status);
  const ms = (state.elapsed_ms || 0) + (running ? performance.now() - stateAt : 0);
  $("elapsed").textContent = `${(ms / 1000).toFixed(2)} s`;
  // Model time only: measured per call, so it never extrapolates between server updates.
  $("model-elapsed").textContent = `model ${((state.model_ms || 0) / 1000).toFixed(2)} s`;
}
setInterval(renderElapsed, 100);
function controls() {
  const live = state?.page && !["done", "blocked"].includes(state.status);
  $("start").disabled = busy;
  $("scenario").disabled = busy;
  $("goal").disabled = busy;
  $("url").disabled = busy;
  $("refine").disabled = busy;
  $("text-model").disabled = busy;
  $("refine-model").disabled = busy || !$("refine").checked;
  $("refine-effort").disabled = busy || !$("refine").checked;
  $("choose").disabled = busy || !live;
  $("execute").disabled = busy || !state?.decision || !live;
  $("auto").disabled = busy || !live;
  $("auto").hidden = automatic;
  $("stop").hidden = !automatic;
  $("download").disabled = !state?.history?.length;
  $("focus-browser").hidden = !state?.page;
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
    automatic = false;
    try {
      state = await fetch("/api/state").then((r) => r.json());
      render();
    } catch {
      /* Preserve the original failure if the server disconnected. */
    }
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("status").textContent = "Paused · needs attention";
  } finally {
    busy = false;
    controls();
  }
}
function fillModels(select, models, current) {
  if (!models.length || select.dataset.filled === models.join()) {
    if (current && !select.dataset.dirty) select.value = current;
    return;
  }
  select.dataset.filled = models.join();
  select.replaceChildren(
    ...models.map((m) => {
      const option = document.createElement("option");
      option.value = option.textContent = m;
      return option;
    }),
  );
  select.value = current || models[0];
}
function renderModels() {
  fillModels($("refine-model"), state?.refine_models || [], state?.refine_model);
  fillModels($("text-model"), state?.text_models || [], state?.text_model);
  if (state?.refine_effort) $("refine-effort").value = state.refine_effort;
}
function renderRefinement() {
  const r = state?.refinement,
    tag = $("refinement");
  tag.hidden = !r?.category;
  if (!r?.category) return;
  tag.textContent = `${r.category} ${Math.round(r.confidence * 100)}%${r.applied ? "" : " · general rules"}`;
}
function render() {
  if (!state) return;
  renderElapsed();
  renderModels();
  renderRefinement();
  $("helper").textContent = document.body.classList.contains("dev")
    ? `writer · ${state.text_model}`
    : "";
  $("plan").innerHTML = (state.plan || [])
    .map(
      (goal, i) =>
        `<div class="plan-step ${i === state.plan_index ? "current" : ""}"><span>${i < state.plan_index ? "✓" : i + 1}</span>${escape(goal)}</div>`,
    )
    .join("");
  const page = state.page,
    d =
      state.decision ||
      (state.status === "done" ? state.decisions?.at(-1) : null);
  const labels = {
    idle: "Ready",
    ready: "Reading the page",
    predicted: "Ready to act",
    done: "Finished. Check the result.",
    blocked: "Stopped. It could not find a way forward.",
  };
  $("status").textContent = labels[state.status] || state.status;
  if (!page) {
    controls();
    return;
  }
  $("empty").hidden = true;
  $("screenshot").hidden = false;
  if (page.screenshot)
    $("screenshot").src = `data:image/jpeg;base64,${page.screenshot}`;
  $("url-bar").textContent = page.url;
  $("page-title").textContent = page.title;
  $("action-count").textContent = `${state.elements.length} elements`;
  const chosen = page.actions.find((a) => a.id === d?.choice);
  const doing = d ? chosen?.label || d.choice : "Choose an action";
  $("choice-detail").textContent = doing;
  $("latency").textContent = d ? `${d.latency_ms} ms` : "–";
  $("confidence").textContent = d?.target_confidence != null ? percent(d.target_confidence) : "–";
  $("completion").textContent = d ? d.operation : "–";
  $("ranking-note").textContent = d ? "Ranked" : "Unranked";
  const op = Object.entries(d?.operation_probabilities || {}).sort((a,b)=>b[1]-a[1]);
  $("operation-choices").innerHTML = op.map(([name,p]) =>
    `<span class="operation-choice ${name === d.operation ? 'best' : ''}">${escape(name)} <b>${percent(p)}</b></span>`).join('');
  const probability = e => d?.target_probabilities[e.index] ??
    Math.max(-1, ...(e.options || []).map(o=>d?.target_probabilities[o.index] ?? -1));
  const selectedIndex = d?.target?.split(':')[0];
  const elements = [...state.elements];
  if (d) elements.sort((a,b)=>probability(b)-probability(a));
  $("choices").innerHTML = elements.map(e => {
    const p = probability(e);
    return `<div class="choice ${selectedIndex === e.index ? 'best' : ''}" data-action="${escape(e.index)}"><span class="choice-id">[${escape(e.index)}]</span><div class="choice-label">${escape(e.label)}<small>${escape(e.role)} · ${escape(e.operations.join(' / '))}${e.value ? ' · '+escape(e.value) : ''}${e.checked !== undefined ? ' · checked '+escape(e.checked) : ''}</small>${p >= 0 ? `<div class="bar" style="--probability:${p*100}%"></div>` : ''}</div><span class="probability">${p >= 0 ? percent(p) : '–'}</span></div>`;
  }).join('');
  const targets = new Map();
  for (const a of page.actions) if (a.rect && !targets.has(a.node)) targets.set(a.node, a);
  $("targets").innerHTML = [...targets.values()].map((a,i) => {
    const index=String(i+1);
    return `<div class="target ${index === selectedIndex ? 'selected' : ''}" data-action="${index}" style="left:${100*a.rect.x/page.w}%;top:${100*a.rect.y/page.h}%;width:${100*a.rect.w/page.w}%;height:${100*a.rect.h/page.h}%"><span>${index}</span></div>`;
  }).join('');
  $("targets").hidden = !$("overlays").checked;
  $("history").innerHTML = state.history.length
    ? state.history
        .map(
          (h) =>
            `<div class="trace-row"><span class="number">${String(h.step).padStart(2, "0")}</span><div>${escape(h.action)}${h.text ? ` <b>“${escape(h.text)}”</b><small>${escape(h.text_helper)}</small>` : ""}</div><span class="time">${h.model_ms ?? h.latency_ms} ms model · ${h.load_ms ?? "–"} ms load${h.frame_ms ? ` · ${h.frame_ms} ms frame` : ""}<small>${percent(h.probability)}</small></span><span class="effect">${h.page_changed ? "Page changed" : "No change observed"}</span></div>`,
        )
        .join("")
    : '<p class="muted">Each executed action leaves an observed result.</p>';
  renderTurn();
  $("step-count").textContent =
    `${state.history.length} actions · ${(state.elapsed_ms / 1000).toFixed(2)} s total · ` +
    `${((state.model_ms || 0) / 1000).toFixed(2)} s model · ${((state.load_ms || 0) / 1000).toFixed(2)} s load` +
    (state.frame_ms ? ` · ${(state.frame_ms / 1000).toFixed(2)} s frames` : "") +
    (state.initial_load_ms ? ` · ${(state.initial_load_ms / 1000).toFixed(2)} s first load` : "");
  $("model-state").textContent = JSON.stringify(
    d?.request || {
      goal: state.goal,
      url: page.url,
      text: page.text,
      actions: page.actions.map(({ rect, node, ...rest }) => rest),
    },
    null,
    2,
  );
  controls();
}
$("task-form").addEventListener("submit", (event) => {
  event.preventDefault();
  automatic = false;
  const refine = $("refine").checked;
  startTurn($("goal").value.trim());
  perform(
    async () => {
      await call("reset", {
        scenario: $("scenario").value,
        goal: $("goal").value,
        url: $("url").value,
        refine,
        refine_model: $("refine-model").value,
        refine_effort: $("refine-effort").value,
        dev: document.body.classList.contains("dev"),
      });
      // The rewritten goal is a developer detail; the user never sees it rewritten.
      if (refine && state?.goal && document.body.classList.contains("dev"))
        $("goal").value = state.goal;
      else $("goal").value = "";
      grow();
      // Same perform(), so the run starts without waiting for a second click.
      if ($("autorun").checked) await runAutomatically();
    },
    refine ? "Refining the goal, then opening a browser…" : "Opening a fresh browser…",
  );
});
$("scenario").addEventListener("change", () => {
  $("goal").value = goals[$("scenario").value];
  $("url").placeholder =
    $("scenario").value === "custom"
      ? "Start on a specific site (optional)"
      : "Fixed for this scenario";
});
$("choose").addEventListener("click", () =>
  perform(() => call("predict"), "Comparing the available actions…"),
);
$("execute").addEventListener("click", () =>
  perform(
    () => call("act", { fingerprint: state.page.fingerprint }),
    "Executing the choice…",
  ),
);
async function runAutomatically() {
  automatic = true;
  controls();
  for (let i = 0; i < state.max_steps * 2 && automatic; i++) {
    $("status").textContent = "Running…";
    if ($("pace").checked) {
      await call("predict");
      await new Promise(resolve => setTimeout(resolve, 450));
      if (!automatic) break;
      await call("act", {fingerprint: state.page.fingerprint});
    } else {
      await call("tick");
    }
    if (["done", "blocked"].includes(state.status)) break;
  }
  automatic = false;
}
$("auto").addEventListener("click", () =>
  perform(runAutomatically, "Running the browser…"),
);
$("stop").addEventListener("click", () => {
  automatic = false;
  $("status").textContent = "Pausing after the current request…";
  controls();
});
$("text-model").addEventListener("change", () =>
  // Applies to the next TYPE_TEXT, so it can be switched while a run is going.
  perform(() => call("model", { text_model: $("text-model").value }), "Switching the agent's model…"),
);
$("refine").addEventListener("change", controls);
$("overlays").addEventListener("change", () => {
  $("targets").hidden = !$("overlays").checked;
});
$("choices").addEventListener("pointerover", (event) => {
  const id = event.target.closest("[data-action]")?.dataset.action;
  document
    .querySelectorAll(".target")
    .forEach((t) =>
      t.classList.toggle(
        "selected",
        t.dataset.action === id || t.dataset.action === state?.decision?.target?.split(':')[0],
      ),
    );
});
$("choices").addEventListener("pointerleave", () =>
  document
    .querySelectorAll(".target")
    .forEach((t) =>
      t.classList.toggle(
        "selected",
        t.dataset.action === state?.decision?.target?.split(':')[0],
      ),
    ),
);
$("download").addEventListener("click", () => {
  const { page, ...rest } = state;
  const blob = new Blob(
    [
      JSON.stringify(
        { ...rest, page: { ...page, screenshot: undefined } },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "typesafe-browser-trace.json";
  a.click();
  URL.revokeObjectURL(url);
});
fetch("/api/state")
  .then((r) => r.json())
  .then((s) => {
    state = s;
    render();
  })
  .catch(() => {
    $("status").textContent = "Cannot reach local demo server";
  });

/* ---- Conversation ------------------------------------------------------- */

// What the agent is doing, said the way a person would. The raw label is an accessibility name
// written for screen readers ("Add to cart - Robin Hood Flour"): precise, but not for reading.
function plainly(action, decision) {
  if (!action) return decision?.operation === "DONE" ? "Finished" : "Thinking";
  const label = (action.label || "").split(" · ")[0].trim();
  switch (action.kind) {
    case "fill": return `Typing “${decision?.text ?? "…"}”`;
    case "scroll": return "Looking further down the page";
    case "wait": return "Waiting for the page";
    case "back": return "Going back";
    case "enter": return "Submitting";
    default: return `Clicking ${label}`;
  }
}
function stepText(h) {
  if (h.kind === "fill") return `Typed “${h.text}”`;
  if (h.kind === "scroll") return "Scrolled the page";
  if (h.kind === "wait") return "Waited for the page";
  if (h.kind === "back") return "Went back";
  if (h.kind === "enter") return "Submitted the search";
  return `Clicked ${(h.action || "").split(" · ")[0]}`;
}
let turn = null;
function startTurn(request) {
  $("intro").hidden = true;
  document.body.classList.remove("idle");
  const thread = $("thread");
  const you = document.createElement("div");
  you.className = "turn user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = request;
  you.append(bubble);
  const agent = document.createElement("div");
  agent.className = "turn agent";
  const site = document.createElement("div");
  site.className = "site";
  const acts = document.createElement("ol");
  acts.className = "acts";
  const verdict = document.createElement("p");
  verdict.className = "verdict";
  verdict.hidden = true;
  agent.append(site, acts, verdict);
  thread.append(you, agent);
  turn = { site, acts, verdict };
  thread.scrollTop = thread.scrollHeight;
}
function renderTurn() {
  if (!turn || !state) return;
  const url = state.chosen_url || state.page?.url;
  if (url && !turn.site.textContent) {
    turn.site.innerHTML = "";
    turn.site.append("Working on ", Object.assign(document.createElement("code"), {
      textContent: new URL(url).hostname.replace(/^www\./, ""),
    }));
  }
  const items = (state.history || []).map((h) => {
    const li = document.createElement("li");
    li.textContent = stepText(h);
    if (h.page_changed === false) li.classList.add("failed");
    return li;
  });
  const running = !["done", "blocked", "idle"].includes(state.status);
  if (running) {
    const li = document.createElement("li");
    li.className = "current";
    const chosen = state.page?.actions?.find((a) => a.id === state.decision?.choice);
    li.textContent = state.decision ? plainly(chosen, state.decision) : "Reading the page";
    items.push(li);
  }
  turn.acts.replaceChildren(...items);
  turn.verdict.hidden = running;
  if (!running) {
    const done = state.status === "done";
    turn.verdict.className = `verdict ${done ? "ok" : "bad"}`;
    turn.verdict.textContent = done
      ? "Done. The browser window shows the result."
      : "I could not finish this one. The browser window shows where I stopped.";
  }
  document.body.classList.toggle("running", running);
  $("thread").scrollTop = $("thread").scrollHeight;
}

/* ---- Developer view ----------------------------------------------------- */

const DEV_KEY = "agent-dev-view";
function applyDev(on) {
  document.body.classList.toggle("dev", on);
  $("dev-toggle").checked = on;
  localStorage.setItem(DEV_KEY, on ? "1" : "");
}
$("dev-toggle").addEventListener("change", () => applyDev($("dev-toggle").checked));
applyDev(localStorage.getItem(DEV_KEY) === "1");

/* ---- Watching the real browser ------------------------------------------ */

$("focus-browser").addEventListener("click", () =>
  perform(() => call("focus"), "Bringing the browser window forward…"),
);

/* ---- Voice ------------------------------------------------------------- */

const Recogniser = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!Recogniser) {
  $("mic").hidden = true;
} else {
  const recogniser = new Recogniser();
  recogniser.lang = navigator.language || "en-US";
  recogniser.interimResults = true;
  recogniser.continuous = false;
  let listening = false,
    committed = "";
  const say = (message) => {
    $("voice-status").textContent = message;
    $("voice-status").hidden = !message;
  };
  $("mic").addEventListener("click", () => {
    if (listening) return recogniser.stop();
    committed = $("goal").value.trim();
    try {
      recogniser.start();
    } catch {
      /* start() throws if it is already running; the state below still corrects itself. */
    }
  });
  recogniser.addEventListener("start", () => {
    listening = true;
    $("mic").classList.add("listening");
    say("Listening. Speak your request.");
  });
  recogniser.addEventListener("result", (event) => {
    let heard = "";
    for (const result of event.results) heard += result[0].transcript;
    heard = heard.trim();
    $("goal").value = committed ? `${committed} ${heard}` : heard;
  });
  recogniser.addEventListener("error", (event) => {
    say(
      event.error === "not-allowed"
        ? "Microphone permission was refused, so dictation is off."
        : `Dictation stopped: ${event.error}.`,
    );
  });
  recogniser.addEventListener("end", () => {
    listening = false;
    $("mic").classList.remove("listening");
    if ($("voice-status").textContent.startsWith("Listening")) say("");
  });
}


/* ---- Composer behaviour -------------------------------------------------- */

function grow() {
  const box = $("goal");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight, 200)}px`;
}
$("goal").addEventListener("input", grow);
$("goal").addEventListener("keydown", (event) => {
  // Enter sends, Shift+Enter makes a new line — the convention everywhere else.
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!busy) $("task-form").requestSubmit();
  }
});
for (const example of document.querySelectorAll(".example"))
  example.addEventListener("click", () => {
    $("goal").value = example.textContent.trim();
    grow();
    $("goal").focus();
  });


/* ---- Theme -------------------------------------------------------------- */

// Follow the system unless the reader says otherwise, and remember that choice.
const THEME_KEY = "agent-theme";
function applyTheme(theme) {
  if (theme) document.body.dataset.theme = theme;
  else delete document.body.dataset.theme;
  localStorage.setItem(THEME_KEY, theme || "");
}
applyTheme(localStorage.getItem(THEME_KEY) || "");
$("theme").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const current = document.body.dataset.theme || (dark ? "dark" : "light");
  applyTheme(current === "dark" ? "light" : "dark");
});
document.body.classList.add("idle");
