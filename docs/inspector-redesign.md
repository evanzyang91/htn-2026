# Inspector redesign — design brief (not built)

Status: parked on 2026-09-20. A worker was started on this and stopped before it wrote
anything; nothing below exists yet. The current page is `src/skillweaver/inspector/static/`
(`index.html`, `app.js`, `style.css`), served by `inspector/server.py` from the state JSON
`inspector/session.py` publishes.

## What is wrong with the current page

Judged from a full-page screenshot of a real 21-move run:

- A marketing hero headline ("Watch it decide. Then let it act.") on an operator tool.
- The task form permanently takes the top third of the viewport.
- The browser frame — the thing you watch — is small and not where attention lands.
- Transport buttons float in a loose row, far from the frame they control.
- The decision trail is an undifferentiated wall of grey monospace: you cannot tell at a
  glance which moves worked, which were slow, or why.
- Timings are text soup (`442 ms decide · 7412 ms act`).
- Run facts (status, budget, spend) are scattered pills.
- The "next move" aside wraps a 300-character URL.
- Everything has the same visual weight.

## Direction: an instrument panel for watching an agent, not a landing page

### Layout — app shell filling the viewport
No page scroll at desktop widths; panels scroll internally.

1. **Top bar (~48px).** Product mark; run status as ONE strong state chip (idle / deciding /
   acting / ready / solved / blocked / failed — colour-coded, pulsing while busy); run facts
   as compact labelled stats, each with a thin budget meter (steps x/y, model calls x/y,
   spend $x/$y, elapsed); the mode line (perception · policy · browser) quiet on the right.
2. **Command bar.** Start URL + task in ONE row (task input grows); "Start run" is the
   single primary button. Read-only, max steps, max $, undo recipe, refine, text model and
   effort collapse behind an "Options" disclosure. Once a run starts the bar stays compact
   and shows the task as text with an Edit affordance.
3. **Main area, two columns.**
   - LEFT (flex, dominant): the browser stage. Frame as large as the column allows at the
     correct aspect ratio; a slim faux-chrome strip with a middle-truncated URL and a LIVE
     dot; target overlay boxes. Directly under the frame, the transport as one grouped
     control: Choose next · Execute · Step · Auto-run switch, then a separated group for
     Reset run / Reset browser / Reset site + recipe. Keyboard shortcuts (C / E / S / A)
     shown as small `kbd` hints.
   - RIGHT (fixed ~400–440px): a tabbed inspector.
     - **Decision** — operation as a big token, target label, confidence as a bar, offered
       operations' probabilities as a small horizontal bar list when present, withheld
       controls and labels, the policy's reason (URLs middle-truncated, full text in `title`).
     - **Elements** — the numbered controls list, filterable; hovering a row highlights its
       box on the frame and the reverse where the data allows.
     - **Skills** — the library view with its filter.
4. **Timeline, docked under the main area** (~32vh, internal scroll, newest at the bottom,
   auto-scrolled unless the user scrolled up). One compact row per move:
   index · operation badge colour-coded by kind (CLICK / TYPE_TEXT / ENTER / SCROLL / WAIT /
   BACK / DONE / BLOCKED) · target label (one line, ellipsis) · a proportional stacked timing
   bar (model / site / frame / judging) on a scale SHARED across rows so slow moves look
   slow, exact ms in a tooltip · a verdict mark (worked / did not work / refused) and whether
   it was the programmatic check or a model call. Click a row to expand its detail: thought,
   expect, withheld note, critic reason, timings table. The "fresh look before DONE" wait is
   its own quiet row type, never a failure. Run totals sit in the timeline header with the
   same four-colour legend.

### Visual language
Dark, dense, calm. One neutral scale, one accent, semantic colours only for state
(ok / warn / fail / busy). Everything as CSS custom properties on `:root`; light theme via
`prefers-color-scheme` using the same tokens. System UI font for text, system monospace only
for data (ms, ids, URLs, code). 12–13px base for dense data; hierarchy by weight, size and
colour rather than boxes; hairline borders, 6–8px radii; no shadow soup, no decorative
gradients, no emoji. Tabular numbers for all figures. Visible focus rings, real button and
label semantics, `aria-live` for status, `prefers-reduced-motion` respected.

### Responsive
Below ~1100px the right panel drops under the stage. Below ~720px everything stacks with a
16px gutter and no horizontal page scroll. Long labels and URLs wrap or truncate
deliberately at every nesting level (`min-width: 0` on flex/grid children).

### Designed empty and edge states
No run yet (stage shows a quiet placeholder with the one thing to do); browser closed;
restarted-inspector banner (exists — restyle); a move that raised; a blocked run (the
policy's reason shown prominently, URL truncated).

### `app.js`
Keep the polling, token handling and stale-token recovery intact. Restructure rendering into
small pure render functions keyed off the state JSON. Never rebuild the frame `<img>` or
lose scroll positions, expanded rows or focus on a poll — use keyed updates. The current
page re-renders everything on every poll, which is also why browser-automation element refs
go stale within a second.

## Constraints
- Only `static/index.html`, `static/app.js`, `static/style.css` should need to change. If a
  field is genuinely missing, make the smallest addition to `session.py`'s state JSON and
  keep `scripts/check_e_inspector.py` passing.
- Keep every capability, endpoint, the `__TOKEN__` placeholder, and the token / same-origin /
  loopback checks exactly. Respect the CSP in `server.py`: no inline script if it forbids
  it, no CDN, no external fonts, no new dependencies.
- The state JSON already carries what the timeline needs: per-step and per-run `wall_ms`,
  `model_ms`, `site_ms`, `frame_ms`, `other_ms` (judging and recording), plus
  `first_load_ms` / `first_frame_ms`; the auto-run flag; withheld controls and labels; the
  run's `text_model`, `text_effort`, `refine_goal` and `goal_shown`.

## Proof when it is built
Drive one real Wikipedia run through the new page (choose, execute, step, auto-run on then
off). Screenshot at 1440×900, ~1000px and ~420px wide, in the empty, mid-run and solved
states, and look at each one: alignment, overflow, contrast, hierarchy. Lint must pass.

Reference screenshot of the page being replaced: taken during the 21-move run on
2026-09-20 (not committed; re-take from a live run).
