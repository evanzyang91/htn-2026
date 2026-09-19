/* Core: state transport, DOM helpers, focus preservation, render loop.

   The server owns the state. Every user action posts to /api/act and the
   response is the complete new state, which is re-rendered from scratch.
   That keeps /__state an exact description of the screen.                */
(function (global) {
  "use strict";

  var App = {
    state: null,
    screens: {},
    seq: 0,
    applied: -1,
    queue: []
  };

  /* ---------------------------------------------------------- dom helpers */

  function add(parent, kids) {
    for (var i = 0; i < kids.length; i++) {
      var k = kids[i];
      if (k === null || k === undefined || k === false || k === true) continue;
      if (Array.isArray(k)) { add(parent, k); continue; }
      parent.appendChild(typeof k === "string" || typeof k === "number"
        ? document.createTextNode(String(k)) : k);
    }
  }

  var PROPS = { checked: 1, indeterminate: 1, value: 1, disabled: 1 };

  function h(tag, attrs) {
    var e = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") e.className = v;
        else if (k === "text") e.textContent = String(v);
        else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2).toLowerCase(), v);
        else if (PROPS[k]) e[k] = v;
        else e.setAttribute(k, v === true ? "" : String(v));
      }
    }
    add(e, Array.prototype.slice.call(arguments, 2));
    return e;
  }

  function frag() {
    var f = document.createDocumentFragment();
    add(f, Array.prototype.slice.call(arguments));
    return f;
  }

  function button(opts) {
    return h("button", {
      type: "button",
      class: "btn" + (opts.variant ? " btn-" + opts.variant : "") + (opts.size ? " btn-" + opts.size : ""),
      disabled: !!opts.disabled,
      "data-testid": opts.testid,
      "data-fkey": opts.testid,
      title: opts.title,
      onclick: opts.onclick
    }, opts.iconName ? global.icon(opts.iconName) : null, opts.label ? h("span", { text: opts.label }) : null,
      opts.trailingIcon ? global.icon(opts.trailingIcon, "icon-sm") : null);
  }

  function iconButton(opts) {
    return h("button", {
      type: "button",
      class: "iconbtn",
      "aria-label": opts.label,
      title: opts.label,
      "data-testid": opts.testid,
      "data-fkey": opts.testid,
      disabled: !!opts.disabled,
      onclick: opts.onclick
    }, global.icon(opts.iconName, opts.iconClass));
  }

  /* <select> with an id-stable testid; options are {value, text} pairs. */
  function selectField(opts) {
    return h("select", {
      class: "field",
      name: opts.testid,
      "aria-label": opts.label,
      "data-testid": opts.testid,
      "data-fkey": opts.testid,
      style: opts.style,
      onchange: opts.onchange
    }, opts.options.map(function (o) {
      return h("option", { value: o.value, selected: opts.value === o.value ? true : null, text: o.text });
    }));
  }

  function banner(text, testid, onDismiss) {
    if (!text) return null;
    return h("div", { class: "banner", role: "status", "data-testid": testid },
      global.icon("check", "icon-sm"),
      h("span", { text: text }),
      h("span", { class: "spacer" }),
      iconButton({ iconName: "close", label: "Dismiss", iconClass: "icon-sm",
        testid: testid + "-dismiss", onclick: onDismiss }));
  }

  function emptyState(iconName, title, note) {
    return h("div", { class: "empty" }, global.icon(iconName),
      h("strong", { text: title }), h("span", { text: note }));
  }

  /* ---------------------------------------------------------- focus keeping */

  function isTextField(e) {
    return e && (e.tagName === "TEXTAREA"
      || (e.tagName === "INPUT" && /^(text|search|email|url)$/.test(e.type)));
  }

  function captureFocus() {
    var a = document.activeElement;
    if (!a || !a.dataset || !a.dataset.fkey) return null;
    var snap = { key: a.dataset.fkey };
    if (isTextField(a)) {
      /* what the user has typed wins over the state we are about to paint,
         so a slow round trip can never rewind the field under the caret */
      snap.value = a.value;
      snap.start = a.selectionStart;
      snap.end = a.selectionEnd;
    }
    return snap;
  }

  function restoreFocus(snap) {
    if (!snap) return;
    var target = document.querySelector('[data-fkey="' + snap.key + '"]');
    if (!target) return;
    target.focus();
    if (typeof snap.value === "string" && isTextField(target)) {
      if (target.value !== snap.value) target.value = snap.value;
      if (typeof target.setSelectionRange === "function") {
        try { target.setSelectionRange(snap.start, snap.end); } catch (e) { /* ignore */ }
      }
    }
  }

  /* ---------------------------------------------------------- transport */

  function applyState(state) {
    App.state = state;
    render();
  }

  /* Actions are sent strictly one at a time. The browser is free to reorder
     parallel fetches, and an out-of-order keystroke would corrupt the server
     state, so the queue is the thing that makes a typed string reproducible. */
  var queue = [];
  var draining = false;

  var COALESCE = {
    "board.search": 1, "board.draftTitle": 1,
    "board.draftComment": 1, "board.composeField": 1
  };

  function send(job) {
    return fetch("/api/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: job.action, payload: job.payload, seq: job.seq })
    }).then(function (r) { return r.json(); }).then(function (data) {
      if (data.error) throw new Error(data.error);
      App.applied = data.seq;
      applyState(data.state);
    }).catch(function (err) {
      console.error("[lantern] action failed", job.action, err);
    });
  }

  function drain() {
    if (draining) return;
    var job = queue.shift();
    if (!job) {
      delete document.body.dataset.busy;
      return;
    }
    draining = true;
    send(job).then(function () {
      draining = false;
      job.done();
      drain();
    });
  }

  function act(action, payload) {
    var job = { action: action, payload: payload || {}, seq: ++App.seq };
    job.promise = new Promise(function (resolve) { job.done = resolve; });
    /* Coalesce repeated keystrokes on the same field: only the latest value
       matters, and collapsing them keeps the queue shallow while typing. */
    var last = queue.length ? queue[queue.length - 1] : null;
    if (last && last.action === action && COALESCE[action]
        && last.payload.field === job.payload.field) {
      queue[queue.length - 1] = job;
    } else {
      queue.push(job);
    }
    App.queue = queue;
    document.body.dataset.busy = "1";
    drain();
    return job.promise;
  }

  /* Test/agent hook: resolves once every queued action has been applied. */
  function settled() {
    if (!queue.length && !draining) return Promise.resolve();
    return new Promise(function (resolve) {
      var tick = function () {
        if (!queue.length && !draining) resolve();
        else requestAnimationFrame(tick);
      };
      tick();
    });
  }

  function load() {
    return fetch("/__state").then(function (r) { return r.json(); }).then(applyState);
  }

  /* ---------------------------------------------------------- render */

  function render() {
    var s = App.state;
    if (!s) return;
    var snap = captureFocus();

    document.body.dataset.screen = "board";
    document.getElementById("username").textContent = s.meta.user.name;

    var root = document.getElementById("screen");
    var overlay = document.getElementById("overlay");
    var builder = App.screens.board;
    root.replaceChildren(builder ? builder(s) : document.createTextNode(""));

    var dialog = App.screens.dialog ? App.screens.dialog(s) : null;
    overlay.replaceChildren.apply(overlay, dialog ? [dialog] : []);

    restoreFocus(snap);
  }

  global.App = App;
  global.h = h;
  global.frag = frag;
  global.act = act;
  global.settled = settled;
  global.load = load;
  global.render = render;
  global.button = button;
  global.iconButton = iconButton;
  global.selectField = selectField;
  global.banner = banner;
  global.emptyState = emptyState;
})(window);
