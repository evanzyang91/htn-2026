/* Settings surface: a short multi-step flow.
   Edit fields -> Save changes -> confirmation dialog -> success state.
   The dialog gives the skill graph a modal node with a confirm/cancel branch. */
(function (global) {
  "use strict";

  var TIMEZONES = ["UTC", "America/Toronto", "America/Los_Angeles", "Europe/Lisbon", "Asia/Tokyo"];
  var DENSITIES = ["comfortable", "cozy", "compact"];

  var FIELDS = [
    { key: "displayName", label: "Display name", hint: "Shown on messages you send.", kind: "text" },
    { key: "timezone", label: "Time zone", hint: "Used for scheduled digests.", kind: "select", options: TIMEZONES },
    { key: "density", label: "List density", hint: "How tightly rows are packed.", kind: "select", options: DENSITIES },
    { key: "notifyEmail", label: "Email notifications", hint: "Send an email for new conversations.", kind: "toggle" },
    { key: "notifyDesktop", label: "Desktop notifications", hint: "Show a desktop alert while the console is open.", kind: "toggle" },
    { key: "weeklyDigest", label: "Weekly digest", hint: "One summary message every week.", kind: "toggle" },
    { key: "autoArchive", label: "Auto-archive read mail", hint: "Archive a conversation once it has been read.", kind: "toggle" }
  ];

  var GROUPS = [
    { title: "Profile", note: "Identity and locale for this account.", keys: ["displayName", "timezone"] },
    { title: "Appearance", note: "How lists and tables are rendered.", keys: ["density"] },
    { title: "Notifications", note: "Choose what the console is allowed to interrupt you for.", keys: ["notifyEmail", "notifyDesktop", "weeklyDigest", "autoArchive"] }
  ];

  function fieldByKey(key) {
    return FIELDS.filter(function (f) { return f.key === key; })[0];
  }

  function changedKeys(s) {
    var draft = s.ui.settings.draft, saved = s.settings;
    return Object.keys(draft).filter(function (k) { return draft[k] !== saved[k]; });
  }

  function display(value) {
    if (value === true) return "On";
    if (value === false) return "Off";
    return String(value);
  }

  function control(s, f) {
    var draft = s.ui.settings.draft;
    if (f.kind === "toggle") {
      return global.toggle({
        checked: !!draft[f.key], label: f.label, testid: "set-" + f.key,
        onchange: function (ev) { act("settings.field", { field: f.key, value: ev.target.checked }); }
      });
    }
    if (f.kind === "select") {
      return h("select", {
        class: "field", name: "set-" + f.key, "aria-label": f.label,
        "data-testid": "set-" + f.key, "data-fkey": "set-" + f.key,
        onchange: function (ev) { act("settings.field", { field: f.key, value: ev.target.value }); }
      }, f.options.map(function (opt) {
        return h("option", { value: opt, selected: draft[f.key] === opt ? true : null, text: opt });
      }));
    }
    return h("input", {
      class: "field", type: "text", name: "set-" + f.key,
      value: draft[f.key], "aria-label": f.label,
      "data-testid": "set-" + f.key, "data-fkey": "set-" + f.key,
      oninput: function (ev) { act("settings.field", { field: f.key, value: ev.target.value }); }
    });
  }

  function group(s, g) {
    return h("section", { class: "card", "data-testid": "card-" + g.title.toLowerCase() },
      h("div", { class: "card-head" }, h("h2", { text: g.title }), h("p", { text: g.note })),
      h("div", { class: "card-body" }, g.keys.map(function (key) {
        var f = fieldByKey(key);
        return h("div", { class: "setrow", "data-testid": "setrow-" + key },
          h("div", null,
            h("div", { class: "label", text: f.label }),
            h("div", { class: "hint", text: f.hint })),
          control(s, f));
      })));
  }

  function successPanel(s) {
    if (!s.ui.settings.success) return null;
    return h("div", { class: "success", role: "status", "data-testid": "settings-success" },
      global.icon("check"),
      h("div", null,
        h("strong", { text: "Settings saved" }),
        h("span", { text: "Your preferences are now live for " + s.settings.displayName + "." })),
      h("span", { class: "spacer" }),
      global.iconButton({ iconName: "close", label: "Dismiss", iconClass: "icon-sm",
        testid: "settings-success-dismiss", onclick: function () { act("settings.dismissSuccess"); } }));
  }

  function screen(s) {
    var changed = changedKeys(s);
    return h("div", { class: "settings", "data-testid": "screen-settings" },
      h("div", { class: "settings-inner" },
        h("div", { class: "set-head" }, global.icon("gear", "icon-lg"), h("h1", { text: "Settings" })),
        h("p", { class: "set-lede", text: "Preferences for this console. Changes are not applied until you confirm them." }),
        successPanel(s),
        GROUPS.map(function (g) { return group(s, g); }),
        h("div", { class: "savebar" },
          h("span", { class: "note", "data-testid": "settings-dirty",
            text: changed.length
              ? changed.length + " unsaved change" + (changed.length === 1 ? "" : "s")
              : "No unsaved changes" }),
          h("span", { class: "spacer" }),
          global.button({ label: "Discard", testid: "settings-discard", disabled: changed.length === 0,
            onclick: function () { act("settings.discard"); } }),
          global.button({ label: "Save changes", iconName: "check", variant: "primary",
            testid: "settings-save", disabled: changed.length === 0,
            onclick: function () { act("settings.save"); } }))));
  }

  function dialog(s) {
    if (!s.ui.settings.dialogOpen) return null;
    var changed = changedKeys(s);
    var draft = s.ui.settings.draft, saved = s.settings;
    return h("div", { class: "scrim", "data-testid": "scrim" },
      h("div", { class: "dialog", role: "dialog", "aria-modal": "true",
        "aria-label": "Apply settings changes", "data-testid": "confirm-dialog" },
        h("div", { class: "dialog-head" },
          global.icon("alert", "icon-lg"),
          h("h2", { text: "Apply " + changed.length + " change" + (changed.length === 1 ? "" : "s") + "?" })),
        h("div", { class: "dialog-body" },
          h("div", { text: "These preferences will be applied to your account immediately." }),
          h("ul", { "data-testid": "confirm-list" }, changed.map(function (k) {
            var f = fieldByKey(k);
            return h("li", null,
              h("span", { class: "k", text: f.label }),
              h("span", { text: ": " + display(saved[k]) + " → " + display(draft[k]) }));
          }))),
        h("div", { class: "dialog-foot" },
          h("span", { class: "spacer" }),
          global.button({ label: "Cancel", testid: "confirm-cancel",
            onclick: function () { act("settings.cancel"); } }),
          global.button({ label: "Confirm and save", iconName: "check", variant: "primary",
            testid: "confirm-save", onclick: function () { act("settings.confirm"); } }))));
  }

  global.App.screens.settings = screen;
  global.App.screens.dialog = dialog;
})(window);
