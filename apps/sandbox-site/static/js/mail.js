/* Mail surface: folders, search, multi-select, archive, label, reader, compose. */
(function (global) {
  "use strict";

  var LABEL_SWATCH = { Work: "#2f6df6", Personal: "#2c9c5b", Finance: "#d08a20", Travel: "#7a52d6" };

  function visibleMessages(s) {
    var ui = s.ui.mail;
    var q = ui.search.trim().toLowerCase();
    var list = s.mail.messages.filter(function (m) {
      return ui.folder === "archived" ? m.archived : !m.archived;
    });
    if (q) {
      list = list.filter(function (m) {
        return (m.sender + " " + m.subject + " " + m.body + " " + m.labels.join(" "))
          .toLowerCase().indexOf(q) !== -1;
      });
    }
    return list;
  }

  function initials(name) {
    var parts = name.split(/\s+/).filter(Boolean);
    return ((parts[0] || "")[0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
  }

  /* ------------------------------------------------------------- sidebar */

  function sidebar(s) {
    var ui = s.ui.mail;
    var inboxUnread = s.mail.messages.filter(function (m) { return !m.archived && m.unread; }).length;
    var archivedCount = s.mail.messages.filter(function (m) { return m.archived; }).length;

    function folder(id, label, iconName, count) {
      return h("button", {
        type: "button", class: "folder",
        "aria-current": ui.folder === id ? "true" : "false",
        "data-testid": "folder-" + id, "data-fkey": "folder-" + id,
        onclick: function () { act("mail.folder", { folder: id }); }
      }, global.icon(iconName), h("span", { text: label }),
        count ? h("span", { class: "count", text: String(count) }) : null);
    }

    return h("aside", { class: "mail-side" },
      global.button({
        label: "Compose", iconName: "pencil", variant: "primary", size: "lg",
        testid: "compose-open", onclick: function () { act("mail.composeOpen"); }
      }),
      h("div", { class: "side-group" },
        h("div", { class: "side-title", text: "Folders" }),
        folder("inbox", "Inbox", "inbox", inboxUnread),
        folder("archived", "Archived", "archive", archivedCount),
        folder("sent", "Sent", "sent", s.mail.sent.length)),
      h("div", { class: "side-group" },
        h("div", { class: "side-title", text: "Labels" }),
        s.mail.labels.map(function (label) {
          return h("div", { class: "side-label", "data-testid": "sidelabel-" + label },
            h("span", { class: "swatch", style: "background:" + LABEL_SWATCH[label] }),
            h("span", { text: label }));
        })));
  }

  /* ------------------------------------------------------------- toolbar */

  function toolbar(s, list) {
    var ui = s.ui.mail;
    var ids = list.map(function (m) { return m.id; });
    var selected = ui.selected.length;
    var allSelected = ids.length > 0 && ids.every(function (id) { return ui.selected.indexOf(id) !== -1; });
    var someSelected = selected > 0 && !allSelected;
    var isSent = ui.folder === "sent";

    var labelMenu = ui.labelMenuOpen ? h("div", { class: "menu", style: "top:36px;left:0", "data-testid": "label-menu" },
      h("div", { class: "menu-label", text: "Apply label" }),
      s.mail.labels.map(function (label) {
        return h("button", {
          type: "button", class: "menuitem",
          "data-testid": "label-option-" + label, "data-fkey": "label-option-" + label,
          onclick: function () { act("mail.label", { label: label }); }
        }, h("span", { class: "swatch", style: "background:" + LABEL_SWATCH[label] }), h("span", { text: label }));
      })) : null;

    return h("div", { class: "mail-toolbar" },
      global.checkbox({
        checked: allSelected, indeterminate: someSelected,
        label: "Select all conversations", testid: "select-all",
        onchange: function (ev) { act("mail.selectAll", { checked: ev.target.checked, ids: ids }); }
      }),
      h("span", { class: "sep" }),
      global.button({
        label: "Archive", iconName: "archive", testid: "archive-btn",
        disabled: selected === 0 || isSent || ui.folder === "archived",
        onclick: function () { act("mail.archive"); }
      }),
      h("div", { style: "position:relative" },
        global.button({
          label: "Label", iconName: "tag", trailingIcon: "chevron", testid: "label-btn",
          disabled: selected === 0 || isSent,
          onclick: function () { act("mail.toggleLabelMenu"); }
        }), labelMenu),
      h("span", { class: "sep" }),
      h("span", { class: "selcount", "data-testid": "sel-count",
        text: selected ? selected + " selected" : list.length + " conversations" }),
      h("span", { class: "spacer" }),
      h("div", { class: "searchbox" },
        global.icon("search", "icon-sm"),
        h("input", {
          class: "field", type: "search", name: "mail-search",
          placeholder: "Search mail", value: ui.search,
          "aria-label": "Search mail", "data-testid": "mail-search", "data-fkey": "mail-search",
          style: "width:240px",
          oninput: function (ev) { act("mail.search", { q: ev.target.value }); }
        })));
  }

  /* ------------------------------------------------------------- list */

  function messageRow(s, m) {
    var ui = s.ui.mail;
    var isSelected = ui.selected.indexOf(m.id) !== -1;
    var cls = "msgrow" + (m.unread ? " unread" : "") + (isSelected ? " selected" : "")
      + (ui.openId === m.id ? " open" : "");

    return h("div", { class: cls, "data-testid": "msg-" + m.id, "data-msgid": m.id },
      global.checkbox({
        checked: isSelected, label: "Select " + m.subject, testid: "msgcheck-" + m.id,
        onchange: function () { act("mail.toggleSelect", { id: m.id }); }
      }),
      h("button", {
        type: "button", class: "row-open",
        "data-testid": "msgopen-" + m.id, "data-fkey": "msgopen-" + m.id,
        onclick: function () { act("mail.open", { id: m.id }); }
      },
        h("div", { class: "sender", text: m.sender }),
        h("div", { class: "subject", text: m.subject }),
        h("div", { class: "preview", text: m.preview }),
        m.labels.length ? h("div", { class: "tags" }, m.labels.map(function (l) {
          return h("span", { class: "tagchip tag-" + l, text: l });
        })) : null),
      h("div", { class: "meta" },
        h("span", { class: "date", text: m.date }),
        m.unread ? h("span", { class: "unread-dot", "aria-label": "Unread" }) : null));
  }

  function sentRow(s, m) {
    return h("div", { class: "msgrow", "data-testid": "sent-" + m.id },
      h("span", { style: "width:16px" }),
      h("div", null,
        h("div", { class: "sender", text: "To: " + m.to }),
        h("div", { class: "subject", text: m.subject || "(no subject)" }),
        h("div", { class: "preview", text: m.body || "(no body)" })),
      h("div", { class: "meta" }, h("span", { class: "date", text: m.date })));
  }

  /* ------------------------------------------------------------- reader */

  function reader(s, m) {
    return h("section", { class: "reader", "data-testid": "reader" },
      h("div", { class: "reader-head" },
        h("h2", { text: m.subject, "data-testid": "reader-subject" }),
        h("div", { class: "reader-from" },
          h("span", { class: "avatar", text: initials(m.sender) }),
          h("div", null,
            h("div", { class: "who", text: m.sender }),
            h("div", { class: "addr", text: m.senderEmail })),
          h("span", { class: "spacer" }),
          h("span", { class: "date", style: "font-size:12.5px;color:var(--ink-400)", text: m.date }),
          global.iconButton({
            iconName: "close", label: "Close reading pane", testid: "reader-close",
            onclick: function () { act("mail.close"); }
          }))),
      h("div", { class: "reader-body", "data-testid": "reader-body", text: m.body }),
      h("div", { class: "reader-foot" },
        m.labels.map(function (l) { return h("span", { class: "tagchip tag-" + l, text: l }); })));
  }

  /* ------------------------------------------------------------- compose */

  function compose(s) {
    var c = s.ui.mail.compose;
    if (!c.open) return null;
    function field(name, label, placeholder) {
      return h("input", {
        class: "field", type: "text", name: "compose-" + name,
        placeholder: placeholder, value: c[name],
        "aria-label": label, "data-testid": "compose-" + name, "data-fkey": "compose-" + name,
        oninput: function (ev) { act("mail.composeField", { field: name, value: ev.target.value }); }
      });
    }
    return h("section", { class: "compose", "data-testid": "compose-window", role: "dialog",
      "aria-label": "New message" },
      h("div", { class: "compose-head" },
        global.icon("pencil", "icon-sm"),
        h("span", { text: "New message" }),
        h("span", { class: "spacer" }),
        global.iconButton({ iconName: "close", label: "Close compose", testid: "compose-close",
          onclick: function () { act("mail.composeClose"); } })),
      h("div", { class: "compose-body" },
        field("to", "Recipient", "To"),
        field("subject", "Subject", "Subject"),
        h("textarea", {
          class: "field", rows: 6, name: "compose-body",
          placeholder: "Write your message", "aria-label": "Message body",
          "data-testid": "compose-body", "data-fkey": "compose-body",
          oninput: function (ev) { act("mail.composeField", { field: "body", value: ev.target.value }); }
        }, c.body)),
      h("div", { class: "compose-foot" },
        global.button({
          label: "Send", iconName: "send", variant: "primary", testid: "compose-send",
          disabled: c.to.trim() === "",
          onclick: function () { act("mail.send"); }
        }),
        h("span", { class: "spacer" }),
        global.button({ label: "Discard", iconName: "close", testid: "compose-discard",
          onclick: function () { act("mail.composeClose"); } })));
  }

  /* ------------------------------------------------------------- screen */

  function screen(s) {
    var ui = s.ui.mail;
    var isSent = ui.folder === "sent";
    var list = isSent ? [] : visibleMessages(s);
    var open = ui.openId ? s.mail.messages.filter(function (m) { return m.id === ui.openId; })[0] : null;

    var rows;
    if (isSent) {
      rows = s.mail.sent.length
        ? s.mail.sent.map(function (m) { return sentRow(s, m); })
        : global.emptyState("sent", "Nothing sent yet", "Messages you send will appear here.");
    } else if (list.length) {
      rows = list.map(function (m) { return messageRow(s, m); });
    } else {
      rows = global.emptyState("search", "No conversations match",
        ui.search ? 'Nothing matches "' + ui.search + '".' : "This folder is empty.");
    }

    var body = h("div", { class: "mail-body" + (open && !isSent ? " split" : "") },
      h("div", { class: "msglist", "data-testid": "msglist" }, rows),
      open && !isSent ? reader(s, open) : null);
    return frag(
      h("div", { class: "mail", "data-testid": "screen-mail" },
        sidebar(s),
        h("div", { class: "mail-main" },
          toolbar(s, list),
          h("div", { class: "mail-bannerbar" },
            global.banner(ui.banner, "mail-banner", function () { act("mail.dismissBanner"); })),
          body)),
      compose(s));
  }

  global.App.screens.mail = screen;
})(window);
