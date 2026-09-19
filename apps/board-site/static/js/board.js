/* Board surface: a four-column kanban with filters, a card move menu, a
   ticket detail panel, a new-ticket modal and an archive-done confirm.
   Deliberately a different visual vocabulary from a mail console or a shop:
   columns of cards, chips, inline menus, and a slide-in detail pane.      */
(function (global) {
  "use strict";

  function columnName(s, colId) {
    for (var i = 0; i < s.board.columns.length; i++) {
      if (s.board.columns[i].id === colId) return s.board.columns[i].name;
    }
    return colId;
  }

  function ticketById(s, tid) {
    return s.board.tickets.filter(function (t) { return t.id === tid; })[0] || null;
  }

  function visibleTickets(s) {
    var ui = s.ui;
    var q = ui.search.trim().toLowerCase();
    return s.board.tickets.filter(function (t) {
      if (ui.assignee !== "all" && t.assignee !== ui.assignee) return false;
      if (ui.priority !== "all" && t.priority !== ui.priority) return false;
      if (q && (t.key + " " + t.title + " " + t.description + " " + t.assignee)
          .toLowerCase().indexOf(q) === -1) return false;
      return true;
    });
  }

  function chip(priority, testid) {
    return h("span", { class: "chip chip-" + priority.toLowerCase(),
      "data-testid": testid, text: priority });
  }

  /* Every filter control carries a visible text label, not just an aria-label,
     so an OCR pass can read what the control is for. */
  function labelled(text, control) {
    return h("label", { class: "ctl" },
      h("span", { class: "ctl-label", text: text }), control);
  }

  /* ------------------------------------------------------------- toolbar */

  function toolbar(s) {
    var ui = s.ui;
    var doneCount = s.board.tickets.filter(function (t) { return t.column === "done"; }).length;

    return h("div", { class: "toolbar" },
      labelled("Search", h("div", { class: "searchbox" },
        global.icon("search", "icon-sm"),
        h("input", {
          class: "field", type: "search", name: "board-search",
          placeholder: "Search tickets", value: ui.search,
          "aria-label": "Search tickets", "data-testid": "board-search", "data-fkey": "board-search",
          style: "width:200px",
          oninput: function (ev) { act("board.search", { q: ev.target.value }); }
        }))),
      labelled("Assignee", global.selectField({
        label: "Filter by assignee", testid: "filter-assignee",
        value: ui.assignee, style: "width:160px",
        options: [{ value: "all", text: "All assignees" }].concat(
          s.board.assignees.map(function (a) { return { value: a, text: a }; })),
        onchange: function (ev) { act("board.filterAssignee", { assignee: ev.target.value }); }
      })),
      labelled("Priority", global.selectField({
        label: "Filter by priority", testid: "filter-priority",
        value: ui.priority, style: "width:130px",
        options: [{ value: "all", text: "All priorities" }].concat(
          s.board.priorities.map(function (p) { return { value: p, text: p }; })),
        onchange: function (ev) { act("board.filterPriority", { priority: ev.target.value }); }
      })),
      h("span", { class: "spacer" }),
      global.button({
        label: "Archive done", iconName: "archive", testid: "archive-done-btn",
        disabled: doneCount === 0,
        onclick: function () { act("board.archiveDialog"); }
      }),
      global.button({
        label: "New ticket", iconName: "plus", variant: "primary", testid: "new-ticket-btn",
        onclick: function () { act("board.composeOpen"); }
      }));
  }

  /* ------------------------------------------------------------- cards */

  function moveMenu(s, t) {
    var others = s.board.columns.filter(function (c) { return c.id !== t.column; });
    return h("div", { class: "movemenu", "data-testid": "move-menu-" + t.id },
      h("div", { class: "menu-label", text: "Move to" }),
      others.map(function (c) {
        return h("button", {
          type: "button", class: "menuitem",
          "data-testid": "move-to-" + t.id + "-" + c.id,
          "data-fkey": "move-to-" + t.id + "-" + c.id,
          onclick: function () { act("board.move", { id: t.id, column: c.id }); }
        }, h("span", { text: c.name }));
      }));
  }

  function card(s, t) {
    var isOpen = s.ui.openId === t.id;
    var menuOpen = s.ui.moveMenuFor === t.id;
    return h("article", { class: "card" + (isOpen ? " open" : ""), "data-testid": "card-" + t.id },
      h("div", { class: "card-top" },
        h("span", { class: "tkey", text: t.key }),
        h("span", { class: "spacer" }),
        chip(t.priority, "card-priority-" + t.id)),
      h("button", {
        type: "button", class: "card-title",
        "data-testid": "card-title-" + t.id, "data-fkey": "card-title-" + t.id,
        title: "Open " + t.key,
        onclick: function () { act("board.open", { id: t.id }); }
      }, h("span", { text: t.title })),
      h("div", { class: "card-foot" },
        h("span", { class: "assignee", "data-testid": "card-assignee-" + t.id, text: t.assignee }),
        h("span", { class: "spacer" }),
        global.button({
          label: "Move", size: "sm", trailingIcon: "chevron", testid: "move-btn-" + t.id,
          onclick: function () { act("board.toggleMoveMenu", { id: t.id }); }
        })),
      menuOpen ? moveMenu(s, t) : null);
  }

  function column(s, c, tickets) {
    return h("section", { class: "col", "data-testid": "col-" + c.id },
      h("div", { class: "col-head", "data-testid": "colhead-" + c.id },
        h("span", { class: "col-name", text: c.name }),
        h("span", { class: "col-count", "data-testid": "colcount-" + c.id,
          text: String(tickets.length) })),
      h("div", { class: "col-stack", "data-testid": "colstack-" + c.id },
        tickets.length
          ? tickets.map(function (t) { return card(s, t); })
          : h("div", { class: "col-empty", text: "No tickets" })));
  }

  /* ------------------------------------------------------------- detail */

  function commentRow(cm) {
    return h("div", { class: "comment", "data-testid": "comment-" + cm.id },
      h("div", { class: "comment-head" },
        h("strong", { text: cm.author }),
        h("span", { class: "comment-date", text: cm.date })),
      h("div", { class: "comment-text", text: cm.text }));
  }

  function detail(s, t) {
    var ui = s.ui;
    return h("aside", { class: "detail", "data-testid": "detail" },
      h("div", { class: "detail-head" },
        h("span", { class: "tkey tkey-lg", "data-testid": "detail-key", text: t.key }),
        chip(t.priority, "detail-priority"),
        h("span", { class: "statuschip", "data-testid": "detail-status", text: columnName(s, t.column) }),
        h("span", { class: "spacer" }),
        global.button({
          label: "Close", size: "sm", iconName: "close", testid: "detail-close",
          onclick: function () { act("board.close"); }
        })),
      h("h2", { class: "detail-title", "data-testid": "detail-title", text: t.title }),

      h("div", { class: "detail-group" },
        h("span", { class: "ctl-label", text: "Title" }),
        h("div", { class: "titlerow" },
          h("input", {
            class: "field", type: "text", name: "detail-title-input",
            value: ui.draft.title, "aria-label": "Edit title",
            "data-testid": "detail-title-input", "data-fkey": "detail-title-input",
            oninput: function (ev) { act("board.draftTitle", { value: ev.target.value }); }
          }),
          global.button({
            label: "Save title", variant: "primary", testid: "detail-save-title",
            disabled: ui.draft.title.trim() === "",
            onclick: function () { act("board.saveTitle"); }
          }))),

      h("div", { class: "detail-group" },
        h("span", { class: "ctl-label", text: "Description" }),
        h("p", { class: "detail-desc", "data-testid": "detail-desc",
          text: t.description || "No description." })),

      h("div", { class: "detail-group" },
        h("span", { class: "ctl-label", text: "Assign to" }),
        global.selectField({
          label: "Assign to", testid: "detail-assign",
          value: t.assignee, style: "width:100%",
          options: s.board.assignees.map(function (a) { return { value: a, text: a }; }),
          onchange: function (ev) { act("board.assign", { assignee: ev.target.value }); }
        })),

      h("div", { class: "detail-group" },
        h("span", { class: "ctl-label",
          text: "Comments (" + t.comments.length + ")" }),
        h("div", { class: "comments", "data-testid": "detail-comments" },
          t.comments.length
            ? t.comments.map(commentRow)
            : h("div", { class: "comments-empty", text: "No comments yet." })),
        h("textarea", {
          class: "field", rows: 3, name: "detail-comment-input",
          placeholder: "Write a comment", "aria-label": "Write a comment",
          "data-testid": "detail-comment-input", "data-fkey": "detail-comment-input",
          oninput: function (ev) { act("board.draftComment", { value: ev.target.value }); }
        }, ui.draft.comment),
        h("div", { class: "commentrow" },
          global.button({
            label: "Add comment", iconName: "comment", testid: "detail-add-comment",
            disabled: ui.draft.comment.trim() === "",
            onclick: function () { act("board.addComment"); }
          }))));
  }

  /* ------------------------------------------------------------- dialogs */

  function composeDialog(s) {
    var c = s.ui.compose;
    function textField(name, label, placeholder) {
      return h("input", {
        class: "field", type: "text", name: "compose-" + name,
        placeholder: placeholder, value: c[name],
        "aria-label": label, "data-testid": "compose-" + name, "data-fkey": "compose-" + name,
        oninput: function (ev) { act("board.composeField", { field: name, value: ev.target.value }); }
      });
    }
    return h("div", { class: "scrim", "data-testid": "scrim" },
      h("div", { class: "dialog", role: "dialog", "aria-modal": "true",
        "aria-label": "New ticket", "data-testid": "compose-dialog" },
        h("div", { class: "dialog-head" },
          global.icon("plus", "icon-lg"),
          h("h2", { text: "New ticket" })),
        h("div", { class: "dialog-body dialog-form" },
          h("label", { class: "formrow" },
            h("span", { class: "ctl-label", text: "Title" }),
            textField("title", "Ticket title", "What needs doing?")),
          h("label", { class: "formrow" },
            h("span", { class: "ctl-label", text: "Description" }),
            h("textarea", {
              class: "field", rows: 4, name: "compose-description",
              placeholder: "Add detail (optional)", "aria-label": "Ticket description",
              "data-testid": "compose-description", "data-fkey": "compose-description",
              oninput: function (ev) { act("board.composeField", { field: "description", value: ev.target.value }); }
            }, c.description)),
          h("label", { class: "formrow" },
            h("span", { class: "ctl-label", text: "Assignee" }),
            global.selectField({
              label: "Ticket assignee", testid: "compose-assignee",
              value: c.assignee,
              options: s.board.assignees.map(function (a) { return { value: a, text: a }; }),
              onchange: function (ev) { act("board.composeField", { field: "assignee", value: ev.target.value }); }
            })),
          h("label", { class: "formrow" },
            h("span", { class: "ctl-label", text: "Priority" }),
            global.selectField({
              label: "Ticket priority", testid: "compose-priority",
              value: c.priority,
              options: s.board.priorities.map(function (p) { return { value: p, text: p }; }),
              onchange: function (ev) { act("board.composeField", { field: "priority", value: ev.target.value }); }
            }))),
        h("div", { class: "dialog-foot" },
          h("span", { class: "spacer" }),
          global.button({ label: "Cancel", testid: "compose-cancel",
            onclick: function () { act("board.composeClose"); } }),
          global.button({
            label: "Create", iconName: "check", variant: "primary", testid: "compose-create",
            disabled: c.title.trim() === "",
            onclick: function () { act("board.create"); }
          }))));
  }

  function archiveDialog(s) {
    var done = s.board.tickets.filter(function (t) { return t.column === "done"; });
    return h("div", { class: "scrim", "data-testid": "scrim" },
      h("div", { class: "dialog", role: "dialog", "aria-modal": "true",
        "aria-label": "Archive done tickets", "data-testid": "archive-dialog" },
        h("div", { class: "dialog-head" },
          global.icon("alert", "icon-lg"),
          h("h2", { text: "Archive " + done.length + " ticket" + (done.length === 1 ? "" : "s") + "?" })),
        h("div", { class: "dialog-body" },
          h("div", { text: "Every ticket in Done moves to the archive and leaves the board." }),
          h("ul", { "data-testid": "archive-list" }, done.map(function (t) {
            return h("li", null,
              h("span", { class: "k", text: t.key }),
              h("span", { text: " " + t.title }));
          }))),
        h("div", { class: "dialog-foot" },
          h("span", { class: "spacer" }),
          global.button({ label: "Keep them", testid: "archive-cancel",
            onclick: function () { act("board.archiveCancel"); } }),
          global.button({ label: "Archive", iconName: "archive", variant: "primary",
            testid: "archive-confirm", onclick: function () { act("board.archiveConfirm"); } }))));
  }

  function dialog(s) {
    if (s.ui.compose.open) return composeDialog(s);
    if (s.ui.dialogOpen) return archiveDialog(s);
    return null;
  }

  /* ------------------------------------------------------------- screen */

  function screen(s) {
    var visible = visibleTickets(s);
    var open = s.ui.openId ? ticketById(s, s.ui.openId) : null;

    return h("div", { class: "board", "data-testid": "screen-board" },
      toolbar(s),
      h("div", { class: "bannerbar" },
        global.banner(s.ui.banner, "board-banner", function () { act("board.dismissBanner"); })),
      h("div", { class: "board-body" + (open ? " split" : "") },
        h("div", { class: "cols" }, s.board.columns.map(function (c) {
          return column(s, c, visible.filter(function (t) { return t.column === c.id; }));
        })),
        open ? detail(s, open) : null));
  }

  global.App.screens.board = screen;
  global.App.screens.dialog = dialog;
})(window);
