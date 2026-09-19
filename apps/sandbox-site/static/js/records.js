/* Records surface: filter, column sort, inline edit, bulk action, export.
   Deliberately a different visual vocabulary from Mail: dense data table,
   sticky sortable headers, zebra rows, status pills, teal accent.        */
(function (global) {
  "use strict";

  var COLUMNS = [
    { key: "name", label: "Name", cls: "name" },
    { key: "owner", label: "Owner", cls: "" },
    { key: "category", label: "Category", cls: "" },
    { key: "status", label: "Status", cls: "" },
    { key: "priority", label: "Priority", cls: "" },
    { key: "records", label: "Records", cls: "num", num: true },
    { key: "updated", label: "Updated", cls: "date" }
  ];

  var BULK_STATUSES = ["Active", "Paused", "Draft", "Archived"];
  var PRIORITY_RANK = { High: 0, Medium: 1, Low: 2 };

  function visibleRows(s) {
    var ui = s.ui.records;
    var q = ui.filter.trim().toLowerCase();
    var rows = s.records.rows.slice();
    if (q) {
      rows = rows.filter(function (r) {
        return (r.name + " " + r.owner + " " + r.category + " " + r.status + " " + r.priority)
          .toLowerCase().indexOf(q) !== -1;
      });
    }
    var key = ui.sortKey, dir = ui.sortDir === "desc" ? -1 : 1;
    rows.sort(function (a, b) {
      var x = a[key], y = b[key];
      if (key === "priority") { x = PRIORITY_RANK[x]; y = PRIORITY_RANK[y]; }
      if (typeof x === "number" && typeof y === "number") {
        if (x !== y) return (x - y) * dir;
      } else {
        var c = String(x).localeCompare(String(y));
        if (c !== 0) return c * dir;
      }
      return a.id.localeCompare(b.id); /* stable tie-break keeps order deterministic */
    });
    return rows;
  }

  function header(s, rows) {
    var ui = s.ui.records;
    var total = s.records.rows.length;
    return h("div", { class: "rec-head" },
      global.icon("table", "icon-lg"),
      h("div", null,
        h("h1", { text: "Records" }),
        h("div", { class: "sub", "data-testid": "rec-subtitle",
          text: rows.length + " of " + total + " datasets" })),
      h("span", { class: "spacer" }),
      global.button({
        label: ui.selected.length ? "Export " + ui.selected.length + " selected" : "Export CSV",
        iconName: "download", variant: "primary", testid: "export-btn",
        onclick: function () {
          var ids = ui.selected.length ? ui.selected.slice() : rows.map(function (r) { return r.id; });
          act("records.export", { ids: ids });
        }
      }));
  }

  function tools(s, rows) {
    var ui = s.ui.records;
    var menu = ui.bulkMenuOpen ? h("div", { class: "menu", style: "top:36px;left:0", "data-testid": "bulk-menu" },
      h("div", { class: "menu-label", text: "Set status to" }),
      BULK_STATUSES.map(function (st) {
        return h("button", {
          type: "button", class: "menuitem",
          "data-testid": "bulk-option-" + st, "data-fkey": "bulk-option-" + st,
          onclick: function () { act("records.bulkStatus", { status: st }); }
        }, h("span", { class: "pill pill-" + st.toLowerCase(), text: st }));
      })) : null;

    return h("div", { class: "rec-tools" },
      h("div", { class: "searchbox" },
        global.icon("filter", "icon-sm"),
        h("input", {
          class: "field", type: "search", name: "rec-filter",
          placeholder: "Filter records", value: ui.filter,
          "aria-label": "Filter records", "data-testid": "rec-filter", "data-fkey": "rec-filter",
          style: "width:260px",
          oninput: function (ev) { act("records.filter", { q: ev.target.value }); }
        })),
      h("span", { class: "sep" }),
      h("div", { style: "position:relative" },
        global.button({
          label: "Bulk actions", iconName: "layers", trailingIcon: "chevron",
          testid: "bulk-btn", disabled: ui.selected.length === 0,
          onclick: function () { act("records.toggleBulkMenu"); }
        }), menu),
      h("span", { class: "selcount", "data-testid": "rec-selcount",
        text: ui.selected.length + " selected" }),
      h("span", { class: "spacer" }),
      h("span", { class: "selcount", text: "Sorted by " + ui.sortKey + " · " + ui.sortDir }));
  }

  function nameCell(s, r) {
    var ui = s.ui.records;
    if (ui.editingId === r.id) {
      return h("div", { class: "editcell" },
        h("input", {
          class: "field", type: "text", name: "edit-name",
          value: ui.editValue, "aria-label": "Edit name",
          "data-testid": "edit-input", "data-fkey": "edit-input",
          oninput: function (ev) { act("records.editValue", { value: ev.target.value }); },
          onkeydown: function (ev) {
            if (ev.key === "Enter") act("records.editCommit");
            if (ev.key === "Escape") act("records.editCancel");
          }
        }),
        global.button({ label: "Save", size: "sm", variant: "primary", testid: "edit-save",
          onclick: function () { act("records.editCommit"); } }),
        global.button({ label: "Cancel", size: "sm", testid: "edit-cancel",
          onclick: function () { act("records.editCancel"); } }));
    }
    return h("div", { class: "namecell" }, h("span", { class: "txt", text: r.name }));
  }

  function table(s, rows) {
    var ui = s.ui.records;
    var ids = rows.map(function (r) { return r.id; });
    var allSelected = ids.length > 0 && ids.every(function (id) { return ui.selected.indexOf(id) !== -1; });
    var someSelected = ui.selected.length > 0 && !allSelected;

    var head = h("tr", null,
      h("th", { class: "col-check" }, global.checkbox({
        checked: allSelected, indeterminate: someSelected,
        label: "Select all records", testid: "rec-select-all",
        onchange: function (ev) { act("records.selectAll", { checked: ev.target.checked, ids: ids }); }
      })),
      COLUMNS.map(function (col) {
        var sorted = ui.sortKey === col.key;
        var iconName = sorted ? (ui.sortDir === "asc" ? "sort-asc" : "sort-desc") : "sort-none";
        return h("th", { class: col.num ? "num" : "" },
          h("button", {
            type: "button", class: "th-sort", "data-sorted": sorted ? "true" : "false",
            "data-testid": "sort-" + col.key, "data-fkey": "sort-" + col.key,
            onclick: function () { act("records.sort", { key: col.key }); }
          }, h("span", { text: col.label }), global.icon(iconName)));
      }),
      h("th", { class: "col-edit" }));

    var body = rows.map(function (r) {
      var isSelected = ui.selected.indexOf(r.id) !== -1;
      return h("tr", { class: isSelected ? "selected" : "", "data-testid": "row-" + r.id },
        h("td", { class: "col-check" }, global.checkbox({
          checked: isSelected, label: "Select " + r.name, testid: "reccheck-" + r.id,
          onchange: function () { act("records.toggleSelect", { id: r.id }); }
        })),
        h("td", { class: "name", "data-testid": "cell-name-" + r.id }, nameCell(s, r)),
        h("td", { text: r.owner }),
        h("td", { text: r.category }),
        h("td", null, h("span", { class: "pill pill-" + r.status.toLowerCase(),
          "data-testid": "status-" + r.id, text: r.status })),
        h("td", { text: r.priority }),
        h("td", { class: "num", text: r.records.toLocaleString("en-US") }),
        h("td", { class: "date", text: r.updated }),
        h("td", { class: "col-edit" }, global.iconButton({
          iconName: "pencil", label: "Edit " + r.name, iconClass: "icon-sm",
          testid: "editbtn-" + r.id, disabled: ui.editingId === r.id,
          onclick: function () { act("records.editStart", { id: r.id }); }
        })));
    });

    return h("table", { class: "dtable", "data-testid": "rec-table" },
      h("thead", null, head),
      h("tbody", null, body));
  }

  function screen(s) {
    var ui = s.ui.records;
    var rows = visibleRows(s);
    var lastExport = s.records.exports.length
      ? s.records.exports[s.records.exports.length - 1] : null;

    return h("div", { class: "records", "data-testid": "screen-records" },
      header(s, rows),
      tools(s, rows),
      h("div", { class: "rec-bannerbar" },
        global.banner(ui.banner, "rec-banner", function () { act("records.dismissBanner"); })),
      h("div", { class: "tablewrap" },
        rows.length ? table(s, rows)
          : global.emptyState("search", "No records match", 'Nothing matches "' + ui.filter + '".')),
      h("div", { class: "rec-foot" },
        h("span", { "data-testid": "rec-footer-count",
          text: "Showing " + rows.length + " of " + s.records.rows.length + " records" }),
        h("span", { class: "spacer" }),
        h("span", { "data-testid": "rec-exports",
          text: "Exports this session: " + s.records.exports.length }),
        lastExport ? h("code", { text: "last=" + lastExport.count + " rows" }) : null));
  }

  global.App.screens.records = screen;
})(window);
