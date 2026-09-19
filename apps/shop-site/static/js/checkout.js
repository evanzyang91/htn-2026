/* Checkout: a form panel inside the drawer, then a confirm dialog.
   The dialog gives the skill graph a modal node with a confirm/back branch. */
(function (global) {
  "use strict";

  function shippingFee(s) {
    var chosen = s.ui.form.shipping;
    var fee = 0;
    s.meta.shipping.forEach(function (opt) { if (opt.id === chosen) fee = opt.fee; });
    return fee;
  }

  function formComplete(s) {
    var f = s.ui.form;
    return f.name.trim() !== "" && f.email.trim() !== "" && f.address.trim() !== "";
  }

  function textField(s, key, label, placeholder) {
    return h("label", { class: "formrow" },
      h("span", { class: "form-label", text: label }),
      h("input", {
        class: "field", type: "text", name: "form-" + key,
        placeholder: placeholder, value: s.ui.form[key],
        "aria-label": label, "data-testid": "form-" + key, "data-fkey": "form-" + key,
        oninput: function (ev) { act("checkout.field", { field: key, value: ev.target.value }); }
      }));
  }

  function panel(s) {
    var lines = s.cart.lines;
    var count = lines.reduce(function (n, line) { return n + line.qty; }, 0);
    var sub = global.cartSubtotal(s);
    var fee = shippingFee(s);

    return frag(
      h("div", { class: "drawer-body", "data-testid": "checkout-panel" },
        global.button({ label: "Back to cart", iconName: "back", size: "sm", testid: "checkout-back",
          onclick: function () { act("checkout.back"); } }),
        h("div", { class: "summary", "data-testid": "checkout-summary",
          text: count + " item" + (count === 1 ? "" : "s") + " · subtotal " + global.money(sub) }),
        textField(s, "name", "Full name", "Jane Mariner"),
        textField(s, "email", "Email", "jane@example.test"),
        textField(s, "address", "Address", "12 Pier Road, Port Town"),
        h("label", { class: "formrow" },
          h("span", { class: "form-label", text: "Shipping speed" }),
          h("select", {
            class: "field", name: "form-shipping", "aria-label": "Shipping speed",
            "data-testid": "form-shipping", "data-fkey": "form-shipping",
            onchange: function (ev) { act("checkout.field", { field: "shipping", value: ev.target.value }); }
          }, s.meta.shipping.map(function (opt) {
            return h("option", { value: opt.id, selected: s.ui.form.shipping === opt.id ? true : null,
              text: opt.label });
          }))),
        h("div", { class: "formrow" },
          global.checkbox({
            checked: s.ui.form.save, label: "Save these details", testid: "form-save",
            onchange: function (ev) { act("checkout.field", { field: "save", value: ev.target.checked }); }
          }))),
      h("div", { class: "drawer-foot" },
        h("div", { class: "totalrow small" },
          h("span", { text: "Shipping" }),
          h("span", { class: "amount", "data-testid": "checkout-shipping-fee", text: global.money(fee) })),
        h("div", { class: "totalrow" },
          h("span", { text: "Total" }),
          h("span", { class: "amount", "data-testid": "checkout-total", text: global.money(sub + fee) })),
        global.button({
          label: "Place order", variant: "primary", size: "lg",
          testid: "place-order", disabled: !formComplete(s) || lines.length === 0,
          onclick: function () { act("checkout.submit"); }
        })));
  }

  function dialog(s) {
    if (!s.ui.dialogOpen) return null;
    var count = s.cart.lines.reduce(function (n, line) { return n + line.qty; }, 0);
    var sub = global.cartSubtotal(s);
    var fee = shippingFee(s);
    return h("div", { class: "scrim", "data-testid": "scrim" },
      h("div", { class: "dialog", role: "dialog", "aria-modal": "true",
        "aria-label": "Confirm this order", "data-testid": "confirm-dialog" },
        h("div", { class: "dialog-head" },
          global.icon("alert", "icon-lg"),
          h("h2", { text: "Place this order?" })),
        h("div", { class: "dialog-body" },
          h("div", { text: "The order will be placed immediately." }),
          h("ul", { "data-testid": "confirm-list" },
            h("li", null, h("span", { class: "k", text: "Items" }),
              h("span", { text: ": " + count })),
            h("li", null, h("span", { class: "k", text: "Total" }),
              h("span", { text: ": " + global.money(sub + fee) })),
            h("li", null, h("span", { class: "k", text: "Ship to" }),
              h("span", { text: ": " + s.ui.form.name.trim() + ", " + s.ui.form.address.trim() })))),
        h("div", { class: "dialog-foot" },
          h("span", { class: "spacer" }),
          global.button({ label: "Back", testid: "confirm-back",
            onclick: function () { act("checkout.cancel"); } }),
          global.button({ label: "Confirm order", iconName: "check", variant: "primary",
            testid: "confirm-order", onclick: function () { act("checkout.confirm"); } }))));
  }

  global.checkoutPanel = panel;
  global.App.screens.dialog = dialog;
})(window);
