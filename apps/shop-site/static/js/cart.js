/* Cart drawer: slides in from the right (present or absent, never animated).
   Line items with quantity steppers, per-line remove, subtotal, checkout.  */
(function (global) {
  "use strict";

  function product(s, pid) {
    return s.catalog.products.filter(function (p) { return p.id === pid; })[0];
  }

  function subtotal(s) {
    return s.cart.lines.reduce(function (sum, line) {
      return sum + product(s, line.productId).price * line.qty;
    }, 0);
  }

  function lineRow(s, line) {
    var p = product(s, line.productId);
    return h("div", { class: "cartline", "data-testid": "line-" + p.id },
      h("div", { class: "cartline-top" },
        h("span", { class: "cartline-name", text: p.name }),
        h("span", { class: "cartline-unit", text: global.money(p.price) + " each" })),
      h("div", { class: "cartline-controls" },
        h("span", { class: "stepper" },
          h("button", {
            type: "button", class: "stepbtn", title: "Decrease quantity",
            "aria-label": "Decrease quantity of " + p.name,
            "data-testid": "qty-minus-" + p.id, "data-fkey": "qty-minus-" + p.id,
            disabled: line.qty <= 1,
            onclick: function () { act("cart.decrement", { productId: p.id }); }
          }, "−"),
          h("span", { class: "qty", "data-testid": "qty-" + p.id, text: String(line.qty) }),
          h("button", {
            type: "button", class: "stepbtn", title: "Increase quantity",
            "aria-label": "Increase quantity of " + p.name,
            "data-testid": "qty-plus-" + p.id, "data-fkey": "qty-plus-" + p.id,
            onclick: function () { act("cart.increment", { productId: p.id }); }
          }, "+")),
        h("span", { class: "spacer" }),
        h("span", { class: "cartline-total", "data-testid": "linetotal-" + p.id,
          text: global.money(p.price * line.qty) }),
        global.button({
          label: "Remove", size: "sm", testid: "remove-" + p.id,
          onclick: function () { act("cart.remove", { productId: p.id }); }
        })));
  }

  function cartView(s) {
    var lines = s.cart.lines;
    return frag(
      h("div", { class: "drawer-body", "data-testid": "cart-lines" },
        lines.length
          ? lines.map(function (line) { return lineRow(s, line); })
          : global.emptyState("cart", "Your cart is empty", "Add products from the grid.")),
      h("div", { class: "drawer-foot" },
        h("div", { class: "totalrow" },
          h("span", { text: "Subtotal" }),
          h("span", { class: "amount", "data-testid": "cart-subtotal", text: global.money(subtotal(s)) })),
        global.button({
          label: "Checkout", variant: "primary", size: "lg",
          testid: "checkout-open", disabled: lines.length === 0,
          onclick: function () { act("checkout.open"); }
        })));
  }

  function drawer(s) {
    if (!s.ui.cartOpen) return null;
    var checkingOut = s.ui.checkoutOpen;
    return h("aside", { class: "drawer", role: "dialog", "aria-label": checkingOut ? "Checkout" : "Your cart",
      "data-testid": "cart-drawer" },
      h("div", { class: "drawer-head" },
        h("h2", { text: checkingOut ? "Checkout" : "Your cart" }),
        h("span", { class: "spacer" }),
        global.button({ label: "Close", iconName: "close", size: "sm", testid: "cart-close",
          onclick: function () { act("cart.close"); } })),
      checkingOut ? global.checkoutPanel(s) : cartView(s));
  }

  global.cartSubtotal = subtotal;
  global.App.screens.drawer = drawer;
})(window);
