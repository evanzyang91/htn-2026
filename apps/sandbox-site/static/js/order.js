/* Pantry Lane: the food-ordering surface.

   Its shape is the point, not its looks. A catalogue you search and filter, a
   detail page you have to navigate into, dishes whose required option groups
   stop an agent from adding anything by clicking blindly, a cart that
   accumulates, and a checkout that COMMITS - moving the cart into an order
   history. That last step is a multi-step state change, which is exactly what
   the rest of the app has none of, and `GET /__reset` is what makes it
   learnable anyway.

   Money is integer cents everywhere. A float subtotal would round differently
   here than in serve.py, and the two have to agree for a screenshot and the
   placed order to tell the same story.                                     */
(function (global) {
  "use strict";

  var VIEWS = [
    { id: "browse", label: "Browse", iconName: "bag" },
    { id: "cart", label: "Cart", iconName: "cart" },
    { id: "orders", label: "Orders", iconName: "receipt" }
  ];

  function money(cents) {
    return "$" + (cents / 100).toFixed(2);
  }

  function restaurantById(s, id) {
    var hit = s.order.restaurants.filter(function (r) { return r.id === id; });
    return hit.length ? hit[0] : null;
  }

  function dishById(r, id) {
    if (!r) return null;
    var hit = r.dishes.filter(function (d) { return d.id === id; });
    return hit.length ? hit[0] : null;
  }

  function visibleRestaurants(s) {
    var ui = s.ui.order;
    var q = ui.search.trim().toLowerCase();
    return s.order.restaurants.filter(function (r) {
      if (ui.cuisine && r.cuisine !== ui.cuisine) return false;
      if (!q) return true;
      var hay = r.name + " " + r.cuisine + " " + r.blurb + " "
        + r.dishes.map(function (d) { return d.name; }).join(" ");
      return hay.toLowerCase().indexOf(q) !== -1;
    });
  }

  function linePrice(dish, picks) {
    var total = dish.priceCents;
    dish.options.forEach(function (g) {
      g.choices.forEach(function (c) { if (picks[g.id] === c.id) total += c.deltaCents; });
    });
    return total;
  }

  function complete(dish, picks) {
    return dish.options.every(function (g) {
      return g.choices.some(function (c) { return c.id === picks[g.id]; });
    });
  }

  function cartCount(s) {
    return s.order.cart.reduce(function (n, line) { return n + line.qty; }, 0);
  }

  function subtotal(s) {
    return s.order.cart.reduce(function (n, line) {
      return n + line.unitPriceCents * line.qty;
    }, 0);
  }

  /* ------------------------------------------------------------- chrome */

  function header(s) {
    var count = cartCount(s);
    return h("div", { class: "ord-head" },
      global.icon("bag", "icon-lg"),
      h("div", null,
        h("h1", { text: "Pantry Lane" }),
        h("div", { class: "sub", "data-testid": "ord-subtitle",
          text: s.order.restaurants.length + " restaurants delivering to "
            + s.ui.order.address })),
      h("span", { class: "spacer" }),
      /* A real control, not a badge. A read-only pill here reads as the cart
         button every ordering site has, and an agent that clicks it and gets
         nothing spends its budget deciding whether it mis-clicked. */
      h("button", {
        type: "button", class: "carttotal",
        "aria-label": "View cart",
        "data-testid": "cart-total", "data-fkey": "cart-total",
        onclick: function () { act("order.view", { view: "cart" }); }
      }, global.icon("cart", "icon-sm"),
        h("span", { text: count ? money(subtotal(s)) + " cart" : "Cart empty" })));
  }

  function tabs(s) {
    var ui = s.ui.order;
    var current = ui.view === "restaurant" ? "browse" : ui.view;
    return h("div", { class: "ord-tabs", role: "tablist" },
      VIEWS.map(function (v) {
        var badge = v.id === "cart" ? cartCount(s)
          : (v.id === "orders" ? s.order.orders.length : 0);
        return h("button", {
          type: "button", class: "ord-tab", role: "tab",
          "aria-selected": current === v.id ? "true" : "false",
          "data-testid": "ord-tab-" + v.id, "data-fkey": "ord-tab-" + v.id,
          onclick: function () { act("order.view", { view: v.id }); }
        }, global.icon(v.iconName, "icon-sm"), h("span", { text: v.label }),
          badge ? h("span", { class: "count", "data-testid": "ord-badge-" + v.id,
            text: String(badge) }) : null);
      }));
  }

  /* ------------------------------------------------------------- browse */

  function filters(s) {
    var ui = s.ui.order;
    return h("div", { class: "ord-filters" },
      h("div", { class: "searchbox" },
        global.icon("search", "icon-sm"),
        h("input", {
          class: "field", type: "search", name: "ord-search",
          placeholder: "Search restaurants or dishes", value: ui.search,
          "aria-label": "Search restaurants", "data-testid": "ord-search", "data-fkey": "ord-search",
          style: "width:280px",
          oninput: function (ev) { act("order.search", { q: ev.target.value }); }
        })),
      h("span", { class: "sep" }),
      h("div", { class: "chips" },
        h("button", {
          type: "button", class: "chip", "aria-pressed": ui.cuisine ? "false" : "true",
          "data-testid": "cuisine-all", "data-fkey": "cuisine-all",
          onclick: function () { act("order.cuisine", { cuisine: "" }); }
        }, h("span", { text: "All" })),
        s.order.cuisines.map(function (c) {
          return h("button", {
            type: "button", class: "chip", "aria-pressed": ui.cuisine === c ? "true" : "false",
            "data-testid": "cuisine-" + c, "data-fkey": "cuisine-" + c,
            onclick: function () { act("order.cuisine", { cuisine: c }); }
          }, h("span", { text: c }));
        })));
  }

  function stars(rating) {
    return h("span", { class: "rating", "aria-label": rating + " out of 5" },
      global.icon("star", "icon-sm"), h("span", { text: rating.toFixed(1) }));
  }

  function restaurantCard(r) {
    return h("button", {
      type: "button", class: "restcard",
      "data-testid": "restaurant-" + r.id, "data-fkey": "restaurant-" + r.id,
      onclick: function () { act("order.open", { id: r.id }); }
    },
      h("div", { class: "restcard-top" },
        h("span", { class: "resticon" }, global.icon("bag")),
        h("div", { class: "restcard-name" },
          h("strong", { text: r.name }),
          h("span", { class: "meta", text: r.cuisine + " · " + r.priceBand })),
        h("span", { class: "spacer" }),
        stars(r.rating)),
      h("p", { class: "restcard-blurb", text: r.blurb }),
      h("div", { class: "restcard-foot" },
        h("span", { text: r.eta }),
        h("span", { class: "spacer" }),
        h("span", { class: "muted", text: r.dishes.length + " dishes" })));
  }

  function browse(s) {
    var list = visibleRestaurants(s);
    return h("div", null,
      filters(s),
      h("div", { class: "ord-count", "data-testid": "ord-result-count",
        text: "Showing " + list.length + " of " + s.order.restaurants.length + " restaurants" }),
      list.length
        ? h("div", { class: "restgrid", "data-testid": "restaurant-grid" },
            list.map(restaurantCard))
        : global.emptyState("search", "No restaurants match",
            "Nothing matches the current search and cuisine filter."));
  }

  /* --------------------------------------------------------- restaurant */

  function dishRow(s, r, d) {
    var ui = s.ui.order;
    return h("button", {
      type: "button", class: "dishrow" + (ui.dishId === d.id ? " open" : ""),
      "data-testid": "dish-" + d.id, "data-fkey": "dish-" + d.id,
      onclick: function () { act("order.dish", { id: d.id }); }
    },
      h("div", { class: "dishrow-main" },
        h("strong", { text: d.name }),
        h("span", { class: "desc", text: d.desc })),
      h("div", { class: "dishrow-side" },
        h("span", { class: "price", text: money(d.priceCents) }),
        d.options.length
          ? h("span", { class: "tagline", "data-testid": "dish-choices-" + d.id,
              text: d.options.length + " choice" + (d.options.length === 1 ? "" : "s") + " required" })
          : h("span", { class: "tagline muted", text: "No choices" })));
  }

  function optionGroup(s, d, g) {
    var picks = s.ui.order.picks;
    return h("div", { class: "optgroup", "data-testid": "optgroup-" + g.id },
      h("div", { class: "optlabel" },
        h("span", { text: g.label }),
        h("span", { class: "req" + (picks[g.id] ? " done" : ""),
          "data-testid": "optreq-" + g.id,
          text: picks[g.id] ? "chosen" : "required" })),
      h("div", { class: "optrow" }, g.choices.map(function (c) {
        return h("button", {
          type: "button", class: "opt",
          "aria-pressed": picks[g.id] === c.id ? "true" : "false",
          "data-testid": "choice-" + g.id + "-" + c.id,
          "data-fkey": "choice-" + g.id + "-" + c.id,
          onclick: function () { act("order.choose", { group: g.id, choice: c.id }); }
        }, h("span", { text: c.label }),
          c.deltaCents ? h("span", { class: "delta", text: "+" + money(c.deltaCents) }) : null);
      })));
  }

  function stepper(opts) {
    return h("div", { class: "stepper" },
      global.iconButton({ iconName: "minus", label: opts.decLabel, iconClass: "icon-sm",
        testid: opts.decTestid, disabled: opts.value <= 1, onclick: opts.onDec }),
      h("span", { class: "stepval", "data-testid": opts.valueTestid, text: String(opts.value) }),
      global.iconButton({ iconName: "plus", label: opts.incLabel, iconClass: "icon-sm",
        testid: opts.incTestid, disabled: opts.value >= 9, onclick: opts.onInc }));
  }

  function dishPanel(s, r) {
    var ui = s.ui.order;
    var d = dishById(r, ui.dishId);
    if (!d) {
      return h("aside", { class: "dishpanel empty", "data-testid": "dish-panel-empty" },
        global.emptyState("bag", "Pick a dish",
          "Choose something from the menu to set its options."));
    }
    var ready = complete(d, ui.picks);
    var unit = linePrice(d, ui.picks);
    return h("aside", { class: "dishpanel", "data-testid": "dish-panel" },
      h("div", { class: "dishpanel-head" },
        h("h2", { "data-testid": "dish-panel-name", text: d.name }),
        h("span", { class: "spacer" }),
        global.iconButton({ iconName: "close", label: "Close dish", iconClass: "icon-sm",
          testid: "dish-close", onclick: function () { act("order.closeDish"); } })),
      h("p", { class: "dishpanel-desc", text: d.desc }),
      h("div", { class: "dishpanel-opts" },
        d.options.length
          ? d.options.map(function (g) { return optionGroup(s, d, g); })
          : h("div", { class: "hint", text: "This dish has no options." })),
      h("div", { class: "dishpanel-foot" },
        stepper({
          value: ui.qty,
          decLabel: "Decrease quantity", incLabel: "Increase quantity",
          decTestid: "qty-dec", incTestid: "qty-inc", valueTestid: "qty-value",
          onDec: function () { act("order.qty", { delta: -1 }); },
          onInc: function () { act("order.qty", { delta: 1 }); }
        }),
        h("span", { class: "spacer" }),
        h("span", { class: "linetotal", "data-testid": "dish-line-total",
          text: money(unit * ui.qty) })),
      global.button({
        label: ready ? "Add to cart" : "Choose every option first",
        iconName: "plus", variant: "primary", testid: "add-to-cart",
        disabled: !ready, onclick: function () { act("order.add"); }
      }));
  }

  function restaurantView(s) {
    var r = restaurantById(s, s.ui.order.restaurantId);
    if (!r) return browse(s);
    return h("div", { class: "restview", "data-testid": "restaurant-view" },
      h("div", { class: "restview-head" },
        global.button({ label: "All restaurants", iconName: "back", testid: "ord-back",
          onclick: function () { act("order.view", { view: "browse" }); } }),
        h("span", { class: "sep" }),
        h("div", null,
          h("h2", { "data-testid": "rest-name", text: r.name }),
          h("span", { class: "meta",
            text: r.cuisine + " · " + r.priceBand + " · " + r.eta })),
        h("span", { class: "spacer" }),
        stars(r.rating)),
      h("div", { class: "restview-body" },
        h("div", { class: "dishlist", "data-testid": "dish-list" },
          r.dishes.map(function (d) { return dishRow(s, r, d); })),
        dishPanel(s, r)));
  }

  /* ----------------------------------------------------------- the cart */

  function cartLine(s, line) {
    return h("div", { class: "cartline", "data-testid": "cartline-" + line.lineId },
      h("div", { class: "cartline-main" },
        h("strong", { text: line.name }),
        h("span", { class: "meta", text: line.restaurant
          + (line.choiceText ? " · " + line.choiceText : "") }),
        h("span", { class: "muted", text: money(line.unitPriceCents) + " each" })),
      stepper({
        value: line.qty,
        decLabel: "Decrease " + line.name, incLabel: "Increase " + line.name,
        decTestid: "cartdec-" + line.lineId, incTestid: "cartinc-" + line.lineId,
        valueTestid: "cartqty-" + line.lineId,
        onDec: function () { act("order.cartQty", { lineId: line.lineId, delta: -1 }); },
        onInc: function () { act("order.cartQty", { lineId: line.lineId, delta: 1 }); }
      }),
      h("span", { class: "linetotal", "data-testid": "cartsum-" + line.lineId,
        text: money(line.unitPriceCents * line.qty) }),
      global.iconButton({ iconName: "trash", label: "Remove " + line.name, iconClass: "icon-sm",
        testid: "cartremove-" + line.lineId,
        onclick: function () { act("order.remove", { lineId: line.lineId }); } }));
  }

  function totalsRows(s) {
    var sub = subtotal(s);
    var delivery = s.order.deliveryCents;
    var tip = s.ui.order.tipCents;
    return { sub: sub, delivery: delivery, tip: tip, total: sub + delivery + tip };
  }

  function cartView(s) {
    var ui = s.ui.order;
    var t = totalsRows(s);
    if (!s.order.cart.length) {
      return h("div", { class: "cartview", "data-testid": "cart-view" },
        global.emptyState("cart", "Your cart is empty",
          "Add something from a restaurant menu to start an order."));
    }
    return h("div", { class: "cartview", "data-testid": "cart-view" },
      h("div", { class: "cartlines" }, s.order.cart.map(function (l) { return cartLine(s, l); })),
      h("section", { class: "card checkout", "data-testid": "checkout-card" },
        h("div", { class: "card-head" },
          h("h2", { text: "Checkout" }),
          h("p", { text: "Confirm where it goes and what to add, then place the order." })),
        h("div", { class: "card-body" },
          h("div", { class: "setrow" },
            h("div", null,
              h("div", { class: "label", text: "Delivery address" }),
              h("div", { class: "hint", text: "Where the courier is sent." })),
            h("input", {
              class: "field", type: "text", name: "ord-address", value: ui.address,
              "aria-label": "Delivery address",
              "data-testid": "ord-address", "data-fkey": "ord-address",
              style: "width:280px",
              oninput: function (ev) { act("order.address", { value: ev.target.value }); }
            })),
          h("div", { class: "setrow" },
            h("div", null,
              h("div", { class: "label", text: "Tip" }),
              h("div", { class: "hint", text: "A flat amount, added to the total." })),
            h("div", { class: "chips" }, s.order.tipOptions.map(function (cents) {
              return h("button", {
                type: "button", class: "chip",
                "aria-pressed": ui.tipCents === cents ? "true" : "false",
                "data-testid": "tip-" + cents, "data-fkey": "tip-" + cents,
                onclick: function () { act("order.tip", { cents: cents }); }
              }, h("span", { text: cents ? money(cents) : "No tip" }));
            }))),
          h("dl", { class: "totals", "data-testid": "cart-totals" },
            h("div", null, h("dt", { text: "Subtotal" }),
              h("dd", { "data-testid": "total-subtotal", text: money(t.sub) })),
            h("div", null, h("dt", { text: "Delivery" }),
              h("dd", { "data-testid": "total-delivery", text: money(t.delivery) })),
            h("div", null, h("dt", { text: "Tip" }),
              h("dd", { "data-testid": "total-tip", text: money(t.tip) })),
            h("div", { class: "grand" }, h("dt", { text: "Total" }),
              h("dd", { "data-testid": "total-grand", text: money(t.total) })))),
        h("div", { class: "card-foot" },
          h("span", { class: "muted", "data-testid": "cart-itemcount",
            text: cartCount(s) + " item" + (cartCount(s) === 1 ? "" : "s") }),
          h("span", { class: "spacer" }),
          global.button({
            label: "Place order", iconName: "check", variant: "primary",
            testid: "checkout-btn", disabled: !ui.address.trim(),
            onclick: function () { act("order.checkout"); }
          }))));
  }

  /* -------------------------------------------------------- the history */

  function orderCard(s, o) {
    var isLatest = s.ui.order.placedId === o.id;
    return h("section", { class: "card ordercard" + (isLatest ? " latest" : ""),
      "data-testid": "order-" + o.id },
      h("div", { class: "card-head" },
        h("h2", { text: "Order " + o.id }),
        h("p", { text: o.restaurants.join(", ") + " · " + o.placedOn })),
      h("div", { class: "card-body" },
        h("ul", { class: "orderitems", "data-testid": "orderitems-" + o.id },
          o.items.map(function (line) {
            return h("li", null,
              h("span", { class: "k", text: line.qty + " x " + line.name }),
              h("span", { text: line.choiceText ? " (" + line.choiceText + ")" : "" }),
              h("span", { class: "spacer" }),
              h("span", { text: money(line.unitPriceCents * line.qty) }));
          }))),
      h("div", { class: "card-foot" },
        h("span", { class: "pill pill-active", "data-testid": "orderstatus-" + o.id,
          text: o.status }),
        h("span", { class: "muted", text: o.itemCount + " items to " + o.address }),
        h("span", { class: "spacer" }),
        h("strong", { "data-testid": "ordertotal-" + o.id, text: money(o.totalCents) })));
  }

  function ordersView(s) {
    if (!s.order.orders.length) {
      return h("div", { class: "ordersview", "data-testid": "orders-view" },
        global.emptyState("receipt", "No orders yet",
          "A placed order shows up here with everything that was in the cart."));
    }
    return h("div", { class: "ordersview", "data-testid": "orders-view" },
      s.order.orders.map(function (o) { return orderCard(s, o); }));
  }

  /* -------------------------------------------------------------- shell */

  var BODIES = { browse: browse, restaurant: restaurantView, cart: cartView, orders: ordersView };

  function screen(s) {
    var ui = s.ui.order;
    return h("div", { class: "order", "data-testid": "screen-order" },
      header(s),
      tabs(s),
      h("div", { class: "ord-bannerbar" },
        global.banner(ui.banner, "ord-banner", function () { act("order.dismissBanner"); })),
      (BODIES[ui.view] || browse)(s));
  }

  /* The overlay has one dialog builder for the whole app, so this chains onto
     whatever was registered before it rather than replacing it. */
  var previousDialog = global.App.screens.dialog;

  function dialog(s) {
    var ui = s.ui.order;
    if (s.ui.screen !== "order" || !ui.dialogOpen) {
      return previousDialog ? previousDialog(s) : null;
    }
    var t = totalsRows(s);
    return h("div", { class: "scrim", "data-testid": "scrim" },
      h("div", { class: "dialog", role: "dialog", "aria-modal": "true",
        "aria-label": "Place this order", "data-testid": "order-dialog" },
        h("div", { class: "dialog-head" },
          global.icon("cart", "icon-lg"),
          h("h2", { text: "Place this order?" })),
        h("div", { class: "dialog-body" },
          h("div", { text: "This confirms the order and moves it to your order history." }),
          h("ul", { "data-testid": "order-confirm-list" },
            s.order.cart.map(function (line) {
              return h("li", null,
                h("span", { class: "k", text: line.qty + " x " + line.name }),
                h("span", { text: " from " + line.restaurant }));
            })),
          h("div", { class: "dialog-total", "data-testid": "dialog-total",
            text: "Total " + money(t.total) + " to " + ui.address })),
        h("div", { class: "dialog-foot" },
          h("span", { class: "spacer" }),
          global.button({ label: "Cancel", testid: "order-cancel",
            onclick: function () { act("order.cancelCheckout"); } }),
          global.button({ label: "Confirm order", iconName: "check", variant: "primary",
            testid: "order-confirm", onclick: function () { act("order.place"); } }))));
  }

  global.App.screens.order = screen;
  global.App.screens.dialog = dialog;
})(window);
