/* Shop surface: category sidebar, price-band radios, search, sort, product grid.
   Deliberately a card-grid idiom - not a table, not a list - so it exercises
   different perception and interaction patterns than a console app.          */
(function (global) {
  "use strict";

  var CATEGORIES = ["All", "Tools", "Outdoor", "Kitchen", "Lighting"];

  var PRICE_BANDS = [
    { id: "any", label: "Any price" },
    { id: "under25", label: "Under $25" },
    { id: "25to75", label: "$25 to $75" },
    { id: "over75", label: "Over $75" }
  ];

  var SORTS = [
    { id: "featured", label: "Featured" },
    { id: "price-asc", label: "Price: low to high" },
    { id: "price-desc", label: "Price: high to low" },
    { id: "rating-desc", label: "Top rated" },
    { id: "name-asc", label: "Name A to Z" }
  ];

  var TILE_CLASS = { Tools: "tile-tools", Outdoor: "tile-outdoor", Kitchen: "tile-kitchen", Lighting: "tile-lighting" };

  function inBand(p, band) {
    if (band === "under25") return p.price < 2500;
    if (band === "25to75") return p.price >= 2500 && p.price <= 7500;
    if (band === "over75") return p.price > 7500;
    return true;
  }

  function visibleProducts(s) {
    var ui = s.ui;
    var q = ui.search.trim().toLowerCase();
    var list = s.catalog.products.filter(function (p) {
      if (ui.category !== "All" && p.category !== ui.category) return false;
      if (!inBand(p, ui.priceBand)) return false;
      if (q && (p.name + " " + p.category).toLowerCase().indexOf(q) === -1) return false;
      return true;
    });
    var sort = ui.sort;
    list.sort(function (a, b) {
      var c = 0;
      if (sort === "price-asc") c = a.price - b.price;
      else if (sort === "price-desc") c = b.price - a.price;
      else if (sort === "rating-desc") c = b.rating - a.rating;
      else if (sort === "name-asc") c = a.name.localeCompare(b.name);
      if (c !== 0) return c;
      return a.id.localeCompare(b.id); /* stable tie-break keeps order deterministic */
    });
    return list;
  }

  function initials(name) {
    var parts = name.split(/\s+/).filter(Boolean);
    return ((parts[0] || "")[0] + ((parts[1] || "")[0] || "")).toUpperCase();
  }

  /* ------------------------------------------------------------- sidebar */

  function sidebar(s) {
    var ui = s.ui;

    function catButton(cat) {
      var count = cat === "All"
        ? s.catalog.products.length
        : s.catalog.products.filter(function (p) { return p.category === cat; }).length;
      return h("button", {
        type: "button", class: "catbtn",
        "aria-current": ui.category === cat ? "true" : "false",
        "data-testid": "cat-" + cat, "data-fkey": "cat-" + cat,
        onclick: function () { act("shop.category", { category: cat }); }
      }, h("span", { text: cat }), h("span", { class: "count", text: String(count) }));
    }

    return h("aside", { class: "shop-side", "data-testid": "shop-side" },
      h("div", { class: "side-group" },
        h("div", { class: "side-title", text: "Categories" }),
        CATEGORIES.map(catButton)),
      h("div", { class: "side-group" },
        h("div", { class: "side-title", text: "Price" }),
        PRICE_BANDS.map(function (band) {
          return global.radio({
            group: "priceBand", value: band.id, label: band.label,
            checked: ui.priceBand === band.id,
            testid: "price-" + band.id,
            onchange: function () { act("shop.priceBand", { band: band.id }); }
          });
        })));
  }

  /* ------------------------------------------------------------- toolbar */

  function toolbar(s, list) {
    var ui = s.ui;
    return h("div", { class: "shop-toolbar" },
      h("span", { class: "result-count", "data-testid": "result-count",
        text: list.length + " of " + s.catalog.products.length + " products" }),
      h("span", { class: "spacer" }),
      h("label", { class: "sortwrap" },
        h("span", { class: "sort-label", text: "Sort by" }),
        h("select", {
          class: "field", name: "sort-select", "aria-label": "Sort by",
          "data-testid": "sort-select", "data-fkey": "sort-select",
          onchange: function (ev) { act("shop.sort", { sort: ev.target.value }); }
        }, SORTS.map(function (opt) {
          return h("option", { value: opt.id, selected: ui.sort === opt.id ? true : null, text: opt.label });
        }))));
  }

  /* ------------------------------------------------------------- grid */

  function card(s, p) {
    var qtyInCart = 0;
    s.cart.lines.forEach(function (line) { if (line.productId === p.id) qtyInCart = line.qty; });
    return h("article", { class: "card", "data-testid": "card-" + p.id },
      h("div", { class: "tile " + TILE_CLASS[p.category] },
        h("span", { class: "tile-mark", text: initials(p.name) })),
      h("div", { class: "card-body" },
        h("div", { class: "card-name", "data-testid": "name-" + p.id, text: p.name }),
        h("div", { class: "card-meta" },
          h("span", { class: "card-cat", text: p.category }),
          h("span", { class: "card-rating", "data-testid": "rating-" + p.id,
            text: "★ " + p.rating.toFixed(1) })),
        h("div", { class: "card-foot" },
          h("span", { class: "card-price", "data-testid": "price-" + p.id, text: global.money(p.price) }),
          global.button({
            label: qtyInCart ? "Add to cart (" + qtyInCart + ")" : "Add to cart",
            variant: "primary", size: "sm", testid: "add-" + p.id,
            onclick: function () { act("cart.add", { productId: p.id }); }
          }))));
  }

  /* ------------------------------------------------------------- screen */

  function screen(s) {
    var list = visibleProducts(s);
    return h("div", { class: "shop", "data-testid": "screen-shop" },
      sidebar(s),
      h("div", { class: "shop-main" },
        toolbar(s, list),
        h("div", { class: "shop-bannerbar" },
          global.banner(s.ui.banner, "shop-banner", function () { act("shop.dismissBanner"); })),
        h("div", { class: "gridwrap", "data-testid": "gridwrap" },
          list.length
            ? h("div", { class: "grid" }, list.map(function (p) { return card(s, p); }))
            : global.emptyState("search", "No products match",
                s.ui.search ? 'Nothing matches "' + s.ui.search + '".' : "No products in this range."))));
  }

  global.App.screens.shop = screen;
})(window);
