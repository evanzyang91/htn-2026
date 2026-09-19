/* skillweaver dashboard - the only script on the page, inlined at build time.
   It does exactly one thing: light up the cheapest known route from a site's entry
   screen to whichever screen you click. Every route was computed at build time by
   the graph module's own cost model and embedded as JSON, so nothing here decides
   anything about routing - it only shows what the agent already knows. */
(function () {
  "use strict";

  var holder = document.getElementById("route-data");
  if (!holder) return;

  var routes;
  try {
    routes = JSON.parse(holder.textContent || "{}");
  } catch (err) {
    return;
  }

  Array.prototype.forEach.call(document.querySelectorAll("[data-graph]"), function (svg) {
    var prefix = svg.getAttribute("data-graph");
    var table = routes[prefix] || {};
    var bar = document.getElementById(prefix + "-routebar");
    var out = bar ? bar.querySelector(".out") : null;
    var reset = bar ? bar.querySelector("button") : null;
    var idle = bar ? bar.querySelector(".idle") : null;
    var active = null;

    function clear() {
      active = null;
      svg.classList.remove("routed");
      Array.prototype.forEach.call(svg.querySelectorAll(".on"), function (el) {
        el.classList.remove("on");
      });
      if (out) out.textContent = "";
      if (reset) reset.hidden = true;
      if (idle) idle.hidden = false;
    }

    function show(key, label) {
      var route = table[key];
      clear();
      if (idle) idle.hidden = true;
      if (!route) {
        if (out) {
          out.textContent = "no route to " + label + " has been proven yet";
        }
        if (reset) reset.hidden = false;
        return;
      }
      active = key;
      svg.classList.add("routed");
      (route.edges || []).forEach(function (edgeKey) {
        var edge = svg.querySelector('[data-edge="' + edgeKey + '"]');
        if (edge) edge.classList.add("on");
      });
      (route.nodes || []).forEach(function (nodeKey) {
        var node = svg.querySelector('[data-node="' + nodeKey + '"]');
        if (node) node.classList.add("on");
      });
      if (out) {
        out.textContent =
          "route to " + label + ": " + route.hops + " hop" + (route.hops === 1 ? "" : "s") +
          ", " + route.steps + " action" + (route.steps === 1 ? "" : "s") +
          ", " + Math.round(route.cost) + "ms expected";
      }
      if (reset) reset.hidden = false;
    }

    Array.prototype.forEach.call(svg.querySelectorAll("[data-node]"), function (node) {
      node.addEventListener("click", function () {
        var key = node.getAttribute("data-node");
        if (key === active) {
          clear();
        } else {
          show(key, node.getAttribute("data-label") || key);
        }
      });
    });

    if (reset) {
      reset.hidden = true;
      reset.addEventListener("click", clear);
    }
  });
})();
