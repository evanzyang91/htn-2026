/* Inline SVG icon helper. Every symbol lives in the sprite in index.html. */
(function (global) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function icon(name, cls) {
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "icon" + (cls ? " " + cls : ""));
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    var use = document.createElementNS(NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  global.icon = icon;
})(window);
