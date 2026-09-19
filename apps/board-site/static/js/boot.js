/* Boot: pull the server state once and render. No timers, no polling. */
(function () {
  "use strict";
  window.addEventListener("DOMContentLoaded", function () {
    window.load().then(function () {
      document.body.dataset.ready = "1";
    });
  });
}());
