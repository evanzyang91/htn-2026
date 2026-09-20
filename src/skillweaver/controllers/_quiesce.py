"""The page-at-rest script both browser controllers run."""

QUIESCE_JS = """
([quietMs, capMs]) => new Promise((resolve) => {
  const start = performance.now();
  let last = start;
  const observer = new MutationObserver(() => { last = performance.now(); });
  observer.observe(document.documentElement,
    { subtree: true, childList: true, characterData: true });
  const lastResource = () => {
    const entries = performance.getEntriesByType('resource');
    return entries.length ? entries[entries.length - 1].responseEnd : 0;
  };
  const tick = () => {
    const now = performance.now();
    if (now - Math.max(last, lastResource()) >= quietMs || now - start >= capMs) {
      observer.disconnect();
      resolve(Math.round(now - start));
    } else {
      setTimeout(tick, 40);
    }
  };
  setTimeout(tick, 40);
})
"""
"""Upstream Jev's ``SETTLE``: resolves once neither the DOM nor the network has moved for
``quietMs``, or at ``capMs``. No attribute observation, so an animation alone does not
hold it open. See :meth:`BrowserController.quiesce`;
both controllers run this one script."""
