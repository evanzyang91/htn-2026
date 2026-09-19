"""Which eyes produced a screen - pixels or DOM - and why the two keep separate libraries.

This project reads a screen two ways now. The original one is PIXELS: a YOLO detector
and an OCR reader over a screenshot, which is what
:class:`~skillweaver.orchestrator.ComposedPerceiver` does and what every stored skill
before this module was learned against. The second is DOM
(:class:`~skillweaver.perception.dom.DomPerceiver`), which asks the page what its
controls are and skips detection and OCR entirely. See ``AGENTS.md`` for the standing
rule this relaxes and for exactly how far it is relaxed.

Why the two cannot share a library
----------------------------------

A skill is retrieved by ranking, admitted by its precondition SCREEN and then run
against element text. The first two go through
:class:`~skillweaver.contracts.Fingerprint`, and the third through
:class:`~skillweaver.contracts.ElementIndex` - and it is the THIRD that makes the two
paths incompatible, not the first:

* The DOM says ``add to cart``, because that is the button's accessible name. OCR says
  ``ADD TO CART`` or ``AOD TO CART`` or nothing at all when the label is an icon. A
  stored ``ctx.see.find_text("Add to cart")`` written against one reader is a coin flip
  against the other.
* The DOM sees a control whose label lives only in ``aria-label``; no camera can read
  it. Equally, OCR sees text baked into an image that the DOM has no node for.

And the fingerprint does NOT protect against that, which is the part worth being
careful about. :class:`~skillweaver.perception.fingerprint.StateFingerprinter` derives
its ``parts`` - the only thing ``similarity`` reads - from the SCREENSHOT and the URL.
Both paths capture the same screenshot of the same page, so a DOM-perceived screen and
a pixel-perceived one score close to **1.000** against each other: the precondition
gate waves the wrong-path skill straight through. Only ``value`` differs, because
``value`` also hashes the element layout - so the site GRAPH keeps its nodes apart and
the skill LIBRARY does not.

So the separation is made where it actually holds: in the namespace a skill is filed
under and looked up in. :func:`namespace` prefixes the domain for the DOM path, which
is one decision that separates the skill store (a directory per domain), the site graph
(a file per domain) and cross-domain resolution
(:func:`~skillweaver.orchestrator.resolve_domain`) all at once. A pixel run and a DOM
run on one site keep two libraries, two graphs, and never read each other's.

This is deliberately NOT the shape :mod:`skillweaver.render_mode` uses. A headed/headless
crossing is recorded beside the skill and merely EXPLAINED when a run loses the screen,
because the crossing is survivable on some pages - measured. A path crossing is not
survivable on any page and has no measurement to appeal to, so it is made impossible
rather than reported.
"""

from __future__ import annotations

__all__ = ["DOM", "PATHS", "PIXELS", "bare", "namespace", "path_of"]

PIXELS = "pixels"
"""YOLO plus OCR over a screenshot. The DEFAULT, and what every skill stored before
this module was learned against."""

DOM = "dom"
"""The page's own control list, read through
:class:`~skillweaver.perception.dom.DomPerceiver`. Browser targets only: a desktop
controller has no DOM to ask."""

PATHS = (PIXELS, DOM)
"""Every perception path, for validating a flag or a setting."""

_PREFIX = f"{DOM}@"
"""What a DOM-path domain is spelled with.

``@`` rather than ``/`` or ``:`` on purpose:
:meth:`~skillweaver.skills.store.FileSkillStore.skill_dir` builds
``<root>/<domain>/<name>`` without sanitizing, so a separator that is a path
separator would let a namespace escape the store's root.
"""


def namespace(domain: str, path: str) -> str:
    """The library namespace ``domain`` belongs to when perceived through ``path``.

    Unchanged for :data:`PIXELS`, so every skill stored before this module keeps its
    name and every pixel run keeps finding it. Prefixed for :data:`DOM`. Applying it
    twice is a no-op, so a caller that cannot easily tell whether a domain has already
    been namespaced may simply call it again.
    """
    if path != DOM or domain.startswith(_PREFIX):
        return domain
    return f"{_PREFIX}{domain}"


def path_of(domain: str) -> str:
    """Which perception path ``domain`` is a namespace of. See :func:`namespace`."""
    return DOM if domain.startswith(_PREFIX) else PIXELS


def bare(domain: str) -> str:
    """``domain`` with any path prefix removed - the host on its own.

    For anything that means the SITE rather than the library namespace: a log line a
    person reads, or the host a URL is checked against.
    """
    return domain[len(_PREFIX) :] if domain.startswith(_PREFIX) else domain
