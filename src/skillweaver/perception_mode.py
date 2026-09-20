"""Which eyes produced a screen - pixels or DOM - and the namespace that keeps the
two skill libraries apart. See ``AGENTS.md`` for the pixels-only rule this relaxes."""

from __future__ import annotations

__all__ = ["DOM", "PATHS", "PIXELS", "bare", "namespace", "path_of"]

PIXELS = "pixels"
"""YOLO plus OCR over a screenshot; the default, and what every pre-existing skill
was learned against."""

DOM = "dom"
"""The page's own control list via ``DomPerceiver``. Browser targets only."""

PATHS = (PIXELS, DOM)

_PREFIX = f"{DOM}@"
"""``@`` and not ``/``: ``FileSkillStore.skill_dir`` joins the domain unsanitized, so a
path separator here would let a namespace escape the store root."""


def namespace(domain: str, path: str) -> str:
    """The library namespace ``domain`` belongs to under ``path``; idempotent, and a
    no-op for ``PIXELS`` so pre-existing skills keep their names."""
    if path != DOM or domain.startswith(_PREFIX):
        return domain
    return f"{_PREFIX}{domain}"


def path_of(domain: str) -> str:
    """Which perception path ``domain`` is a namespace of."""
    return DOM if domain.startswith(_PREFIX) else PIXELS


def bare(domain: str) -> str:
    """``domain`` with any path prefix removed - the host, for anything that means the
    site rather than the library namespace."""
    return domain[len(_PREFIX) :] if domain.startswith(_PREFIX) else domain
