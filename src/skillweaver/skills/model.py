"""Construction, validation and serialization for :class:`~skillweaver.contracts.Skill`.

A skill is code that a model wrote and that a later run will execute, so the only
thing standing between a bad generation and a broken library is this module. Every
check here is structural - "could this ever work?" - and cheap enough to run on every
synthesis. Whether the skill *does* work is the admission harness's job.

::

    skill = make_skill(
        name="search_invoice",
        domain="example.com",
        summary="Search the invoice list for a company.",
        docstring="Types the company name into the search box and waits for rows.",
        params={"company": {"type": "string"}},
        code="def run(ctx, company):\\n    ctx.ctl.type_text(company)\\n",
        provenance=Provenance("run-1", "pay an invoice", "claude", utcnow()),
    )
    signature(skill)                      # 'search_invoice(company: str)'
    from_dict(to_dict(skill)) == skill    # True

Every failure raises a :class:`SkillInvalid` subclass naming exactly what was wrong.
They all derive from :class:`~skillweaver.errors.AdmissionRejected`, so a synthesizer
that already catches "this skill may not enter the library" catches these too.
"""

from __future__ import annotations

import ast
import json
import keyword
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from skillweaver.contracts import Fingerprint, Provenance, Skill, SkillStats
from skillweaver.errors import AdmissionRejected, SkillWeaverError

__all__ = [
    "JSON_TYPES",
    "SCHEMA_VERSION",
    "InvalidSkillCode",
    "InvalidSkillDomain",
    "InvalidSkillName",
    "InvalidSkillParams",
    "InvalidSkillRequires",
    "InvalidSkillText",
    "SkillInvalid",
    "from_dict",
    "make_skill",
    "signature",
    "to_dict",
    "validate_code",
    "validate_domain",
    "validate_identity",
    "validate_name",
    "validate_params",
    "validate_requires",
    "validate_skill",
]


# --------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------


class SkillInvalid(AdmissionRejected):
    """A skill is structurally malformed and cannot enter the library.

    Catch this to reject any bad skill; catch a subclass to tell the synthesizer
    precisely what to fix.
    """


class InvalidSkillName(SkillInvalid):
    """``name`` is not a snake-case Python identifier."""


class InvalidSkillDomain(SkillInvalid):
    """``domain`` is empty or cannot be used as a directory name."""


class InvalidSkillText(SkillInvalid):
    """``summary`` or ``docstring`` is empty, or ``summary`` is not one line."""


class InvalidSkillParams(SkillInvalid):
    """``params`` is not a coherent JSON schema, or disagrees with ``run``."""


class InvalidSkillCode(SkillInvalid):
    """``code`` does not parse, or does not define a usable ``run(ctx, ...)``."""


class InvalidSkillRequires(SkillInvalid):
    """``requires`` holds a malformed, duplicated or self-referential name."""


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------

SCHEMA_VERSION = 1
"""Version of the dict produced by :func:`to_dict`; bumped on a breaking change."""

JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})
"""The JSON-schema primitive type names a param may declare."""

_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}

_TYPE_CHECKS: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

_BAD_PATH_CHARS = frozenset({"/", "\\", "\x00"})


def validate_name(name: str, *, what: str = "name") -> None:
    """Check a skill name: a lower-case Python identifier that is not a keyword and
    does not start with an underscore.

    The name is both a callable name in prompts and a directory component on disk, so
    an identifier is exactly the right shape for it.

    Raises:
        InvalidSkillName: with the reason in the message.
    """
    if not isinstance(name, str) or not name:
        raise InvalidSkillName(f"{what} must be a non-empty string, got {name!r}")
    if not name.isidentifier():
        raise InvalidSkillName(f"{what} {name!r} is not a legal Python identifier")
    if keyword.iskeyword(name) or keyword.issoftkeyword(name):
        raise InvalidSkillName(f"{what} {name!r} is a Python keyword")
    if name.startswith("_"):
        raise InvalidSkillName(f"{what} {name!r} must not start with an underscore")
    if any(ch.isupper() for ch in name):
        raise InvalidSkillName(f"{what} {name!r} must be snake_case, not {name!r}")


def validate_domain(domain: str) -> None:
    """Check a domain: a non-empty string usable as one directory name
    (``"example.com"``, ``"desktop:finder"``).

    Raises:
        InvalidSkillDomain: if it is empty, padded, a path traversal or holds a
            separator.
    """
    if not isinstance(domain, str) or not domain:
        raise InvalidSkillDomain(f"domain must be a non-empty string, got {domain!r}")
    if domain != domain.strip():
        raise InvalidSkillDomain(f"domain {domain!r} has leading or trailing whitespace")
    if domain in {".", ".."}:
        raise InvalidSkillDomain(f"domain {domain!r} is a path traversal")
    bad = sorted(_BAD_PATH_CHARS & set(domain))
    if bad:
        raise InvalidSkillDomain(f"domain {domain!r} may not contain {bad}")


def validate_identity(skill: Skill) -> None:
    """Check only ``name`` and ``domain`` - the part that decides where the skill is
    stored. :meth:`~skillweaver.contracts.SkillStore.put` runs this so a malformed
    key can never reach the filesystem.

    Raises:
        InvalidSkillName, InvalidSkillDomain: per :func:`validate_name` /
            :func:`validate_domain`.
    """
    validate_name(skill.name)
    validate_domain(skill.domain)


def _validate_one_schema(name: str, schema: Any, *, where: str) -> None:
    if not isinstance(schema, Mapping):
        raise InvalidSkillParams(f"{where} must be a JSON-schema object, got {schema!r}")
    if any(not isinstance(key, str) for key in schema):
        raise InvalidSkillParams(f"{where} has a non-string key")

    declared = schema.get("type")
    if declared is None:
        raise InvalidSkillParams(f"{where} is missing a 'type'")
    types = declared if isinstance(declared, list) else [declared]
    if not types:
        raise InvalidSkillParams(f"{where} declares an empty 'type' list")
    for one in types:
        if one not in JSON_TYPES:
            raise InvalidSkillParams(
                f"{where} declares unknown type {one!r}; expected one of {sorted(JSON_TYPES)}"
            )

    description = schema.get("description")
    if description is not None and not isinstance(description, str):
        raise InvalidSkillParams(f"{where} 'description' must be a string, got {description!r}")

    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise InvalidSkillParams(f"{where} 'enum' must be a non-empty list")

    if "default" in schema:
        default = schema["default"]
        allowed = tuple(t for one in types for t in _TYPE_CHECKS[one])
        # bool is an int in Python; only accept it where the schema really says so.
        if isinstance(default, bool) and "boolean" not in types:
            raise InvalidSkillParams(f"{where} default {default!r} is not a {declared!r}")
        if not isinstance(default, allowed):
            raise InvalidSkillParams(f"{where} default {default!r} is not a {declared!r}")

    if "items" in schema:
        _validate_one_schema(name, schema["items"], where=f"{where} 'items'")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise InvalidSkillParams(f"{where} 'properties' must be an object")
        for prop, sub in properties.items():
            _validate_one_schema(prop, sub, where=f"{where} property {prop!r}")

    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list) or any(not isinstance(r, str) for r in required)
    ):
        raise InvalidSkillParams(f"{where} 'required' must be a list of strings")


def validate_params(params: Mapping[str, Any]) -> None:
    """Check that ``params`` is a coherent JSON schema per parameter.

    Each key must be a legal Python identifier (it becomes a keyword argument) and
    each value a JSON-schema object with a known ``type``. ``description``, ``enum``,
    ``default``, ``items``, ``properties`` and ``required`` are checked for shape when
    present, and a ``default`` must match its declared ``type``. The whole mapping
    must be JSON-serializable, because it is stored as JSON.

    Raises:
        InvalidSkillParams: with the offending parameter in the message.
    """
    if not isinstance(params, Mapping):
        raise InvalidSkillParams(f"params must be a mapping, got {type(params).__name__}")
    for name, schema in params.items():
        if not isinstance(name, str):
            raise InvalidSkillParams(f"param name {name!r} must be a string")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise InvalidSkillParams(f"param name {name!r} is not a legal Python identifier")
        _validate_one_schema(name, schema, where=f"param {name!r}")
    try:
        json.dumps(dict(params))
    except (TypeError, ValueError) as exc:
        raise InvalidSkillParams(f"params is not JSON-serializable: {exc}") from exc


def _module_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _signature_of(fn: ast.FunctionDef) -> tuple[list[str], list[str], bool]:
    """``(accepted, required, takes_kwargs)`` for a parsed ``def``, excluding the
    leading ``ctx``."""
    args = fn.args
    positional = [a.arg for a in (*args.posonlyargs, *args.args)]
    first_optional = len(positional) - len(args.defaults)
    required = [a for a in positional[1:first_optional]]
    required += [a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d is None]
    accepted = positional[1:] + [a.arg for a in args.kwonlyargs]
    return accepted, required, args.kwarg is not None


def validate_code(code: str, params: Mapping[str, Any] | None = None) -> None:
    """Check that ``code`` parses and defines a module-level ``def run(ctx, ...)``
    whose parameters agree with ``params``.

    Agreement means: ``run`` accepts every declared param (directly or via
    ``**kwargs``), and every parameter ``run`` requires is declared - so
    ``run(ctx, **skill.params_filled_in)`` can never raise ``TypeError``.

    Raises:
        InvalidSkillCode: if the source does not parse, or ``run`` is missing,
            asynchronous, takes no ``ctx`` first, or rejects a declared param.
        InvalidSkillParams: if ``run`` requires a parameter ``params`` never declared.
    """
    if not isinstance(code, str) or not code.strip():
        raise InvalidSkillCode("code must be a non-empty string of Python source")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise InvalidSkillCode(f"code does not parse: {exc.msg} (line {exc.lineno})") from exc

    fn = _module_function(tree, "run")
    if fn is None:
        raise InvalidSkillCode("code must define a module-level `def run(ctx, ...)`")
    if isinstance(fn, ast.AsyncFunctionDef):
        raise InvalidSkillCode("run() must be a plain `def`, not `async def`")

    accepted, required, takes_kwargs = _signature_of(fn)
    if not (fn.args.posonlyargs or fn.args.args) or (
        [*fn.args.posonlyargs, *fn.args.args][0].arg != "ctx"
    ):
        raise InvalidSkillCode("run()'s first parameter must be `ctx`")

    declared = dict(params or {})
    if not takes_kwargs:
        unknown = [p for p in declared if p not in accepted]
        if unknown:
            raise InvalidSkillCode(f"run() does not accept declared param(s) {sorted(unknown)}")
    undeclared = [p for p in required if p not in declared]
    if undeclared:
        raise InvalidSkillParams(f"run() requires undeclared param(s) {sorted(undeclared)}")


def _validate_verifier(verifier_code: str) -> None:
    try:
        tree = ast.parse(verifier_code)
    except SyntaxError as exc:
        raise InvalidSkillCode(
            f"verifier_code does not parse: {exc.msg} (line {exc.lineno})"
        ) from exc
    fn = _module_function(tree, "verify")
    if fn is None or isinstance(fn, ast.AsyncFunctionDef):
        raise InvalidSkillCode("verifier_code must define a module-level `def verify(ctx, result)`")
    positional = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    if positional[:2] != ["ctx", "result"]:
        raise InvalidSkillCode(f"verify() must take (ctx, result), got ({', '.join(positional)})")


def validate_requires(requires: Sequence[str], own_name: str = "") -> None:
    """Check that every name in ``requires`` is a well-formed skill name, that none
    repeats, and that the skill does not require itself.

    Raises:
        InvalidSkillRequires: with the offending name in the message.
    """
    if isinstance(requires, str) or not isinstance(requires, Sequence):
        raise InvalidSkillRequires(f"requires must be a sequence of names, got {requires!r}")
    seen: set[str] = set()
    for dependency in requires:
        try:
            validate_name(dependency, what="required skill name")
        except InvalidSkillName as exc:
            raise InvalidSkillRequires(str(exc)) from exc
        if dependency in seen:
            raise InvalidSkillRequires(f"requires lists {dependency!r} twice")
        if dependency == own_name:
            raise InvalidSkillRequires(f"skill {own_name!r} requires itself")
        seen.add(dependency)


def validate_skill(skill: Skill) -> Skill:
    """Run every structural check on ``skill`` and return it unchanged.

    Checks, in the order they are reported: name, domain, summary and docstring,
    params, code against params, verifier code, requires, and the stats invariants.

    Raises:
        SkillInvalid: a subclass naming the first thing that was wrong.
    """
    validate_name(skill.name)
    validate_domain(skill.domain)

    if not skill.summary or not skill.summary.strip():
        raise InvalidSkillText(f"skill {skill.name!r} has an empty summary")
    if "\n" in skill.summary.strip():
        raise InvalidSkillText(
            f"skill {skill.name!r} summary must be ONE line, got {skill.summary!r}"
        )
    if not skill.docstring or not skill.docstring.strip():
        raise InvalidSkillText(f"skill {skill.name!r} has an empty docstring")

    validate_params(skill.params)
    validate_code(skill.code, skill.params)
    if skill.verifier_code is not None:
        _validate_verifier(skill.verifier_code)
    validate_requires(skill.requires, skill.name)

    stats = skill.stats
    if stats.runs < 0 or stats.successes < 0 or stats.successes > stats.runs:
        raise SkillInvalid(
            f"skill {skill.name!r} has impossible stats: "
            f"{stats.successes} successes out of {stats.runs} runs"
        )
    if skill.version < 0:
        raise SkillInvalid(f"skill {skill.name!r} has a negative version {skill.version}")
    return skill


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def make_skill(
    *,
    name: str,
    domain: str,
    summary: str,
    docstring: str,
    code: str,
    provenance: Provenance,
    params: Mapping[str, Any] | None = None,
    requires: Sequence[str] = (),
    precondition: Fingerprint | None = None,
    verifier_code: str | None = None,
    version: int = 0,
    stats: SkillStats | None = None,
    demoted_reason: str | None = None,
    validate: bool = True,
) -> Skill:
    """Build a validated :class:`~skillweaver.contracts.Skill`.

    Normalizes the shapes a generator is likely to get slightly wrong - ``params`` as
    a plain dict, ``requires`` as a list - and validates the result.

    Args:
        validate: ``False`` to skip :func:`validate_skill`; only for tests that need
            a deliberately broken skill.

    Raises:
        SkillInvalid: a subclass naming what was wrong.
    """
    skill = Skill(
        name=name,
        domain=domain,
        summary=summary,
        docstring=docstring,
        params=dict(params or {}),
        code=code,
        requires=tuple(requires),
        precondition=precondition,
        verifier_code=verifier_code,
        provenance=provenance,
        version=version,
        stats=stats if stats is not None else SkillStats(),
        demoted_reason=demoted_reason,
    )
    return validate_skill(skill) if validate else skill


def signature(skill: Skill) -> str:
    """Render ``skill`` as the call a planner would write.

    ``search_invoice(company: str, limit: int = 10)``. Parameters keep their declared
    order except that optional ones move after required ones, so the result is always
    a legal Python signature. ``ctx`` is left out: the runner supplies it.
    """
    required: list[str] = []
    optional: list[str] = []
    for name, schema in skill.params.items():
        declared = schema.get("type") if isinstance(schema, Mapping) else None
        types = declared if isinstance(declared, list) else [declared]
        rendered = " | ".join(_PYTHON_TYPES.get(t, "Any") for t in types if t is not None)
        annotation = f"{name}: {rendered}" if rendered else name
        if isinstance(schema, Mapping) and "default" in schema:
            optional.append(f"{annotation} = {schema['default']!r}")
        else:
            required.append(annotation)
    return f"{skill.name}({', '.join(required + optional)})"


# --------------------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _parse_time(raw: Any, field: str) -> datetime:
    if not isinstance(raw, str):
        raise SkillWeaverError(f"{field} must be an ISO-8601 string, got {raw!r}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SkillWeaverError(f"{field} is not an ISO-8601 timestamp: {raw!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def to_dict(skill: Skill, *, include_code: bool = True) -> dict[str, Any]:
    """Serialize ``skill`` to JSON-ready primitives.

    Args:
        include_code: ``False`` omits ``code`` and ``verifier_code``, for a store
            that keeps the source in its own file next to the metadata.
    """
    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": skill.name,
        "domain": skill.domain,
        "summary": skill.summary,
        "docstring": skill.docstring,
        "params": dict(skill.params),
        "requires": list(skill.requires),
        "precondition": (
            None
            if skill.precondition is None
            else {"value": skill.precondition.value, "parts": dict(skill.precondition.parts)}
        ),
        "provenance": {
            "trajectory_id": skill.provenance.trajectory_id,
            "task_text": skill.provenance.task_text,
            "model": skill.provenance.model,
            "created_at": _iso(skill.provenance.created_at),
        },
        "version": skill.version,
        "stats": {
            "runs": skill.stats.runs,
            "successes": skill.stats.successes,
            "mean_ms": skill.stats.mean_ms,
            "last_ok_at": None if skill.stats.last_ok_at is None else _iso(skill.stats.last_ok_at),
        },
        "demoted_reason": skill.demoted_reason,
    }
    if include_code:
        data["code"] = skill.code
        data["verifier_code"] = skill.verifier_code
    return data


def from_dict(
    data: Mapping[str, Any],
    *,
    code: str | None = None,
    verifier_code: str | None = None,
    validate: bool = False,
) -> Skill:
    """Rebuild a :class:`~skillweaver.contracts.Skill` from :func:`to_dict` output.

    Args:
        code: the source, when ``data`` was written with ``include_code=False``.
            It wins over any ``code`` in ``data``.
        verifier_code: likewise for the verifier.
        validate: ``True`` to re-run :func:`validate_skill` on the result. Off by
            default so a store can always read back what it once wrote, even after
            the rules tightened.

    Raises:
        SkillWeaverError: on a missing field, a bad timestamp or an unknown
            ``schema_version``.
        SkillInvalid: when ``validate`` is on and the rebuilt skill is malformed.
    """
    found = data.get("schema_version", SCHEMA_VERSION)
    if found != SCHEMA_VERSION:
        raise SkillWeaverError(
            f"skill schema_version {found!r} is not supported (this build reads {SCHEMA_VERSION})"
        )
    try:
        provenance_raw = data["provenance"]
        provenance = Provenance(
            trajectory_id=provenance_raw["trajectory_id"],
            task_text=provenance_raw["task_text"],
            model=provenance_raw["model"],
            created_at=_parse_time(provenance_raw["created_at"], "provenance.created_at"),
        )
        stats_raw = data.get("stats") or {}
        last_ok_raw = stats_raw.get("last_ok_at")
        stats = SkillStats(
            runs=int(stats_raw.get("runs", 0)),
            successes=int(stats_raw.get("successes", 0)),
            mean_ms=float(stats_raw.get("mean_ms", 0.0)),
            last_ok_at=(
                None if last_ok_raw is None else _parse_time(last_ok_raw, "stats.last_ok_at")
            ),
        )
        precondition_raw = data.get("precondition")
        precondition = (
            None
            if precondition_raw is None
            else Fingerprint(
                value=precondition_raw["value"], parts=dict(precondition_raw.get("parts") or {})
            )
        )
        skill = Skill(
            name=data["name"],
            domain=data["domain"],
            summary=data["summary"],
            docstring=data["docstring"],
            params=dict(data.get("params") or {}),
            code=code if code is not None else data["code"],
            requires=tuple(data.get("requires") or ()),
            precondition=precondition,
            verifier_code=(
                verifier_code if verifier_code is not None else data.get("verifier_code")
            ),
            provenance=provenance,
            version=int(data.get("version", 0)),
            stats=stats,
            demoted_reason=data.get("demoted_reason"),
        )
    except KeyError as exc:
        raise SkillWeaverError(f"serialized skill is missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise SkillWeaverError(f"serialized skill is malformed: {exc}") from exc
    return validate_skill(skill) if validate else skill
