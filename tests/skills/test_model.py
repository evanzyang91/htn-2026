"""Validation and serialization of a :class:`Skill`.

Every rejection test names the specific exception, because the synthesizer reads the
exception type to decide what to regenerate.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from skillweaver.contracts import Fingerprint, Provenance, Skill, SkillStats
from skillweaver.errors import AdmissionRejected, SkillWeaverError
from skillweaver.skills.model import (
    InvalidSkillCode,
    InvalidSkillDomain,
    InvalidSkillName,
    InvalidSkillParams,
    InvalidSkillRequires,
    InvalidSkillText,
    SkillInvalid,
    from_dict,
    make_skill,
    signature,
    to_dict,
    validate_skill,
)

PROVENANCE = Provenance("run-7", "pay an invoice", "fake-llm", datetime(2026, 9, 19, tzinfo=UTC))


def build(**overrides: object) -> Skill:
    """A valid skill with ``overrides`` applied, built WITHOUT validation so a test
    can hand a deliberately broken one to :func:`validate_skill`."""
    fields: dict[str, object] = {
        "name": "search_invoice",
        "domain": "example.com",
        "summary": "Search the invoice list for a company.",
        "docstring": "Types the company name into the search box and waits for rows.",
        "params": {"company": {"type": "string"}},
        "code": "def run(ctx, company):\n    ctx.ctl.type_text(company)\n",
        "provenance": PROVENANCE,
        "validate": False,
    }
    fields.update(overrides)
    return make_skill(**fields)  # type: ignore[arg-type]


# -- the happy path --------------------------------------------------------------------


def test_a_valid_skill_passes_and_is_returned_unchanged() -> None:
    skill = build()
    assert validate_skill(skill) is skill


def test_make_skill_validates_and_normalizes_shapes() -> None:
    skill = make_skill(
        name="pay_invoice",
        domain="example.com",
        summary="Pay the selected invoice.",
        docstring="Clicks Confirm payment and waits for the receipt.",
        code="def run(ctx):\n    pass\n",
        provenance=PROVENANCE,
        requires=["search_invoice"],  # a list, as a generator would give it
    )
    assert skill.requires == ("search_invoice",)  # normalized to a tuple
    assert skill.version == 0 and skill.stats == SkillStats() and skill.demoted_reason is None


def test_sample_skill_fixture_passes_validation(sample_skill: Skill) -> None:
    assert validate_skill(sample_skill) is sample_skill


# -- names -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["search invoice", "search-invoice", "2search", "class", "", "_private", "searchInvoice"],
)
def test_an_illegal_name_is_rejected(name: str) -> None:
    with pytest.raises(InvalidSkillName) as caught:
        validate_skill(build(name=name))
    assert repr(name) in str(caught.value)


@pytest.mark.parametrize("domain", ["", "a/b", "..", " example.com", "null\x00byte"])
def test_an_unusable_domain_is_rejected(domain: str) -> None:
    with pytest.raises(InvalidSkillDomain):
        validate_skill(build(domain=domain))


def test_a_domain_with_a_colon_is_fine() -> None:
    # Desktop domains look like "desktop:finder"; only path separators are banned.
    assert validate_skill(build(domain="desktop:finder")).domain == "desktop:finder"


# -- text ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"summary": ""},
        {"summary": "   "},
        {"summary": "two\nlines"},
        {"docstring": ""},
    ],
    ids=["empty-summary", "blank-summary", "multiline-summary", "empty-docstring"],
)
def test_unusable_prompt_text_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidSkillText):
        validate_skill(build(**overrides))


# -- code ------------------------------------------------------------------------------


def test_code_that_does_not_parse_is_rejected() -> None:
    with pytest.raises(InvalidSkillCode) as caught:
        validate_skill(build(code="def run(ctx:\n  oops", params={}))
    assert "does not parse" in str(caught.value)


def test_code_without_run_is_rejected() -> None:
    with pytest.raises(InvalidSkillCode, match="must define a module-level"):
        validate_skill(build(code="def helper(ctx):\n    pass\n", params={}))


def test_run_nested_inside_a_class_does_not_count() -> None:
    code = "class Skill:\n    def run(ctx):\n        pass\n"
    with pytest.raises(InvalidSkillCode, match="module-level"):
        validate_skill(build(code=code, params={}))


def test_run_must_take_ctx_first() -> None:
    with pytest.raises(InvalidSkillCode, match="first parameter must be `ctx`"):
        validate_skill(build(code="def run(company):\n    pass\n"))


def test_async_run_is_rejected() -> None:
    with pytest.raises(InvalidSkillCode, match="not `async def`"):
        validate_skill(build(code="async def run(ctx, company):\n    pass\n"))


def test_run_that_cannot_accept_a_declared_param_is_rejected() -> None:
    with pytest.raises(
        InvalidSkillCode, match=r"does not accept declared param\(s\) \['company'\]"
    ):
        validate_skill(build(code="def run(ctx):\n    pass\n"))


def test_run_with_kwargs_accepts_any_declared_param() -> None:
    validate_skill(build(code="def run(ctx, **kw):\n    pass\n"))


def test_keyword_only_and_defaulted_params_are_understood() -> None:
    skill = build(
        code="def run(ctx, company, *, limit=10):\n    pass\n",
        params={"company": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
    )
    assert validate_skill(skill) is skill


def test_a_param_run_requires_but_never_declares_is_rejected() -> None:
    with pytest.raises(InvalidSkillParams, match=r"requires undeclared param\(s\) \['company'\]"):
        validate_skill(build(code="def run(ctx, company):\n    pass\n", params={}))


def test_a_broken_verifier_is_rejected() -> None:
    with pytest.raises(InvalidSkillCode, match="verifier_code"):
        validate_skill(build(verifier_code="def verify(ctx result):\n    return True\n"))
    with pytest.raises(InvalidSkillCode, match=r"verify\(\) must take"):
        validate_skill(build(verifier_code="def verify(ctx):\n    return True\n"))


def test_a_good_verifier_is_accepted() -> None:
    validate_skill(build(verifier_code="def verify(ctx, result):\n    return result.ok\n"))


# -- params ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "needle"),
    [
        ({"company": {"description": "who"}}, "missing a 'type'"),
        ({"company": {"type": "text"}}, "unknown type 'text'"),
        ({"company": "string"}, "JSON-schema object"),
        ({"company": {"type": []}}, "empty 'type' list"),
        ({"company": {"type": "string", "description": 7}}, "'description' must be a string"),
        ({"company": {"type": "string", "enum": []}}, "'enum' must be a non-empty list"),
        ({"company": {"type": "string", "default": 7}}, "is not a 'string'"),
        ({"company": {"type": "integer", "default": True}}, "is not a 'integer'"),
        ({"rows": {"type": "array", "items": {"type": "nope"}}}, "unknown type 'nope'"),
        ({"class": {"type": "string"}}, "not a legal Python identifier"),
        ({"a b": {"type": "string"}}, "not a legal Python identifier"),
    ],
)
def test_an_incoherent_params_schema_is_rejected(params: dict, needle: str) -> None:
    with pytest.raises(InvalidSkillParams) as caught:
        validate_skill(build(params=params, code="def run(ctx, **kw):\n    pass\n"))
    assert needle in str(caught.value)


def test_params_that_are_not_json_serializable_are_rejected() -> None:
    with pytest.raises(InvalidSkillParams, match="not JSON-serializable"):
        validate_skill(
            build(
                params={"when": {"type": "string", "default": "x", "seen": {datetime.now(UTC)}}},
                code="def run(ctx, **kw):\n    pass\n",
            )
        )


def test_a_rich_params_schema_is_accepted() -> None:
    params = {
        "company": {"type": "string", "description": "Legal name", "enum": ["Acme", "Globex"]},
        "rows": {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
        },
        "dry_run": {"type": "boolean", "default": False},
        "amount": {"type": ["number", "null"], "default": None},
    }
    validate_skill(build(params=params, code="def run(ctx, **kw):\n    pass\n"))


# -- requires --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requires", "needle"),
    [
        (("open panel",), "not a legal Python identifier"),
        (("open_panel", "open_panel"), "twice"),
        (("search_invoice",), "requires itself"),
        (("Open",), "snake_case"),
    ],
)
def test_malformed_requires_is_rejected(requires: tuple[str, ...], needle: str) -> None:
    with pytest.raises(InvalidSkillRequires) as caught:
        validate_skill(build(requires=requires))
    assert needle in str(caught.value)


def test_well_formed_requires_is_accepted() -> None:
    assert validate_skill(build(requires=("open_panel", "pay_invoice"))).requires == (
        "open_panel",
        "pay_invoice",
    )


# -- stats invariants ------------------------------------------------------------------


def test_impossible_stats_are_rejected() -> None:
    with pytest.raises(SkillInvalid, match="impossible stats"):
        validate_skill(build(stats=SkillStats(runs=1, successes=2)))


def test_every_rejection_is_catchable_as_admission_rejected() -> None:
    # One except clause at the synthesis boundary must cover all of them.
    with pytest.raises(AdmissionRejected):
        validate_skill(build(name="nope!"))


# -- signature -------------------------------------------------------------------------


def test_signature_renders_the_call_a_planner_would_write() -> None:
    skill = build(
        params={"company": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
        code="def run(ctx, company, limit=10):\n    pass\n",
    )
    assert signature(skill) == "search_invoice(company: str, limit: int = 10)"


def test_signature_puts_optional_params_last_so_it_is_legal_python() -> None:
    skill = build(
        params={"limit": {"type": "integer", "default": 10}, "company": {"type": "string"}},
        code="def run(ctx, company, limit=10):\n    pass\n",
    )
    rendered = signature(skill)
    assert rendered == "search_invoice(company: str, limit: int = 10)"
    compile(f"def {rendered}: pass", "<signature>", "exec")


def test_signature_of_a_skill_with_no_params() -> None:
    assert signature(build(params={}, code="def run(ctx):\n    pass\n")) == "search_invoice()"


def test_signature_renders_every_json_type_and_unions() -> None:
    skill = build(
        params={
            "s": {"type": "string"},
            "n": {"type": "number"},
            "i": {"type": "integer"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
            "z": {"type": "null"},
            "u": {"type": ["string", "integer"]},
        },
        code="def run(ctx, **kw):\n    pass\n",
    )
    assert signature(skill) == (
        "search_invoice(s: str, n: float, i: int, b: bool, a: list, o: dict, z: None, u: str | int)"
    )


# -- serialization ---------------------------------------------------------------------


def full_skill() -> Skill:
    return build(
        params={"company": {"type": "string", "description": "Legal name"}},
        requires=("open_panel",),
        precondition=Fingerprint("fp-list", {"url": "u1", "layout": "l1"}),
        verifier_code="def verify(ctx, result):\n    return result.ok\n",
        version=3,
        stats=SkillStats(
            runs=5, successes=4, mean_ms=412.5, last_ok_at=datetime(2026, 9, 18, tzinfo=UTC)
        ),
        demoted_reason=None,
    )


def test_to_dict_from_dict_round_trips_every_field() -> None:
    skill = full_skill()
    assert from_dict(to_dict(skill)) == skill


def test_the_round_trip_survives_actual_json() -> None:
    skill = full_skill()
    assert from_dict(json.loads(json.dumps(to_dict(skill)))) == skill


def test_code_can_be_carried_beside_the_metadata() -> None:
    skill = full_skill()
    meta = to_dict(skill, include_code=False)
    assert "code" not in meta and "verifier_code" not in meta
    rebuilt = from_dict(meta, code=skill.code, verifier_code=skill.verifier_code)
    assert rebuilt == skill


def test_a_demoted_skill_round_trips_with_its_reason() -> None:
    skill = build(demoted_reason="selector moved in the 2026-09 redesign")
    assert from_dict(to_dict(skill)).demoted_reason == "selector moved in the 2026-09 redesign"


def test_a_naive_timestamp_is_read_back_as_utc() -> None:
    data = to_dict(build())
    data["provenance"]["created_at"] = "2026-09-19T00:00:00"
    assert from_dict(data).provenance.created_at == datetime(2026, 9, 19, tzinfo=UTC)


def test_an_unknown_schema_version_is_refused_rather_than_guessed() -> None:
    data = to_dict(build()) | {"schema_version": 99}
    with pytest.raises(SkillWeaverError, match="schema_version"):
        from_dict(data)


def test_a_missing_field_names_itself() -> None:
    data = to_dict(build())
    del data["summary"]
    with pytest.raises(SkillWeaverError, match="missing field 'summary'"):
        from_dict(data)


def test_a_bad_timestamp_names_itself() -> None:
    data = to_dict(build())
    data["stats"]["last_ok_at"] = "yesterday"
    with pytest.raises(SkillWeaverError, match="stats.last_ok_at"):
        from_dict(data)


def test_from_dict_can_revalidate_on_the_way_in() -> None:
    data = to_dict(replace(build(), name="not an identifier"))
    from_dict(data)  # tolerant by default: whatever was written can be read back
    with pytest.raises(InvalidSkillName):
        from_dict(data, validate=True)
