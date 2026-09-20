"""Checks for the multi-item errand: the split, the binder it is shaped for, the ground
truth, and the rule that no efficiency number is printed without it.

Pure logic: no browser, no model, no network.

    uv run python scripts/check_j_errand.py

One line per check; exits non-zero if any failed. What these CANNOT show is whether the
learned skill replays for a second product on a real shop - that is a live run
(``scripts/run_errand.py``).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from skillweaver.agent.errand import (
    Decomposition,
    ErrandReport,
    ItemResult,
    Line,
    _check_orchestrator_seam,
    check_truth,
    decompose,
    quantity_of,
)
from skillweaver.agent.planner import MIN_ACCOUNTED_FOR, account_of, asks_for, bind_args
from skillweaver.contracts import Provenance, Skill, TaskSpec
from skillweaver.errors import ProviderError

_failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{f' - {detail}' if detail else ''}")
    if not ok:
        _failed.append(name)


LEARNED = 'Add the "LPF Glow Legended Low Profile MX Keycaps" to the cart'
ERRAND = 'Add the "Alpha Caps", "Beta Switch Puller" and "Gamma Cable" to the cart'


def _skill() -> Skill:
    """The skill a live splitkb run stored (f-skb-B1): one required, quoted parameter."""
    return Skill(
        name="search_and_add_product_to_cart",
        domain="dom@splitkb.com",
        summary="Search for a product and add it to the cart",
        docstring="",
        params={
            "product": {
                "type": "string",
                "description": "Product name, e.g. 'LPF Glow Legended Low Profile MX Keycaps'.",
            }
        },
        code="def run(ctx, product):\n    return product\n",
        requires=(),
        precondition=None,
        verifier_code="def verify(ctx, result):\n    return True\n",
        provenance=Provenance("t", LEARNED, "m", datetime.now(UTC)),
    )


def _task(text: str) -> TaskSpec:
    return TaskSpec(text=text, domain="dom@splitkb.com", target="browser", params={})


def the_split() -> None:
    plan = decompose(ERRAND)
    check("a quoted list splits", plan.how == "quoted-list" and len(plan.tasks) == 3, plan.why)
    check(
        "each task is the person's own head and tail around one item",
        plan.tasks[1] == 'Add the "Beta Switch Puller" to the cart',
        str(plan.tasks),
    )
    check(
        "items keep their order", plan.items == ("Alpha Caps", "Beta Switch Puller", "Gamma Cable")
    )
    for sep in (", ", " and ", ", and ", " & ", "; ", ", then "):
        text = f'Add "A one"{sep}"B two" to the cart'
        check(f"separator {sep!r} is a list", len(decompose(text).tasks) == 2)
    one = decompose('Add "A one" to the cart')
    check("one quotation is one task, verbatim", one.how == "single" and one.tasks == (one.errand,))
    dup = decompose('Add "A one", "a ONE" and "B two" to the cart')
    check("a repeated item is asked for once", dup.items == ("A one", "B two"))
    mixed = decompose('Add "A one" to the cart and remove "B two" from it')
    check("two different errands are refused, not split", not mixed and "separator" in mixed.why)
    bare = decompose("Add A one, B two and C three to the cart")
    check("an unquoted errand declines without a fallback", not bare and "no fallback" in bare.why)


def the_fallback() -> None:
    text = "Add Alpha Caps, Beta Puller and Gamma Cable to the cart"
    named = lambda _: 'Add "Alpha Caps", "Beta Puller" and "Gamma Cable" to the cart'  # noqa: E731
    plan = decompose(text, name_items=named)
    check(
        "a named list is split around the PERSON'S words",
        plan.how == "named-list" and plan.tasks[0] == 'Add "Alpha Caps" to the cart',
        str(plan.tasks) + plan.why,
    )
    invented = lambda _: 'Add "Alpha Caps Deluxe" and "Beta Puller" to the cart'  # noqa: E731
    refused = decompose(text, name_items=invented)
    check("an item the person never typed is refused", not refused, refused.why)

    def broken(_: str) -> str:
        raise ProviderError("no reply")

    check(
        "a failed fallback is an answer, not an exception", not decompose(text, name_items=broken)
    )


def the_binder() -> None:
    """Why the split has to happen BEFORE the planner: what it does with the whole list,
    and what it does with one item of it."""
    skill = _skill()
    check("the planner cannot bind the whole errand", bind_args(skill, _task(ERRAND)) is None)
    for text in decompose(ERRAND).tasks:
        task = _task(text)
        args = bind_args(skill, task)
        item = text.split('"')[1]
        check(f"binds {item!r} from its quoted slot", args == {"product": item}, str(args))
        if args is None:
            continue
        check(f"  and it is the same intent ({item})", asks_for(task, skill, args))
        share, _ = account_of(task, skill, args, lambda: [skill])
        check(f"  and accounted for ({item})", share >= MIN_ACCOUNTED_FOR, f"{share:.2f}")
    other_shape = _task('Put "Beta Switch Puller" in my basket')
    check("a differently worded item does NOT bind", bind_args(skill, other_shape) is None)


def the_truth() -> None:
    lines = [Line("Alpha Caps", 1), Line("alpha  caps", 1), Line("Alpha Caps Add-on", 1)]
    check("quantity sums lines that ARE the item", quantity_of("Alpha Caps", lines) == 2)
    good = check_truth(["A", "B"], [Line("A", 1), Line("B", 1)])
    check("every item exactly once passes", good.ok, str(good))
    check("an item twice fails", not check_truth(["A"], [Line("A", 2)]).ok)
    check("an item missing fails", not check_truth(["A", "B"], [Line("A", 1)]).ok)
    check("an unasked item fails", not check_truth(["A"], [Line("A", 1), Line("Z", 1)]).ok)
    check("an empty errand never passes", not check_truth([], []).ok)


def _row(item: str, path: str, ms: float, on_site: int | None, ok: bool = True) -> ItemResult:
    return ItemResult(item, f'Add "{item}"', path, ok, ms, 0.0, 5, 0, 0.0, on_site=on_site)  # type: ignore[arg-type]


def no_number_without_truth() -> None:
    plan = Decomposition("e", ("A", "B"), ('Add "A"', 'Add "B"'), "quoted-list")
    rows = (_row("A", "cold", 15000, 1), _row("B", "warm", 4000, 1))
    unread = ErrandReport(plan, rows, None)
    check("no reader: no seconds per item", "seconds per item" not in unread.explain())
    check("  and no efficiency in the JSON", "efficiency" not in unread.to_json())
    failed = ErrandReport(plan, rows, check_truth(["A", "B"], [Line("A", 1)]))
    check("failed truth: no seconds per item", "seconds per item" not in failed.explain())
    check("  and the errand is not ok", not failed.ok)
    passed = ErrandReport(plan, rows, check_truth(["A", "B"], [Line("A", 1), Line("B", 1)]))
    check("passing truth prints it", "warm: 4.0s per item (n=1" in passed.explain())
    check("  and the JSON carries it", passed.to_json()["efficiency"]["warm"]["mean_ms"] == 4000)
    ghost = (_row("A", "warm", 900, 0), _row("B", "warm", 4000, 1))
    kept = ErrandReport(plan, ghost, None).measured("warm")
    check("a fast row the site does not confirm is not measured", kept == [4000], str(kept))


def the_seam() -> None:
    try:
        _check_orchestrator_seam()
        check("the orchestrator openers this module borrows exist", True)
    except Exception as exc:  # noqa: BLE001
        check("the orchestrator openers this module borrows exist", False, str(exc))


if __name__ == "__main__":
    the_split()
    the_fallback()
    the_binder()
    the_truth()
    no_number_without_truth()
    the_seam()
    print(f"\n{len(_failed)} failed" if _failed else "\nall passed")
    sys.exit(1 if _failed else 0)
