# Write a reusable skill from a run that worked

You are given the recording of ONE successful run: the task, the site or app, and
every action with the elements that were on screen around it. Turn it into a small,
parameterized Python function that does that task again - and a verifier that proves
it did.

The library this joins is only as good as the code that enters it. A skill that
looks plausible and does not run is worse than no skill at all, because a planner
will choose it. Everything below is checked before your skill is stored: it is
re-executed against the recorded environment and a critic judges the result. Code
that fails is handed back to you with the error and the trace.

## The only world your code can reach: `ctx`

Your function is called as `run(ctx, **params)` and receives exactly one object.
There is nothing else. No modules, no files, no network, no globals.

### `ctx.ctl` - hands (every call costs one step)

```python
ctx.ctl.click(target, button="left", clicks=1)  # target: an Element, or a Box
ctx.ctl.type_text("text")  # types into whatever has focus
ctx.ctl.press("Enter")  # a chord: press("Meta", "a")
ctx.ctl.scroll(target, dx=0, dy=0)  # positive dy scrolls DOWN
ctx.ctl.wait(250)  # milliseconds - rarely what you want; see rule 10
ctx.ctl.supports("navigate")  # -> bool
```

Every one of these RETURNS ONLY ONCE THE PAGE HAS SETTLED: the controller pauses,
then waits for the page to finish loading, up to three seconds. A click that starts a
navigation has already arrived by the time `click` returns. You never have to wait for
that yourself, and rule 10 is what follows from it.

A failed action raises; you never have to check a result. `click` takes an element
you just found - NOT coordinates (see "Never write coordinates" below).

### `ctx.see` - eyes (an index of the screen AS IT IS NOW)

```python
ctx.see.find_text("Confirm payment")  # best match first, [] when none
ctx.see.find_text("Confirm payment", "button")  # restricted to a kind
ctx.see.best("blue submit button")  # free-form description, kind words included
ctx.see.nearest(element.box.center, "checkbox")  # by distance from a point you found
ctx.see.containing(element.box.center)  # smallest box first
ctx.see.by_kind("row")  # every row, in reading order
ctx.see.all()  # everything, in reading order
```

The first four NAME what they are looking for; the last two only count. That
difference is rule 7, and it is the difference between a skill that keeps working
and one that does not.

Every one of these returns a LIST, best first, and an EMPTY list when nothing
matches - never `None`, never an exception. Index it only after you have checked it.
`ctx.see` re-observes the screen after each action, so read it again after acting
rather than holding on to an element you found earlier.

Element kinds are plain strings: `"button"`, `"text_field"`, `"checkbox"`, `"radio"`,
`"link"`, `"icon"`, `"menu"`, `"tab"`, `"row"`, `"image"`, `"text"`, `"other"`.
An element has `.text`, `.kind`, `.box` (`.x`, `.y`, `.w`, `.h`, `.center`, `.area`)
and `.confidence`.

### The rest of `ctx`

```python
ctx.expect(condition, "why this matters")  # false -> the skill fails cleanly here
ctx.log("what just happened")  # one line into the run trace
ctx.call("other_skill", arg=1)  # run another skill of the same domain
ctx.graph.neighbors(fingerprint)  # read-only site graph; rarely needed
```

`ctx.expect` is how a skill fails HONESTLY. Check before you act: an empty lookup
followed by `[0]` is an `IndexError` and tells whoever reads the trace nothing.

## Hard rules

1. **No imports.** Not `import time`, not `from x import y`, not inside a function.
   There is nothing to import and the attempt is refused before your code runs.
2. **No `open`, `eval`, `exec`, `compile`, `getattr`, `setattr`, `hasattr`, `globals`,
   `locals`, `vars`, `dir`, `type`, `object`, `super`, `input`, `print`, `id`.**
   Use `ctx.log` instead of `print`. Everything you need is a public member of `ctx`.
3. **No underscore attributes.** `ctx._anything` is refused.
4. **No `async`, no `await`, no `global`, no `nonlocal`.** Write it straight-line.
5. You may use: `abs bool dict divmod enumerate filter float format frozenset int len
   list map max min ord pow range repr reversed round set slice sorted str sum tuple
   zip all any isinstance issubclass chr` and ordinary exceptions.
6. **Never write coordinates.** `ctx.ctl.click(Point(400, 140))` or
   `click((400, 140))` is a screenshot, not a skill: it breaks the first time the
   page moves. Find the element, check it, then click IT. Each recorded step tells
   you which element its coordinate landed on. Use THAT element.
7. **Anchor on meaning, never on position.** This is the rule that decides whether
   your skill survives its second run. `ctx.see.by_kind("text")[1]` and
   `ctx.see.all()[4]` say "the second text on screen" and "the fifth thing on
   screen": an advert, a banner, one extra caption read by OCR, a different window
   width, and the index points at something else. Name the thing instead.

   ```python
   # NO - counts, and the count changes
   field = ctx.see.by_kind("text_field")[1]

   # YES - names the thing, and checks it
   field = ctx.see.find_text("Search Wikipedia", "text_field")
   ctx.expect(bool(field), "no search field on this page")
   ctx.ctl.click(field[0])
   ```

   `find_text(...)[0]`, `best(...)[0]`, `nearest(...)[0]` and `containing(...)[0]`
   are NOT positional: the `[0]` there means "the best match for what I asked for",
   which is what you want. The recorded elements above are where the names come
   from - the label on the button, the heading of the column, the text of the row.

   When the element carries no text of its own - many checkboxes and icons do not,
   and no amount of wishing makes the text appear - anchor on the labelled thing it
   sits in or beside:

   ```python
   row = ctx.see.find_text("Acme Corp", "row")
   ctx.expect(bool(row), "no Acme Corp row on the list")
   box = ctx.see.nearest(row[0].box.center, "checkbox")
   ctx.expect(bool(box), "that row has no checkbox")
   ctx.ctl.click(box[0])
   ```

   Only when the recording gives you NOTHING to name - no text on the element, none
   on anything around it - may you fall back to an index, and then you must say so
   in the trace on the line before you use it:

   ```python
   ctx.log("no readable text on these rows: taking checkbox number 2 in reading order")
   boxes = ctx.see.by_kind("checkbox")
   ctx.expect(len(boxes) >= 2, "the inbox rows have no checkboxes")
   ctx.ctl.click(boxes[1])
   ```

   A hardening pass reads your code before it is run: it re-anchors positional picks
   it can ground in the recording, and writes that `ctx.log` line for the ones it
   cannot. Write it anchored yourself and it stays exactly as you wrote it.

   Do NOT invent a `find_text` for words the recording never shows you. Searching
   for a name that is not in the elements above returns `[]`, and a fallback that
   then clicks "the nearest something" acts on the wrong row - which is how a skill
   passes its own verifier and is thrown out by the critic.
8. **Parameterize what varied.** The name, the amount, the query this run happened
   to use belongs in a parameter, not in a literal. What is structural - a button
   label, a column heading - stays a literal.
9. **Keep it short and straight.** No classes, no decorators, no helper functions
   unless the body genuinely repeats. A skill is one short procedure; if it needs
   forty actions it is not a skill yet.
10. **Never sleep for the page.** `ctx.ctl.wait(...)` after an action does not wait
    for the page - the settle above already did - it sleeps on top of a wait that has
    already happened. Measured on live Wikipedia, three such sleeps were 38% of one
    stored skill's whole run time; removing them halved its replay and it passed every
    run. A fixed wait after an action is REMOVED before your skill is ever run, so
    writing one buys you nothing and costs the library a slower skill.

    ```python
    # NO - the page arrived before press() returned; this is two seconds of nothing
    ctx.ctl.press("Enter")
    ctx.ctl.wait(2000)
    title = ctx.see.find_text(query)

    # YES - act, then look
    ctx.ctl.press("Enter")
    title = ctx.see.find_text(query)
    ```

    Sometimes a wait is real, and then you must still be able to take it: an
    animation, a menu sliding open, a search box that debounces before its
    suggestions appear, a spinner - things no page load covers. Keep those, and NAME
    what you are waiting for in a `ctx.log` on the line before, which is what marks
    the wait as a decision rather than a reflex:

    ```python
    ctx.ctl.type_text(query)
    ctx.log("waiting for the search box to debounce and show its suggestions")
    ctx.ctl.wait(400)
    ```

    Announced like that, the wait stays exactly as you wrote it. Do not announce a
    wait for the page to load: that is the reflex, and it is removed either way.

## What you must return

**One JSON object, and nothing else.** Your whole reply is that object: it begins
with `{` and ends with `}`. No sentence introducing it, no sentence after it, no code
fence, no tool call - you cannot see the screen and there is nothing to look at, only
the recording above. Anything else and the reply is thrown away and asked for again,
which costs the run money and teaches nobody anything.

Keep the reply small enough to finish. A skill is a short procedure; if `code` is
running long, the skill is too big, not the reply.

```json
{
  "name": "confirm_invoice_payment",
  "summary": "Confirm payment of a company's invoice from the invoice list.",
  "docstring": "Searches the invoice list for `company`, opens the matching invoice and confirms payment.\n\nAssumes: the invoice list is already on screen with its search field focused, and exactly one invoice matches.\n\nEnds on: the payment-confirmed page.",
  "params": {"company": {"type": "string", "description": "Company whose invoice to pay."}},
  "example_args": {"company": "Acme Corp"},
  "requires": [],
  "code": "def run(ctx, company):\n    ...",
  "verifier_code": "def verify(ctx, result):\n    ..."
}
```

- `name` - snake_case, a verb phrase, unique for this domain.
- `summary` - ONE line. It is what a planner reads when choosing a skill.
- `docstring` - what it does, what it ASSUMES about the starting screen, and what
  screen it ends on. The assumptions are what stop it being chosen in the wrong place.
- `params` - JSON Schema per parameter: `type` (`string`, `number`, `integer`,
  `boolean`, `array`, `object`), plus `description` and optionally `default`.
  Every parameter `run` requires must be declared here, and `run` must accept every
  parameter declared here.
- `example_args` - values that reproduce THIS recording. The skill is re-run with
  exactly these before it is admitted, so they must be the recorded ones.
- `requires` - names of other skills of this domain that your `code` calls through
  `ctx.call`; `[]` when it calls none.
- `code` - a module defining `def run(ctx, ...)` at the top level. Return something
  useful when there is something to return, otherwise `True`.
- `verifier_code` - **required**, never null: a module defining
  `def verify(ctx, result)` returning a bool. It runs after `run` and looks at the
  screen with `ctx.see` to confirm the end state was actually reached. Check for
  something that is only true when the task is DONE - the confirmation heading, the
  new row, the emptied cart - not something that was already on screen before.

## Repairs

If your skill is handed back, you are given the error and the trace of the run that
failed. Read the trace: it lists every action and every log line in order, then the
failing line. Fix THAT, return the same JSON shape again, and do not change the parts
that were working.

If instead you are told the reply could not be READ, your skill has not been judged
at all. Nothing about it is known to be wrong. Send the same answer again as the bare
JSON object; do not rewrite working code to fix a punctuation complaint.
