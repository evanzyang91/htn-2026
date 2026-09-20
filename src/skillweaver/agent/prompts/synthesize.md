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

What that settle does NOT cover is a control the page answers WITHOUT navigating - an
"Add to cart" that fires a background request, a filter that swaps a list in place,
anything that redirects a moment later. The load event those are settled against fired
long before they were clicked. `ctx.wait_for_text` below is for exactly that, and rule
11 is when to reach for it.

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
ctx.wait_for_text("Subtotal")  # find_text, allowed to look again while the page answers
ctx.wait_for_text("Subtotal", "button")  # a kind filter, exactly like find_text
ctx.log("what just happened")  # one line into the run trace
ctx.call("other_skill", arg=1)  # run another skill of the same domain
ctx.graph.neighbors(fingerprint)  # read-only site graph; rarely needed
```

`ctx.wait_for_text` returns a LIST, best first, and an EMPTY list when the text never
appeared - so you check it exactly like a `find_text`, and `ctx.expect` is what turns
"it never arrived" into an honest failure. It matches on containment only, never
fuzzily, because a near match answers on the screen you were waiting to leave.

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
11. **Wait for a THING, not for a TIME.** Rule 10 is not "never wait"; it is "never
    wait a duration". When you click something the page answers WITHOUT navigating -
    "Add to cart", "Apply filter", a control that redirects a moment later - the
    settle does not cover it, and the very next `ctx.see` reads the page you are still
    standing on. Measured on a live shop: the click returned in 130ms with the old
    page complete, and the page it leads to did not commit until 1170ms.

    ```python
    # NO - reads the page the click has not finished answering yet
    ctx.ctl.click(add_to_cart[0])
    cart = ctx.see.find_text("Subtotal")
    ctx.expect(bool(cart), "the cart never appeared")

    # YES - looks again until it is there, and returns the instant it is
    ctx.ctl.click(add_to_cart[0])
    cart = ctx.wait_for_text("Subtotal")
    ctx.expect(bool(cart), "the cart never appeared")
    ```

    This is not the sleep rule 10 removes and it is not charged like one: on a page
    that has already answered, the first look IS the observation you were about to
    make, so it costs nothing. Name text that **only the answered screen says**. "Add
    to cart" is on the page you are leaving, so waiting for it returns at once and
    proves nothing; "Subtotal", "Your cart", the confirmation heading, are the cart's
    own words. The hardening pass writes this rewrite in for you where your code makes
    a read and then `ctx.expect`s it, or hands its result to `ctx.ctl`, so write it
    yourself and it stays as you wrote it - and a read you only branch on is left
    alone, which is right, because you are asking what is on screen rather than
    waiting for something to arrive.

    A page also answers in PHASES, and the same rule covers it: a search result's
    title is on screen before its button is. Measured on a live shop, the title was
    there at 2.32s and the "Add to cart" beside it 0.61s later. So waiting for the
    title does not make the button safe to `find_text` - wait for THE THING YOU ARE
    ABOUT TO PRESS. And when a named lookup for something you are about to press comes
    back empty, do NOT fall back to `ctx.see.best`: it ranks what IS on screen and has
    a winner even when nothing fits, so it answers with the wrong control - on that
    shop, the header's cart button - and your skill presses it. An empty lookup for a
    press target is `ctx.expect`'s job.

    ```python
    # NO - the button has not arrived, and the fallback presses something else
    adds = ctx.see.find_text("Add to cart - " + product, "button")
    if not adds:
        adds = ctx.see.best("Add to cart button for " + product)
    ctx.ctl.click(adds[0])

    # YES
    adds = ctx.wait_for_text("Add to cart - " + product, "button")
    ctx.expect(bool(adds), "no Add to cart control for that product")
    ctx.ctl.click(adds[0])
    ```

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
  screen with `ctx.see` to confirm the end state was actually reached. See the rule
  below - it is checked, and it is the most common reason a skill is handed back.

## Your verifier must be able to FAIL

A verifier is only worth storing if it can say NO to a screen your skill might
plausibly land on - above all the screen it STARTS on. After your skill runs, your
verifier is re-run against that starting screen. **If it passes there too, your skill
is rejected**, because a check that is true before and after proves nothing: it would
say yes to a run that did nothing at all.

What usually goes wrong is SITE CHROME. The top navigation, the footer, the logo, a
category or section word the site puts on every one of its pages - all of those are
on your end screen, so matching one looks like a check and is not one. A real skill
was once given `ctx.see.find_text("Keycaps")` on a keyboard shop whose top nav says
"Keycaps" on every page; it passed eight replays of an empty cart.

So key on something that CHANGED because your skill ran, in this order of preference:

1. **A count or a quantity that moved.** A cart badge, "3 items", a result count, a
   total or a price that only appears once there is something to total.
2. **A row, card or line that is NEWLY present and names what your parameters
   chose** - the article you searched for, the product you added. Prefer the
   parameter's own value over a fixed string; it is what makes the check specific to
   this run rather than to this site.
3. **A URL that differs from the starting one**, when it is readable on screen.
4. **Text that exists ONLY in the finished state** - a confirmation heading, an
   "added to your cart" line, an empty-state message that has now gone away.

`verify` receives only `ctx` and `result`, so a verifier that needs a parameter's
value has to be handed it: **return it from `run`** and read it off `result`. That is
what `result` is for.

```python
# NO - "Keycaps" is in the top nav of every page, including the one we started on
def verify(ctx, result):
    return bool(ctx.see.find_text("Keycaps"))


# YES - run hands the verifier what it chose...
def run(ctx, product):
    ...
    return product


# ...and the check is then specific to THIS run: the product name is in the cart
# only once it has been added, and the cart shows a subtotal only when it holds
# something
def verify(ctx, result):
    named = ctx.see.find_text(str(result))
    subtotal = ctx.see.find_text("Subtotal")
    return bool(named) and bool(subtotal)
```

Before you send it, read your verifier against the recorded FIRST screen above: if
every string it looks for was already there, rewrite it.

## Repairs

If your skill is handed back, you are given the error and the trace of the run that
failed. Read the trace: it lists every action and every log line in order, then the
failing line. Fix THAT, return the same JSON shape again, and do not change the parts
that were working.

If instead you are told the reply could not be READ, your skill has not been judged
at all. Nothing about it is known to be wrong. Send the same answer again as the bare
JSON object; do not rewrite working code to fix a punctuation complaint.
