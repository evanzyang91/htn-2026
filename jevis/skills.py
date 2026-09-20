"""What followed a situation last time, so a repeated cycle stops re-deriving itself.

A multi-item task repeats one shape per item: type the query, submit it, reach the control, act on
it. A measured eight-item run spent 69 model calls on that — 8.6 per item — re-deciding the same
four moves. This remembers the move that followed a situation and offers it when the situation
returns.

It is deliberately not a map of the site. Page identity is unreliable on live pages: suggestions,
timestamps and re-rendered overlays make the same place look new on every visit, which is what
defeated fingerprint-keyed memory on Amazon and GitHub. A transition keyed on "where we are and
what we just did" needs no such identity.

Nothing here decides anything on its own. A remembered move is only offered when exactly one
control on the current page can be the one used before, and the executor still checks freshness,
occlusion and geometry before it runs.
"""

from urllib.parse import urlparse

OPERATIONS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}


def shape(action):
    """A control's label with its item-specific tail removed.

    "Add to cart - Robin Hood Flour" and "Add to cart - Redpath Sugar" are the same move on
    different items, and a cycle only repeats if they compare equal.
    """
    label = action.get("label", "")
    for separator in (" - ", " — ", ","):
        head = label.split(separator)[0]
        if head != label:
            label = head
            break
    return f"{action['kind']}:{label.strip()[:40]}"


class Skills:
    """Remembered next moves, keyed by the page's path and the move before them."""

    def __init__(self):
        self.moves = {}
        self.seen = {}

    @staticmethod
    def situation(url, previous):
        return urlparse(url).path, shape(previous) if previous else "start"

    def learn(self, url, previous, action):
        where, move = self.situation(url, previous), shape(action)
        # Count agreement, not occurrence. One observation is an anecdote: the first pass through a
        # page is often the exploring one, and repeating its detours faster only wastes the budget
        # sooner. A move is trusted once the run has chosen it twice from the same situation.
        self.seen[where] = self.seen.get(where, 0) + 1 if self.moves.get(where) == move else 1
        self.moves[where] = move

    def forget(self, url, previous):
        where = self.situation(url, previous)
        self.moves.pop(where, None)
        self.seen.pop(where, None)

    def recall(self, page, previous):
        """The move that followed this situation before, if the page offers exactly one candidate.

        Several matches mean the choice carries information this cannot supply — which of seven
        "Add to cart" buttons is the right item — so the decision goes back to the model.
        """
        where = self.situation(page["url"], previous)
        wanted = self.moves.get(where)
        if not wanted or self.seen.get(where, 0) < 2:
            return None
        matches = [a for a in page["actions"] if shape(a) == wanted]
        return matches[0] if len(matches) == 1 else None


def remembered_decision(action):
    """A decision record for a move taken from memory, shaped like one the model would return."""
    return {
        "choice": action["id"],
        "operation": OPERATIONS.get(action["kind"], action["kind"].upper()),
        "target": None,
        "confidence": 1.0,
        "probabilities": {action["id"]: 1.0},
        "operation_probabilities": {},
        "target_probabilities": {},
        "target_confidence": None,
        "raw_answers": {},
        "model": "remembered",
        "usage": {},
        "latency_ms": 0,
        "request": None,
        "remembered": True,
    }
