"""Every action in the vocabulary round-trips through POST /api/act.

No browser here: the server owns all state, so the whole vocabulary can be
exercised exactly the way the client exercises it - one POST per action,
each returning the complete new state.
"""

import json
import urllib.error
import urllib.request

from helpers import ticket


def post(base_url, action, payload=None, seq=0):
    req = urllib.request.Request(
        base_url + "/api/act",
        data=json.dumps({"action": action, "payload": payload or {}, "seq": seq}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


def reset(base_url):
    urllib.request.urlopen(base_url + "/__reset").read()


def test_act_echoes_the_seq_and_returns_the_complete_state(base_url):
    reset(base_url)
    r = post(base_url, "board.search", {"q": "gauge"}, seq=41)
    assert r["seq"] == 41
    assert sorted(r["state"].keys()) == ["archived", "board", "meta", "ui"]
    assert r["state"]["ui"]["search"] == "gauge"


def test_unknown_action_is_a_400_with_an_error_body(base_url):
    reset(base_url)
    try:
        post(base_url, "board.doesNotExist")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        assert json.loads(exc.read()) == {"error": "unknown action: board.doesNotExist"}
    else:
        raise AssertionError("expected HTTP 400")


def test_the_full_action_vocabulary_round_trips(base_url):
    """Walk all twenty actions in one scripted session, asserting each effect."""
    reset(base_url)

    # filters
    s = post(base_url, "board.search", {"q": "log"})["state"]
    assert s["ui"]["search"] == "log"
    s = post(base_url, "board.search", {"q": ""})["state"]
    assert s["ui"]["search"] == ""
    s = post(base_url, "board.filterAssignee", {"assignee": "Mira Solano"})["state"]
    assert s["ui"]["assignee"] == "Mira Solano"
    s = post(base_url, "board.filterAssignee", {"assignee": "all"})["state"]
    assert s["ui"]["assignee"] == "all"
    s = post(base_url, "board.filterPriority", {"priority": "High"})["state"]
    assert s["ui"]["priority"] == "High"
    s = post(base_url, "board.filterPriority", {"priority": "all"})["state"]
    assert s["ui"]["priority"] == "all"

    # move menu and move
    s = post(base_url, "board.toggleMoveMenu", {"id": "t04"})["state"]
    assert s["ui"]["moveMenuFor"] == "t04"
    s = post(base_url, "board.toggleMoveMenu", {"id": "t04"})["state"]
    assert s["ui"]["moveMenuFor"] is None
    s = post(base_url, "board.move", {"id": "t04", "column": "review"})["state"]
    assert ticket(s, "t04")["column"] == "review"
    assert s["ui"]["banner"] == "Moved LAN-4 to Review"

    # detail panel: open, drafts, save title, assign, comment, close
    s = post(base_url, "board.open", {"id": "t04"})["state"]
    assert s["ui"]["openId"] == "t04"
    assert s["ui"]["draft"]["title"] == "Add bulk import for device registries"
    s = post(base_url, "board.draftTitle", {"value": "Bulk import, phase one"})["state"]
    assert s["ui"]["draft"]["title"] == "Bulk import, phase one"
    s = post(base_url, "board.saveTitle")["state"]
    assert ticket(s, "t04")["title"] == "Bulk import, phase one"
    assert s["ui"]["banner"] == "Renamed LAN-4 to Bulk import, phase one"
    s = post(base_url, "board.assign", {"assignee": "Ada Lindgren"})["state"]
    assert ticket(s, "t04")["assignee"] == "Ada Lindgren"
    s = post(base_url, "board.draftComment", {"value": "Checked with the vendor."})["state"]
    assert s["ui"]["draft"]["comment"] == "Checked with the vendor."
    s = post(base_url, "board.addComment")["state"]
    assert ticket(s, "t04")["comments"][-1]["text"] == "Checked with the vendor."
    assert ticket(s, "t04")["comments"][-1]["id"] == "c07"
    assert s["ui"]["draft"]["comment"] == ""
    s = post(base_url, "board.close")["state"]
    assert s["ui"]["openId"] is None

    # compose: open, fields, close, reopen, create
    s = post(base_url, "board.composeOpen")["state"]
    assert s["ui"]["compose"]["open"] is True
    s = post(base_url, "board.composeField", {"field": "title", "value": "Throwaway"})["state"]
    assert s["ui"]["compose"]["title"] == "Throwaway"
    s = post(base_url, "board.composeClose")["state"]
    assert s["ui"]["compose"] == {
        "open": False, "title": "", "description": "",
        "assignee": "Mira Solano", "priority": "Medium",
    }
    post(base_url, "board.composeOpen")
    post(base_url, "board.composeField", {"field": "title", "value": "Ship the export"})
    post(base_url, "board.composeField", {"field": "description", "value": "CSV first."})
    post(base_url, "board.composeField", {"field": "assignee", "value": "Theo Barros"})
    s = post(base_url, "board.composeField", {"field": "priority", "value": "High"})["state"]
    assert s["ui"]["compose"]["priority"] == "High"
    s = post(base_url, "board.create")["state"]
    new = ticket(s, "t15")
    assert (new["key"], new["title"], new["assignee"], new["priority"], new["column"]) == \
        ("LAN-15", "Ship the export", "Theo Barros", "High", "backlog")
    assert s["ui"]["banner"] == "Created LAN-15 in Backlog"

    # archive dialog: open, cancel, open, confirm
    s = post(base_url, "board.archiveDialog")["state"]
    assert s["ui"]["dialogOpen"] is True
    s = post(base_url, "board.archiveCancel")["state"]
    assert s["ui"]["dialogOpen"] is False
    assert s["archived"] == []
    post(base_url, "board.archiveDialog")
    s = post(base_url, "board.archiveConfirm")["state"]
    assert sorted(t["id"] for t in s["archived"]) == ["t06", "t12", "t14"]
    assert not any(t["column"] == "done" for t in s["board"]["tickets"])
    assert s["ui"]["banner"] == "Archived 3 tickets"

    # banner
    s = post(base_url, "board.dismissBanner")["state"]
    assert s["ui"]["banner"] is None


def test_guarded_actions_do_not_mutate_blindly(base_url):
    reset(base_url)
    # a move to the ticket's own column is a no-op with no banner
    s = post(base_url, "board.move", {"id": "t01", "column": "backlog"})["state"]
    assert ticket(s, "t01")["column"] == "backlog"
    assert s["ui"]["banner"] is None
    # an empty title cannot be saved
    post(base_url, "board.open", {"id": "t01"})
    post(base_url, "board.draftTitle", {"value": "   "})
    s = post(base_url, "board.saveTitle")["state"]
    assert ticket(s, "t01")["title"] == "Set up nightly export of usage metrics"
    # an empty comment is not added
    post(base_url, "board.draftComment", {"value": ""})
    s = post(base_url, "board.addComment")["state"]
    assert ticket(s, "t01")["comments"] == []
    # create without a title does nothing
    post(base_url, "board.composeOpen")
    s = post(base_url, "board.create")["state"]
    assert len(s["board"]["tickets"]) == 14
    # an unknown assignee is rejected
    s = post(base_url, "board.assign", {"assignee": "Nobody Real"})["state"]
    assert ticket(s, "t01")["assignee"] == "Mira Solano"
    # archive with an empty Done column never opens the dialog
    post(base_url, "board.archiveDialog")
    post(base_url, "board.archiveConfirm")  # archives the three seed Done tickets
    s = post(base_url, "board.archiveDialog")["state"]
    assert s["ui"]["dialogOpen"] is False
