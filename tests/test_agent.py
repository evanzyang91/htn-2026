"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import Mock

import pytest

from jevis import agent as loop
from jevis import model, pricing, skills
from jevis.browser import StalePage, browser_operation, fingerprint


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner():
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.speculation = None
    a.workers = ThreadPoolExecutor(max_workers=1)
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
        "model_ms": 0,
        "load_ms": 0,
        "frame_ms": 0,
    }
    a.stale_streak, a.stale_at, a.stale_known = 0, None, 0
    a.covered = {}
    a.inert = {}
    a.taken = {}
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # One value, generated once and reused across the retry. No speculative call follows, because
    # the step that just ran was itself a fill.
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    # Regenerated because the context changed, so two generations for two differing contexts.
    assert helper.call_count == 2
    assert helper.call_args_list[1].args[0]["page"]["text"] == "Different page context"


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jevis.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jevis.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jevis.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)



@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


def steps(labels):
    return [{"action": label, "kind": "click", "page_changed": True} for label in labels]


def test_churning_between_a_few_actions_is_detected_as_a_cycle():
    # Every step changes the page, so no page_changed or fingerprint rule can see these. The real
    # Amazon loop doubles back (open, go, open, open, go), so strict alternation must not be required.
    assert loop.Agent.cycling(steps(["Open Search", "Go", "Open Search", "Open Search", "Go", "Open Search"])) == {
        "Open Search",
        "Go",
    }
    assert loop.Agent.cycling(steps(["Search", "Open Search", "Clear"] * 3)) == {"Search", "Open Search", "Clear"}


def test_progress_is_not_mistaken_for_a_cycle():
    # One action repeating is usually progress: quantity steppers, "Load more", pagination.
    assert loop.Agent.cycling(steps(["Increase quantity by 1"] * 8)) == set()
    # A varied multi-item flow must never be pruned.
    assert loop.Agent.cycling(steps(["Search", "Go", "Item A", "Add to cart", "Search", "Go", "Item B"])) == set()
    assert loop.Agent.cycling(steps(["Open Search", "Go"])) == set()  # too short to conclude


def test_scroll_that_reveals_nothing_counts_against_itself(runner):
    # A scroll always moves the offset, so page_changed can never flag a pointless one.
    scroll = {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}
    settled = page()
    settled["actions"].append(scroll)
    runner.state["page"]["actions"].append(scroll)
    runner.state["browser"].observe.return_value = settled
    for _ in range(3):
        runner.state["decision"] = decision("scroll_down")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.inert["Scroll down"] == 3
    assert all(h["revealed"] == 0 for h in runner.state["history"])


def action(kind, label, ident="e1"):
    return {"id": ident, "kind": kind, "label": label, "node": 1}


def test_shape_ignores_the_item_specific_tail():
    # The same move on different items must compare equal, or a cycle never repeats.
    assert skills.shape(action("click", "Add to cart - Robin Hood Flour")) == skills.shape(
        action("click", "Add to cart - Redpath Sugar")
    )
    assert skills.shape(action("click", "Add to cart")) != skills.shape(action("click", "Go to cart"))


def test_a_move_is_recalled_only_when_one_control_can_be_it():
    memory = skills.Skills()
    previous = action("fill", "Search")
    one = {"url": "https://shop.test/s", "actions": [action("click", "Add to cart - Sugar", "e4")]}
    memory.learn("https://shop.test/s", previous, action("click", "Add to cart - Flour", "e9"))
    # One sighting is an anecdote: the first pass through a page is often the exploring one.
    assert memory.recall(one, previous) is None
    memory.learn("https://shop.test/s", previous, action("click", "Add to cart - Salt", "e7"))
    assert memory.recall(one, previous)["id"] == "e4"
    # Seven identical add buttons carry a choice this cannot make: defer to the model.
    many = {"url": "https://shop.test/s", "actions": [
        action("click", "Add to cart - Sugar", "e4"), action("click", "Add to cart - Salt", "e5")]}
    assert memory.recall(many, previous) is None
    # A different situation is not this one.
    assert memory.recall(one, action("click", "Something else")) is None


def priced_run():
    return {
        "decisions": [{"usage": {"input_tokens": 10_000, "output_tokens": 20}}],
        "text_calls": [{"model": "gpt-5.6-luna", "usage": {"prompt_tokens": 1_000, "completion_tokens": 200}}],
        "setup_calls": [
            {"meter": "policy", "usage": {"input_tokens": 5_000, "output_tokens": 10}},
            {"meter": "writer", "model": "gpt-5.6-luna", "usage": {"prompt_tokens": 400, "completion_tokens": 100}},
        ],
        "history": [{"step": 1}, {"step": 2}],
    }


def test_the_work_done_before_the_first_action_is_still_charged_for():
    money = pricing.spend(priced_run())
    assert money["policy_tokens"] == 15_030
    assert money["writer_tokens"] == 1_700
    assert money["tokens"] == 16_730
    # Jev is input only: 15,000 read tokens at $0.042 per million.
    assert money["policy_usd"] == pytest.approx(0.00063)
    assert money["usd"] == pytest.approx(money["policy_usd"] + money["writer_usd"])


def test_a_model_without_a_published_rate_is_counted_but_not_priced():
    unknown = {"model": "moonshot", "usage": {"prompt_tokens": 900, "completion_tokens": 0}}
    run = {**priced_run(), "text_calls": [unknown]}
    money = pricing.spend(run)
    assert money["unpriced"] == ["moonshot"]
    assert money["writer_tokens"] == 1_400
    assert money["writer_usd"] == pytest.approx(0.0002)  # the priced setup call alone


def test_the_comparison_charges_the_rival_for_a_screenshot_of_every_step(monkeypatch):
    monkeypatch.setenv("RIVAL_FRAME_TOKENS", "1000")
    run = priced_run()
    ours = pricing.spend(run)
    theirs = pricing.rival(run, ours)
    # 15,000 read tokens plus 1,000 a step at $10 per million, then 30 written at $50.
    assert theirs["usd"] == pytest.approx(0.1715)
    assert theirs["times"] == pytest.approx(round(0.1715 / ours["usd"], 1))


def test_nothing_spent_means_nothing_to_compare():
    assert pricing.rival({}, pricing.spend({})) is None


def test_a_suggested_site_opens_on_its_canadian_storefront():
    assert model.localise("https://www.walmart.com") == "https://www.walmart.ca"
    assert model.localise("https://amazon.com/") == "https://amazon.ca/"
    # Already a subdomain, so it must not collect a second prefix.
    assert model.localise("https://www.indeed.com") == "https://ca.indeed.com"


def test_a_site_with_no_canadian_storefront_is_left_alone():
    assert model.localise("https://www.target.com") == "https://www.target.com"
    assert model.localise("https://en.wikipedia.org/wiki/Main_Page") == "https://en.wikipedia.org/wiki/Main_Page"


def test_another_country_takes_the_site_it_was_given(monkeypatch):
    monkeypatch.setenv("JEVIS_COUNTRY", "US")
    assert model.localise("https://www.walmart.com") == "https://www.walmart.com"
