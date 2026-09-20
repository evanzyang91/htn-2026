"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import time

import httpx

from .questions import (
    CATEGORY_RULES,
    CLASSIFY_TASK,
    NEXT_ACTION,
    REFINE_GOAL,
    TARGET,
    TASK_CATEGORIES,
    TEXT_VALUE,
)

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            detail = " ".join(response.text[:300].split())
            raise RuntimeError(
                f"{response.request.url.host} returned HTTP {response.status_code}; no action executed. {detail}"
            )
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def progress(history, window=20, limit=60):
    """Every action that changed the page, plus the recent tail.

    A plain tail loses completed work on long runs and the policy restarts finished items. Actions
    that changed nothing are only useful as immediate context, so older ones are dropped instead.
    The oldest completed work goes first past `limit`, to bound the request on a long run.
    """
    kept = [h for h in history[:-window] if h.get("page_changed")] + history[-window:]
    return [{k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in kept[-limit:]]


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            keep = ("role", "value", "checked", "selected", "expanded", "section", "opens")
            element = {k: action[k] for k in keep if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def choose(state, goal, history, covered=(), inert=(), limit=None):
    actions = state["actions"]
    if limit is not None:
        # An oversized page cannot be answered at all, so a narrowed question beats no answer.
        # Keep every control (scroll, wait, back) and the first `limit` observed elements.
        elements_first = [a for a in actions if a["kind"] in {"click", "fill", "select"}][:limit]
        actions = elements_first + [a for a in actions if a["kind"] not in {"click", "fill", "select"}]
    elements, targets, controls = action_space(actions)
    # Do not re-offer actions that observably changed nothing since the page last changed:
    # repeating them invites a dead loop (including alternating pairs), and re-executing a
    # mutation whose effect was not observed is never safe. A WAIT resets the streak — time
    # passed, so a previously dead action may work now.
    # Action ids are positional per snapshot, so a streak entry only speaks about THIS page state.
    # Targets the executor refused as covered on this same page: offering them again only repeats
    # the refusal, because nothing about the page has changed.
    streak = []
    for h in reversed(history):
        if h.get("page_changed") is not False:
            break
        if h.get("page_fingerprint") != state["fingerprint"]:
            break
        streak.append(h["choice"])
    dead = set(covered) | set(streak)
    # A control is not dead on its first miss: results really can arrive during a second WAIT, and
    # a SCROLL can reveal content the first one did not. Two misses in a row on an unchanged page
    # is a loop, not patience.
    repeated = {choice for choice in streak if streak.count(choice) >= 2}
    if dead or inert:
        for operation in list(targets):
            candidates = targets[operation]
            for index in [t for t, a in candidates.items() if a["id"] in dead or a["label"] in inert]:
                del candidates[index]
                element = elements[int(index.split(":")[0]) - 1]
                if operation in element["operations"] and not any(
                    t.split(":")[0] == index.split(":")[0] for t in candidates
                ):
                    element["operations"].remove(operation)
            if not candidates:
                del targets[operation]
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    # Controls need the same pruning as targets: scroll, wait and back are re-offered every step,
    # so one that provably changed nothing can otherwise be chosen until the budget runs out.
    operations.update(
        {key: v["label"] for key, v in controls.items() if v["id"] not in repeated and v["label"] not in inert}
    )
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded", "section", "opens") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    page = {k: state.get(k) for k in ("url", "title", "text", "loading")}
    if limit is not None:
        page["text"] = (page.get("text") or "")[:1500]
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": page,
            "elements": elements,
            # The whole run must stay visible: with multi-item goals, a short window hides completed
            # items and the policy restarts them.
            "recent_actions": progress(history, limit=20 if limit is not None else 60),
        },
        "questions": questions,
    }
    started = time.perf_counter()
    try:
        result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    except RuntimeError as error:
        # A page too large to answer is not a dead end: nothing executed, so ask a smaller question.
        # Both limits — total input size and the per-question choice cap — have the same remedy.
        oversized = "max_tokens_exceeded" in str(error) or "Too many choices" in str(error)
        smaller = len(elements) // 2 if limit is None else limit // 2
        if not oversized or smaller < 20:
            raise
        return choose(state, goal, history, covered, inert, limit=smaller)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        target_answer = validate_choice(result["answers"].get(operation.lower() + "_target", {}), targets[operation])
        target = target_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        # 2,000 chars measured ~200 ms faster per call than 6,000 with identical answers: the value
        # comes from the goal and the field, not from the bulk of the page.
        "page": {"title": page["title"], "text": page["text"][:2000], "loading": page.get("loading")},
        # The helper writes the value, so it needs the same full progress record as the policy:
        # a short window hides finished items and the helper types the first goal item again.
        "recent_actions": progress(history),
    }


EFFORTS = ("low", "medium", "high")


def text_completion(purpose, system, content, model=None, effort=None):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError(f"{purpose} needs TEXT_MODEL_API_KEY; nothing is hardcoded or guessed.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = model or os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    if "api.openai.com/" in base:
        # OpenAI chat completions rejects the OpenRouter/DeepSeek reasoning objects above; it takes
        # a flat reasoning_effort instead, and only on models that reason.
        reasoning = {"reasoning_effort": effort} if effort in EFFORTS else {}
    body = {
        "model": model,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        **reasoning,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    started = time.perf_counter()
    try:
        result = post_json(base + "/chat/completions", key, body)
    except RuntimeError as error:
        # Newer OpenAI models reject max_tokens and require max_completion_tokens. Swap on that
        # exact complaint so the model is chosen in .env, not pinned by a parameter name here.
        if "max_completion_tokens" not in str(error):
            raise
        body["max_completion_tokens"] = body.pop("max_tokens")
        result = post_json(base + "/chat/completions", key, body)
    return result, model, round((time.perf_counter() - started) * 1000)


# Five categories, so chance alone is 0.2. Below this the verb list is a guess and would push the
# wrong vocabulary onto the goal; the general rules are safer than a confident-sounding mistake.
MIN_CATEGORY_CONFIDENCE = 0.5


def classify_task(goal, url):
    """One Jev choice over task kinds. Same constrained-choice API the policy uses."""
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {"goal": goal, "site": url},
        "questions": {
            "category": {
                "type": "choice",
                "criteria": TASK_CATEGORIES,
                "instructions": {"goal": goal, "rules": CLASSIFY_TASK},
            }
        },
    }
    started = time.perf_counter()
    result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    answer = validate_choice(result["answers"].get("category", {}), TASK_CATEGORIES)
    return {
        "category": answer["choice"],
        "confidence": answer["confidence"],
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


def refine_goal(goal, url, model=None, effort=None):
    # Refinement runs once before the task, so it can afford a slower, more careful model than the
    # per-keystroke helper. Falls back to the in-loop model when REFINE_MODEL is unset.
    model = model or os.environ.get("REFINE_MODEL") or None
    effort = effort or os.environ.get("REFINE_EFFORT") or None
    try:
        info = classify_task(goal, url)
    except (RuntimeError, ValueError, KeyError, TypeError):
        # Classification only selects wording guidance; refinement must still run without it.
        info = {"category": None, "confidence": 0.0, "latency_ms": 0}
    rules = CATEGORY_RULES.get(info["category"], "") if info["confidence"] >= MIN_CATEGORY_CONFIDENCE else ""
    info["applied"] = bool(rules)
    result, _, _ = text_completion(
        "Goal refinement",
        REFINE_GOAL + ("\n" + rules if rules else ""),
        json.dumps({"goal": goal, "site": url}),
        model,
        effort,
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["goal"]
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Goal refinement returned no usable goal; retry or start with Refine off.") from None
    return value.strip(), info


class NoFieldValue(ValueError):
    """The helper deliberately declined to supply a value, answering {"text": null}.

    Distinct from a malformed answer: the helper judged that this field should not be typed into,
    so the caller should choose a different action rather than fail the run.
    """


def field_text(context):
    result, model, latency = text_completion("TYPE_TEXT", TEXT_VALUE, json.dumps(context))
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) == {"text"} and value is None:
            raise NoFieldValue("The text helper supplied no value for this field; nothing typed.")
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except NoFieldValue:
        raise
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": latency,
        "usage": result.get("usage", {}),
    }
