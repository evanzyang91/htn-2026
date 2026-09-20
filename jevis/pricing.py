"""What a run cost, in tokens and in money.

Two meters run at once: the policy charges for what it reads, the writer charges for both halves.
Rates are per million tokens, published by each provider. A model with no published rate still has
its tokens counted, and the money it contributed is reported as unpriced rather than guessed at.

Override or extend with TOKEN_RATES in the environment: "model:input:output,model:input:output".
"""

import os

MILLION = 1_000_000

# Jev is charged on input alone; output is free.
JEVIS_RATES = (0.042, 0.0)

TEXT_RATES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (4.00, 20.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4-nano": (0.10, 0.40),
}


def rates(model):
    """Published price per million input and output tokens, or None if the model has no rate."""
    for pair in os.environ.get("TOKEN_RATES", "").split(","):
        name, _, prices = pair.partition(":")
        if name.strip() == model and prices.count(":") == 1:
            try:
                given, produced = prices.split(":")
                return float(given), float(produced)
            except ValueError:
                break
    return TEXT_RATES.get(model)


def spend(state):
    """Tokens and money for one run, split by the two models that charge for it."""
    # Choosing the site, sorting the task and rewriting the goal all charge before the first
    # action, so a total that ignored them would understate every run.
    setup = state.get("setup_calls") or []
    policy = [*(d.get("usage") or {} for d in state.get("decisions", [])),
              *(c.get("usage") or {} for c in setup if c.get("meter") == "policy")]
    policy_in = sum(u.get("input_tokens", 0) for u in policy)
    policy_out = sum(u.get("output_tokens", 0) for u in policy)
    policy_cost = policy_in * JEVIS_RATES[0] / MILLION + policy_out * JEVIS_RATES[1] / MILLION

    writer_in = writer_out = writer_cost = 0
    unpriced = set()
    for call in [*state.get("text_calls", []), *(c for c in setup if c.get("meter") == "writer")]:
        usage = call.get("usage") or {}
        given = usage.get("prompt_tokens", 0)
        produced = usage.get("completion_tokens", 0)
        writer_in += given
        writer_out += produced
        price = rates(call.get("model", ""))
        if price:
            writer_cost += given * price[0] / MILLION + produced * price[1] / MILLION
        elif given or produced:
            unpriced.add(call.get("model", "unknown"))

    return {
        "policy_tokens": policy_in + policy_out,
        "writer_tokens": writer_in + writer_out,
        "tokens": policy_in + policy_out + writer_in + writer_out,
        "policy_usd": round(policy_cost, 6),
        "writer_usd": round(writer_cost, 6),
        "usd": round(policy_cost + writer_cost, 6),
        "unpriced": sorted(unpriced),
    }
