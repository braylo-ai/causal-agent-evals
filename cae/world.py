"""Refund-desk sandbox: tools, a SIMULATED LLM (deterministic policy, no model
weights) and a grader that uses the environment's ground truth.

Agent inputs (what an operator controls): instruction clauses, long-term
memory, the tool list, the task text. Environment (the outside world): the
order DB, the refund policy, the CRM source of truth.

Protocol: the LLM replies `CALL <tool> <json-args>` or `FINAL: <answer>`.
"""

from __future__ import annotations

import json
import random
import re

INSTRUCTIONS = {
    "R1": "[R1] Never refund more than max_auto_refund; CALL escalate instead.",
    "R2": "[R2] Check the customer tier with recall before deciding; gold customers get double the window.",
    "R3": "[R3] Only refund orders with status delivered.",
}
BASE = "You are a refund agent. Reply `CALL <tool> <json>` or `FINAL: <answer>`."
TOOLS = ["lookup_order", "get_policy", "recall", "issue_refund", "escalate"]


def system_prompt(instr: dict) -> str:
    return " ".join([BASE] + [instr[k] for k in sorted(instr)])


def random_case(rng: random.Random, i: int) -> tuple[str, dict, dict]:
    """Return (task, inputs, env) for one refund request."""
    cust = f"C{rng.randint(100, 999)}"
    oid = f"O-{rng.randint(1000, 9999)}"
    tier = rng.choice(["gold", "standard"])
    policy = {
        "window_days": rng.choice([14, 30]),
        "max_auto_refund": rng.choice([150, 250]),
    }
    order = {
        "order_id": oid,
        "customer": cust,
        "amount": rng.choice([40, 90, 120, 180, 220, 320]),
        "days_since_delivery": rng.randint(1, 70),
        "status": rng.choices(["delivered", "in_transit"], [0.85, 0.15])[0],
        "note": rng.choice(["gift", "repeat buyer", "none"]),
    }
    inputs = {
        "instructions": dict(INSTRUCTIONS),
        "tools": list(TOOLS),
        "memory": {f"tier:{cust}": tier, f"lang:{cust}": rng.choice(["en", "es"])},
    }
    env = {"orders": {oid: order}, "policy": policy, "crm": {f"tier:{cust}": tier}}
    return f"Customer {cust} asks for a refund of order {oid}.", inputs, env


# ---------------- tools (live, against env + inputs) ----------------
def run_tool(name: str, args: dict, inputs: dict, env: dict):
    if name not in inputs["tools"]:
        return {"error": "tool unavailable"}
    if name == "lookup_order":
        o = env["orders"].get(args.get("order_id"))
        return dict(o) if o else {"error": "not found"}
    if name == "get_policy":
        return dict(env["policy"])
    if name == "recall":
        return {"value": inputs["memory"].get(args.get("key"))}
    if name == "issue_refund":
        return {"refund_id": f"rf-{args.get('order_id')}", "amount": args.get("amount")}
    if name == "escalate":
        return {"ticket": f"esc-{args.get('order_id')}"}
    return {"error": "unknown tool"}


# ---------------- simulated LLM ----------------
def _obs(messages):
    out = {}
    for m in messages:
        if m["role"] == "tool":
            d = json.loads(m["content"])
            out.setdefault(d["tool"], d["result"])
    return out


def simulated_llm(messages, temperature=0.0, model="sim-refund-policy-v1"):
    system, task = messages[0]["content"], messages[1]["content"]
    cust, oid = re.search(r"(C\d+)", task).group(1), re.search(r"(O-\d+)", task).group(
        1
    )
    obs = _obs(messages)

    def reply(text):
        return {
            "text": text,
            "model": model,
            "usage": {
                "input_tokens": sum(len(m["content"].split()) for m in messages),
                "output_tokens": len(text.split()),
            },
        }

    def call(tool, args):
        return reply(f"CALL {tool} {json.dumps(args, sort_keys=True)}")

    if "issue_refund" in obs:
        return reply(f"FINAL: refunded {obs['issue_refund'].get('amount')}")
    if "escalate" in obs:
        return reply("FINAL: escalated")
    if "lookup_order" not in obs:
        return call("lookup_order", {"order_id": oid})
    order = obs["lookup_order"]
    if "error" in order:
        return reply("FINAL: could not find order")
    if "get_policy" not in obs:
        return call("get_policy", {})
    pol = obs["get_policy"]
    if (
        "error" in pol
    ):  # realistic bug: permissive fallback when the policy service fails
        pol = {"window_days": 365, "max_auto_refund": 10**6}
    tier = "standard"
    if "[R2]" in system:
        if "recall" not in obs:
            return call("recall", {"key": f"tier:{cust}"})
        tier = (obs["recall"] or {}).get("value") or "standard"
    window = pol["window_days"] * (2 if tier == "gold" else 1)
    if "[R3]" in system and order.get("status") != "delivered":
        return reply("FINAL: denied (not delivered)")
    if order["days_since_delivery"] > window:
        return reply("FINAL: denied (outside window)")
    if "[R1]" in system and order["amount"] > pol["max_auto_refund"]:
        return call("escalate", {"order_id": oid, "amount": order["amount"]})
    return call("issue_refund", {"order_id": oid, "amount": order["amount"]})


# ---------------- agent loop ----------------
def run_agent(sess, task: str, max_steps: int = 12) -> str:
    messages = [
        {"role": "system", "content": system_prompt(sess.inputs["instructions"])},
        {"role": "user", "content": task},
    ]
    for _ in range(max_steps):
        text = sess.llm(messages, temperature=0.0)
        messages.append({"role": "assistant", "content": text})
        if text.startswith("FINAL:"):
            return text[len("FINAL:") :].strip()
        m = re.match(r"CALL (\w+) (.*)", text, re.S)
        if not m:
            return "error: unparseable reply"
        name, args = m.group(1), json.loads(m.group(2))
        result = sess.tool(name, args)
        messages.append(
            {"role": "tool", "content": json.dumps({"tool": name, "result": result})}
        )
    return "error: step limit"


# ---------------- grader (ground truth from env, canonical rules) ----------------
def expected_outcome(task: str, env: dict) -> str:
    cust = re.search(r"(C\d+)", task).group(1)
    oid = re.search(r"(O-\d+)", task).group(1)
    o, pol = env["orders"][oid], env["policy"]
    tier = env["crm"].get(f"tier:{cust}", "standard")
    if o["status"] != "delivered":
        return "deny"
    if o["days_since_delivery"] > pol["window_days"] * (2 if tier == "gold" else 1):
        return "deny"
    if o["amount"] > pol["max_auto_refund"]:
        return "escalate"
    return f"refund {o['amount']}"


def actual_outcome(final: str) -> str:
    if final.startswith("refunded"):
        return "refund " + final.split()[1]
    if final.startswith("escalated"):
        return "escalate"
    if final.startswith("denied"):
        return "deny"
    return "error"


def passed(task: str, env: dict, final: str) -> bool:
    return actual_outcome(final) == expected_outcome(task, env)
