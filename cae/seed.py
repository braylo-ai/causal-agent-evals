"""Seeded failures with known root causes (SYNTHETIC).

For each case: record a clean run (must pass), then inject ONE causal fault,
optionally plus one non-causal decoy deviation. A seed is kept only if
  * the fault alone makes the run fail,
  * the decoy alone (if any) still passes, and
  * fault + decoy fails.
The ground-truth blame is the candidate id of the causal fault.
"""

from __future__ import annotations

import copy
import random

from .replay import record
from .world import passed, random_case

FAULT_TYPES = [
    "wrong_output:lookup_order",
    "wrong_output:get_policy",
    "wrong_output:recall",
    "timeout:get_policy",
    "timeout:recall",
    "drop_instruction",
    "corrupt_memory",
    "remove_tool:recall",
    "llm_wrong_arg",
]


def _tool_pos(trace, name):
    for s in trace.steps:
        if s["kind"] == "tool.call" and s["request"]["tool"] == name:
            return s["step"]
    return None


def _make_fault(ftype, rng, clean, inputs, task):
    """Return (fault_dicts, ground_truth_id) or None if not applicable."""
    cust = task.split()[1]
    if ftype.startswith("wrong_output:") or ftype.startswith("timeout:"):
        kind, tool = ftype.split(":")
        pos = _tool_pos(clean, tool)
        if pos is None:
            return None
        if kind == "timeout":
            return [{"type": "timeout", "step": pos}], f"step:{pos}"
        patch = {
            "lookup_order": rng.choice(
                [
                    {"days_since_delivery": 200},
                    {"days_since_delivery": 1},
                    {"amount": 500},
                    {"amount": 30},
                    {"status": "in_transit"},
                    {"status": "delivered"},
                ]
            ),
            "get_policy": rng.choice(
                [
                    {"window_days": 3},
                    {"window_days": 120},
                    {"max_auto_refund": 20},
                    {"max_auto_refund": 10000},
                ]
            ),
            "recall": rng.choice([{"value": "gold"}, {"value": "standard"}]),
        }[tool]
        return [{"type": "wrong_output", "step": pos, "patch": patch}], f"step:{pos}"
    if ftype == "drop_instruction":
        rid = rng.choice(["R1", "R2", "R3"])
        return [{"type": "drop_instruction", "id": rid}], f"input:instruction:{rid}"
    if ftype == "corrupt_memory":
        k = f"tier:{cust}"
        return [
            {
                "type": "set_memory",
                "key": k,
                "value": "standard" if inputs["memory"][k] == "gold" else "gold",
            }
        ], f"input:memory:{k}"
    if ftype == "remove_tool:recall":
        return [{"type": "remove_tool", "tool": "recall"}], "input:tool:recall"
    if ftype == "llm_wrong_arg":
        for s in clean.steps:
            if s["kind"] == "llm.call" and s["response"]["text"].startswith(
                "CALL issue_refund"
            ):
                txt = s["response"]["text"].replace('"amount": ', '"amount": 1')
                return [
                    {"type": "llm_text", "step": s["step"], "text": txt}
                ], f"step:{s['step']}"
        return None
    raise ValueError(ftype)


def _make_decoy(rng, clean, inputs, task, avoid_step):
    cust = task.split()[1]
    opts = []
    p = _tool_pos(clean, "lookup_order")
    if p is not None and p != avoid_step:
        opts.append(
            (
                [{"type": "wrong_output", "step": p, "patch": {"note": "VIP!!"}}],
                f"step:{p}",
            )
        )
    p = _tool_pos(clean, "get_policy")
    if p is not None and p != avoid_step:
        opts.append(
            (
                [{"type": "wrong_output", "step": p, "patch": {"updated": "never"}}],
                f"step:{p}",
            )
        )
    k = f"lang:{cust}"
    opts.append(
        ([{"type": "set_memory", "key": k, "value": "fr"}], f"input:memory:{k}")
    )
    return rng.choice(opts)


def seeded_failures(n: int, seed: int = 0, decoy_rate: float = 0.5) -> list[dict]:
    rng = random.Random(seed)
    out, i = [], 0
    while len(out) < n:
        ftype = FAULT_TYPES[len(out) % len(FAULT_TYPES)]
        task, inputs, env = random_case(rng, i)
        i += 1
        clean = record(task, inputs, env)
        if not passed(task, env, clean.final):
            continue
        made = _make_fault(ftype, rng, clean, inputs, task)
        if made is None:
            continue
        faults, gt = made
        alone = record(task, inputs, env, faults)
        if passed(task, env, alone.final):
            continue
        decoy = None
        if rng.random() < decoy_rate:
            avoid = faults[0].get("step")
            dfaults, did = _make_decoy(rng, clean, inputs, task, avoid)
            if passed(task, env, record(task, inputs, env, dfaults).final):
                decoy = did
                faults = faults + dfaults
        trace = record(task, inputs, env, faults, meta={"seeded": True})
        if passed(task, env, trace.final):
            continue
        out.append(
            {
                "trace": trace,
                "reference_inputs": copy.deepcopy(inputs),
                "fault_type": ftype,
                "ground_truth": gt,
                "decoy": decoy,
                "faults": faults,
            }
        )
    return out


def overdetermined_failures(n: int, seed: int = 0) -> list[dict]:
    """Two independent faults, EACH sufficient to fail the run on its own.
    No single restore can fix the run; ground truth is the pair. Draws where one
    fault masks the other (so it is not visible in the trace) are rejected."""
    rng = random.Random(seed)
    out, i = [], 0
    while len(out) < n and i < n * 200:
        i += 1
        task, inputs, env = random_case(rng, i)
        clean = record(task, inputs, env)
        if not passed(task, env, clean.final):
            continue
        fa, fb = rng.sample(FAULT_TYPES, 2)
        ma, mb = _make_fault(fa, rng, clean, inputs, task), _make_fault(fb, rng, clean, inputs, task)
        if not ma or not mb or ma[1] == mb[1]:
            continue
        if passed(task, env, record(task, inputs, env, ma[0]).final) or \
                passed(task, env, record(task, inputs, env, mb[0]).final):
            continue
        trace = record(task, inputs, env, ma[0] + mb[0], meta={"seeded": True})
        if passed(task, env, trace.final):
            continue
        # validity: both faults must remain observable deviations in the combined
        # trace (one fault can mask the other, e.g. dropping R2 removes the recall step)
        from .attribution import candidates
        present = {c for c, _ in candidates(trace, inputs)}
        if not {ma[1], mb[1]} <= present:
            continue
        out.append({"trace": trace, "reference_inputs": copy.deepcopy(inputs),
                    "fault_type": f"{fa} & {fb}", "ground_truth": "+".join(sorted([ma[1], mb[1]])),
                    "decoy": None, "faults": ma[0] + mb[0]})
    return out
