"""Failure attribution: which step or input caused this run to fail?

Candidates are every factor that deviates from a reference:
  * input:instruction:<id>  - clause missing or changed vs the reference instructions
  * input:memory:<key>      - memory entry differs from the reference memory
  * input:tool:<name>       - tool missing vs the reference tool list
  * step:<i>                - recorded step that a live re-run would not reproduce
                              (timeout, corrupted tool output, LLM glitch)

`counterfactual_attribution` restores one candidate at a time, replays the
run with every other deviation held fixed, and blames the candidate(s) whose
restoration makes the run pass. If no single restore flips the outcome it
tries pairs. Baselines (no replay) are included for comparison.
"""

from __future__ import annotations

from itertools import combinations

from .replay import replay, unfaithful_steps
from .trace import Trace
from .world import passed


def candidates(trace: Trace, reference_inputs: dict) -> list[tuple[str, dict]]:
    out = []
    ri, ti = reference_inputs, trace.inputs
    for k in sorted(ri["instructions"]):
        if ti["instructions"].get(k) != ri["instructions"][k]:
            out.append(
                (
                    f"input:instruction:{k}",
                    {"type": "set_instruction", "id": k, "text": ri["instructions"][k]},
                )
            )
    for k in sorted(set(ri["memory"]) | set(ti["memory"])):
        if ti["memory"].get(k) != ri["memory"].get(k):
            out.append(
                (
                    f"input:memory:{k}",
                    {"type": "set_memory", "key": k, "value": ri["memory"].get(k)},
                )
            )
    removed = [t for t in ri["tools"] if t not in ti["tools"]]
    for t in removed:
        out.append((f"input:tool:{t}", {"type": "add_tool", "tool": t}))
    for i in unfaithful_steps(trace):
        out.append((f"step:{i}", {"type": "restore", "step": i}))
    return out


def _replay_with(trace, ivs):
    # "set_tools" restores a removed tool; expressed via a patched trace copy
    adds = [iv["tool"] for iv in ivs if iv["type"] == "set_tools"]
    if adds:
        trace = Trace(
            trace.run_id,
            trace.task,
            {**trace.inputs, "tools": trace.inputs["tools"] + adds},
            trace.env,
            trace.steps,
            trace.final,
            trace.meta,
        )
    return replay(trace, [iv for iv in ivs if iv["type"] != "set_tools"])


def counterfactual_attribution(
    trace: Trace, reference_inputs: dict, max_pairs: int = 50
) -> dict:
    cands = candidates(trace, reference_inputs)
    replays, flips = 0, []
    for cid, iv in cands:
        replays += 1
        if passed(trace.task, trace.env, _replay_with(trace, [iv]).final):
            flips.append(cid)
    if flips:
        return {
            "blame": flips[0],
            "causal_set": flips,
            "replays": replays,
            "candidates": [c for c, _ in cands],
        }
    for (a, ia), (b, ib) in list(combinations(cands, 2))[:max_pairs]:
        replays += 1
        if passed(trace.task, trace.env, _replay_with(trace, [ia, ib]).final):
            return {
                "blame": "+".join(sorted([a, b])),
                "causal_set": sorted([a, b]),
                "replays": replays,
                "candidates": [c for c, _ in cands],
            }
    return {
        "blame": "unknown",
        "causal_set": [],
        "replays": replays,
        "candidates": [c for c, _ in cands],
    }


# ---------------- baselines (no replay) ----------------
def last_tool_step(trace: Trace, reference_inputs: dict) -> str:
    tools = [s["step"] for s in trace.steps if s["kind"] == "tool.call"]
    return f"step:{tools[-1]}" if tools else "unknown"


def first_error(trace: Trace, reference_inputs: dict) -> str:
    for s in trace.steps:
        r = s["response"].get("result") if s["kind"] == "tool.call" else None
        if isinstance(r, dict) and "error" in r:
            return f"step:{s['step']}"
    return last_tool_step(trace, reference_inputs)


def first_anomaly(trace: Trace, reference_inputs: dict) -> str:
    c = candidates(trace, reference_inputs)
    return c[0][0] if c else "unknown"


BASELINES = {
    "last_tool_step": last_tool_step,
    "first_error": first_error,
    "first_anomaly": first_anomaly,
}
