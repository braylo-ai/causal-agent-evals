"""Record, replay and intervene.

`record()` runs the agent live (optionally with injected faults) and returns
a Trace. `replay()` re-runs the agent from a recorded Trace under a list of
interventions and returns a new Trace.

Counterfactual rule (ceteris paribus): during replay, every recorded step
that was *unfaithful* - its recorded response differs from a live re-run of
the same request against the recorded inputs/env (a timeout, a corrupted tool
output, a sampling glitch) - is held fixed whenever the replayed agent makes
the same request again. Everything else runs live, so downstream effects of an
intervention propagate. Only the intervened factor changes.
Held tool deviations match by (tool, args) hash. Held LLM deviations match
by step position, because an upstream intervention changes the prompt text
and would otherwise silently "fix" an exogenous sampling glitch.

Intervention dicts:
  {"type": "restore", "step": i}                    run step i live (undo its deviation)
  {"type": "set_response", "step": i, "response": r} swap step i's response
  {"type": "timeout", "step": i}                    tool result -> {"error": "timeout"}
  {"type": "wrong_output", "step": i, "patch": {..}} merge fields into a tool result
  {"type": "llm_text", "step": i, "text": "..."}    swap an LLM reply
  {"type": "remove_tool", "tool": name}             ablation
  {"type": "add_tool", "tool": name}                restore a removed tool
  {"type": "drop_instruction", "id": "R1"}          ablation
  {"type": "set_instruction", "id": "R1", "text": t}
  {"type": "set_memory", "key": k, "value": v}      v=None deletes the key
  {"type": "clear_memory"}                          ablation
"""

from __future__ import annotations

import copy
import json
import uuid

from .trace import Trace, req_hash
from .world import run_agent, run_tool, simulated_llm

STEP_TYPES = {"restore", "set_response", "timeout", "wrong_output", "llm_text"}


def apply_input_interventions(inputs: dict, interventions: list[dict]) -> dict:
    inp = copy.deepcopy(inputs)
    for iv in interventions:
        t = iv["type"]
        if t == "add_tool":
            if iv["tool"] not in inp["tools"]:
                inp["tools"].append(iv["tool"])
        elif t == "remove_tool":
            inp["tools"] = [x for x in inp["tools"] if x != iv["tool"]]
        elif t == "drop_instruction":
            inp["instructions"].pop(iv["id"], None)
        elif t == "set_instruction":
            inp["instructions"][iv["id"]] = iv["text"]
        elif t == "set_memory":
            if iv["value"] is None:
                inp["memory"].pop(iv["key"], None)
            else:
                inp["memory"][iv["key"]] = iv["value"]
        elif t == "clear_memory":
            inp["memory"] = {}
    return inp


def live_response(kind: str, request: dict, inputs: dict, env: dict) -> dict:
    if kind == "llm.call":
        return simulated_llm(request["messages"], **request.get("params", {}))
    return {"result": run_tool(request["tool"], request["args"], inputs, env)}


def unfaithful_steps(trace: Trace) -> list[int]:
    """Recorded steps whose response a live re-run would not reproduce."""
    out = []
    for s in trace.steps:
        live = live_response(s["kind"], s["request"], trace.inputs, trace.env)
        a = live["text"] if s["kind"] == "llm.call" else live["result"]
        b = (
            s["response"]["text"]
            if s["kind"] == "llm.call"
            else s["response"]["result"]
        )
        if a != b:
            out.append(s["step"])
    return out


class Session:
    def __init__(self, inputs, env, patches, held):
        self.inputs, self.env = inputs, env
        self.patches = patches  # list of [kind, hash|None, position|None, iv, used]
        self.held = held  # {(kind, hash): recorded_response}
        self.step = 0
        self.steps: list[dict] = []

    def _patch_for(self, kind, h):
        for p in self.patches:
            if p[4]:
                continue
            if (p[1] is not None and p[0] == kind and p[1] == h) or (
                p[1] is None and p[2] == self.step
            ):
                p[4] = True
                return p[3]
        return None

    def _call(self, kind, request):
        h = req_hash(request)
        iv = self._patch_for(kind, h)
        held_key = ("llm.call", self.step) if kind == "llm.call" else (kind, h)
        if iv is None and held_key in self.held:
            resp = copy.deepcopy(self.held[held_key])
        else:
            resp = live_response(kind, request, self.inputs, self.env)
            if iv is not None:
                t = iv["type"]
                if t == "set_response":
                    resp = copy.deepcopy(iv["response"])
                elif t == "timeout" and kind == "tool.call":
                    resp = {"result": {"error": "timeout"}}
                elif t == "wrong_output" and kind == "tool.call":
                    resp = {"result": {**resp["result"], **iv["patch"]}}
                elif t == "llm_text" and kind == "llm.call":
                    resp = {**resp, "text": iv["text"]}
                # "restore": keep the live response
        self.steps.append(
            {"step": self.step, "kind": kind, "request": request, "response": resp}
        )
        self.step += 1
        return resp

    def llm(self, messages, **params):
        return self._call(
            "llm.call", {"messages": copy.deepcopy(messages), "params": params}
        )["text"]

    def tool(self, name, args):
        return self._call("tool.call", {"tool": name, "args": args})["result"]


def _execute(task, inputs, env, patches, held, meta) -> Trace:
    sess = Session(inputs, env, patches, held)
    final = run_agent(sess, task)
    return Trace(uuid.uuid4().hex[:12], task, inputs, env, sess.steps, final, meta)


def record(
    task: str, inputs: dict, env: dict, faults: list[dict] | None = None, meta=None
) -> Trace:
    """Run live. Step faults are applied by position (step index)."""
    patches = [
        [None, None, f["step"], f, False]
        for f in (faults or [])
        if f["type"] in STEP_TYPES
    ]
    inp = apply_input_interventions(inputs, faults or [])
    return _execute(task, inp, copy.deepcopy(env), patches, {}, dict(meta or {}))


def replay(trace: Trace, interventions: list[dict] | None = None) -> Trace:
    interventions = interventions or []
    by_step = {s["step"]: s for s in trace.steps}
    patches = []
    for iv in interventions:
        if iv["type"] in STEP_TYPES:
            s = by_step[iv["step"]]
            patches.append([s["kind"], req_hash(s["request"]), None, iv, False])
    held = {}
    restored = {iv["step"] for iv in interventions if iv["type"] in STEP_TYPES}
    for i in unfaithful_steps(trace):
        if i not in restored:
            s = by_step[i]
            key = ("llm.call", i) if s["kind"] == "llm.call" else (s["kind"], req_hash(s["request"]))
            held[key] = s["response"]
    inp = apply_input_interventions(trace.inputs, interventions)
    return _execute(
        trace.task,
        inp,
        trace.env,
        patches,
        held,
        {"parent_run": trace.run_id, "interventions": interventions},
    )


def diff_steps(a: Trace, b: Trace) -> int | None:
    """First step index where two traces diverge (None if identical)."""
    for i, (x, y) in enumerate(zip(a.steps, b.steps)):
        if json.dumps(x["request"], sort_keys=True) != json.dumps(
            y["request"], sort_keys=True
        ) or json.dumps(x["response"], sort_keys=True) != json.dumps(
            y["response"], sort_keys=True
        ):
            return i
    return None if len(a.steps) == len(b.steps) else min(len(a.steps), len(b.steps))
