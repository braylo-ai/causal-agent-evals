"""Benchmarks (all SYNTHETIC: simulated LLM policy + sandbox tools).

1. Attribution accuracy on seeded failures, counterfactual vs baselines.
2. Ablation impact on clean runs: remove a tool / instruction / memory.
3. Fault-injection sweep on clean runs: timeout or empty output per tool.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .attribution import BASELINES, counterfactual_attribution
from .replay import record, replay
from .seed import overdetermined_failures, seeded_failures
from .world import INSTRUCTIONS, TOOLS, passed, random_case


def attribution_bench(n: int, seed: int, cases=None) -> dict:
    cases = cases if cases is not None else seeded_failures(n, seed)
    methods = ["counterfactual"] + list(BASELINES)
    hits = {m: 0 for m in methods}
    by_type = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_decoy = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    replays, rows = [], []
    for c in cases:
        cf = counterfactual_attribution(c["trace"], c["reference_inputs"])
        preds = {"counterfactual": cf["blame"]}
        preds.update(
            {m: f(c["trace"], c["reference_inputs"]) for m, f in BASELINES.items()}
        )
        replays.append(cf["replays"])
        d = "with_decoy" if c["decoy"] else "no_decoy"
        for m in methods:
            ok = preds[m] == c["ground_truth"]
            hits[m] += ok
            by_type[m][c["fault_type"]][0] += ok
            by_type[m][c["fault_type"]][1] += 1
            by_decoy[m][d][0] += ok
            by_decoy[m][d][1] += 1
        rows.append(
            {
                "fault_type": c["fault_type"],
                "ground_truth": c["ground_truth"],
                "decoy": c["decoy"],
                "predictions": preds,
                "causal_set": cf["causal_set"],
                "replays": cf["replays"],
            }
        )
    return {
        "n_seeded_failures": len(cases),
        "seed": seed,
        "accuracy": {m: round(hits[m] / len(cases), 3) for m in methods},
        "hits": hits,
        "accuracy_by_fault_type": {
            m: {k: f"{a}/{b}" for k, (a, b) in sorted(v.items())}
            for m, v in by_type.items()
        },
        "accuracy_by_decoy": {
            m: {k: f"{a}/{b}" for k, (a, b) in sorted(v.items())}
            for m, v in by_decoy.items()
        },
        "counterfactual_replays": {
            "mean": round(sum(replays) / len(replays), 2),
            "max": max(replays),
        },
        "cases": rows,
    }


def clean_cases(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        task, inputs, env = random_case(rng, len(out))
        tr = record(task, inputs, env)
        if passed(task, env, tr.final):
            out.append(tr)
    return out


def ablation_bench(traces) -> dict:
    ablations = (
        [("remove_tool:" + t, {"type": "remove_tool", "tool": t}) for t in TOOLS]
        + [
            ("drop_instruction:" + k, {"type": "drop_instruction", "id": k})
            for k in INSTRUCTIONS
        ]
        + [("clear_memory", {"type": "clear_memory"})]
    )
    res = {}
    for name, iv in ablations:
        fails = sum(not passed(t.task, t.env, replay(t, [iv]).final) for t in traces)
        res[name] = {
            "fail_rate": round(fails / len(traces), 3),
            "fails": fails,
            "n": len(traces),
        }
    return res


def fault_sweep(traces) -> dict:
    res = defaultdict(lambda: {"injected": 0, "fails": 0})
    for t in traces:
        for s in t.steps:
            if s["kind"] != "tool.call":
                continue
            tool = s["request"]["tool"]
            for fname, iv in (
                ("timeout", {"type": "timeout", "step": s["step"]}),
                (
                    "empty_output",
                    {
                        "type": "set_response",
                        "step": s["step"],
                        "response": {"result": {}},
                    },
                ),
            ):
                key = f"{fname}:{tool}"
                res[key]["injected"] += 1
                try:
                    ok = passed(t.task, t.env, replay(t, [iv]).final)
                except (KeyError, TypeError):
                    ok = False  # agent crashed on the malformed output
                res[key]["fails"] += not ok
    return {
        k: {**v, "fail_rate": round(v["fails"] / v["injected"], 3)}
        for k, v in sorted(res.items())
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--heldout-seed", type=int, default=2026)
    ap.add_argument("--n-over", type=int, default=40)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(exist_ok=True)
    attr = attribution_bench(a.n, a.seed)
    clean = clean_cases(a.clean, a.seed + 1)
    abl, sweep = ablation_bench(clean), fault_sweep(clean)
    note = "SYNTHETIC: simulated LLM policy (no model weights) + sandbox tools"
    Path(a.out, "attribution.json").write_text(
        json.dumps({"data": note, **attr}, indent=2) + "\n"
    )
    Path(a.out, "ablations.json").write_text(
        json.dumps(
            {"data": note, "n_clean_runs": len(clean), "ablations": abl}, indent=2
        )
        + "\n"
    )
    Path(a.out, "fault_sweep.json").write_text(
        json.dumps(
            {"data": note, "n_clean_runs": len(clean), "faults": sweep}, indent=2
        )
        + "\n"
    )
    held = attribution_bench(a.n, a.heldout_seed)
    Path(a.out, "attribution_heldout.json").write_text(
        json.dumps({"data": note, **held}, indent=2) + "\n"
    )
    over = attribution_bench(
        0, a.seed, cases=overdetermined_failures(a.n_over, a.seed + 2)
    )
    Path(a.out, "attribution_overdetermined.json").write_text(
        json.dumps({"data": note, **over}, indent=2) + "\n"
    )
    summary = {
        "single_cause": attr["accuracy"],
        "single_cause_heldout": held["accuracy"],
        "overdetermined": over["accuracy"],
        "n_overdetermined": over["n_seeded_failures"],
        "by_decoy": attr["accuracy_by_decoy"],
        "replays_single": attr["counterfactual_replays"],
        "replays_overdetermined": over["counterfactual_replays"],
    }
    print(json.dumps(summary, indent=2))
    return attr, abl, sweep


if __name__ == "__main__":
    main()
