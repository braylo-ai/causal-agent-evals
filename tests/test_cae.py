import copy
import json
import random
import sqlite3

import pytest

from cae.attribution import BASELINES, candidates, counterfactual_attribution
from cae.bench import ablation_bench, attribution_bench, clean_cases, fault_sweep
from cae.replay import diff_steps, record, replay, unfaithful_steps
from cae.seed import FAULT_TYPES, overdetermined_failures, seeded_failures
from cae.trace import Trace, import_tracereplay_sqlite, req_hash
from cae.world import expected_outcome, passed, random_case


@pytest.fixture
def case():
    rng = random.Random(4)
    while True:
        task, inputs, env = random_case(rng, 0)
        tr = record(task, inputs, env)
        if passed(task, env, tr.final) and expected_outcome(task, env).startswith(
            "refund"
        ):
            return task, inputs, env, tr


def test_clean_record_passes_and_is_faithful(case):
    *_, tr = case
    assert unfaithful_steps(tr) == []


def test_replay_without_interventions_is_identical(case):
    *_, tr = case
    assert diff_steps(tr, replay(tr)) is None


def test_trace_roundtrip(tmp_path, case):
    *_, tr = case
    tr.save(tmp_path / "t.jsonl")
    back = Trace.load(tmp_path / "t.jsonl")
    assert (
        back.steps == tr.steps and back.final == tr.final and back.inputs == tr.inputs
    )


def test_steps_use_tracereplay_cassette_shape(case):
    *_, tr = case
    for s in tr.steps:
        assert set(s) == {"step", "kind", "request", "response"}
        assert s["kind"] in ("llm.call", "tool.call")
        if s["kind"] == "tool.call":
            assert set(s["request"]) == {"tool", "args"} and "result" in s["response"]
        else:
            assert "messages" in s["request"] and "text" in s["response"]


def test_import_from_tracereplay_sqlite(tmp_path, case):
    *_, tr = case
    db = tmp_path / "traces.db"
    c = sqlite3.connect(db)
    c.executescript(
        "CREATE TABLE spans (run_id TEXT, trace_id TEXT, span_id TEXT PRIMARY KEY, parent_id TEXT,"
        " name TEXT, step INTEGER, start_ns INTEGER, end_ns INTEGER, status TEXT, attributes TEXT);"
    )
    c.execute(
        "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "r1",
            "t",
            "root",
            None,
            "agent.run",
            None,
            0,
            1,
            "UNSET",
            json.dumps({"agent.task": tr.task, "agent.final_answer": tr.final}),
        ),
    )
    for s in tr.steps:
        c.execute(
            "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "r1",
                "t",
                f"s{s['step']}",
                "root",
                s["kind"],
                s["step"],
                0,
                1,
                "UNSET",
                json.dumps(
                    {
                        "replay.request": json.dumps(s["request"]),
                        "replay.response": json.dumps(s["response"]),
                        "replay.request_hash": req_hash(s["request"]),
                    }
                ),
            ),
        )
    c.commit()
    c.close()
    imp = import_tracereplay_sqlite(str(db), "r1", tr.inputs, tr.env)
    assert imp.steps == tr.steps and imp.final == tr.final
    assert diff_steps(imp, replay(imp)) is None


def test_counterfactual_swap_changes_outcome(case):
    task, inputs, env, tr = case
    pos = next(
        s["step"]
        for s in tr.steps
        if s["kind"] == "tool.call" and s["request"]["tool"] == "lookup_order"
    )
    alt = copy.deepcopy(tr.steps[pos]["response"])
    alt["result"]["days_since_delivery"] = 999
    out = replay(tr, [{"type": "set_response", "step": pos, "response": alt}])
    assert out.final.startswith("denied") and not passed(task, env, out.final)


def test_timeout_on_policy_triggers_permissive_fallback(case):
    task, inputs, env, tr = case
    pos = next(
        s["step"]
        for s in tr.steps
        if s["kind"] == "tool.call" and s["request"]["tool"] == "get_policy"
    )
    out = replay(tr, [{"type": "timeout", "step": pos}])
    assert out.steps[pos]["response"]["result"] == {"error": "timeout"}


def test_ablation_remove_lookup_breaks_run(case):
    task, inputs, env, tr = case
    out = replay(tr, [{"type": "remove_tool", "tool": "lookup_order"}])
    assert out.final == "could not find order"


def test_ablation_drop_instruction_changes_prompt(case):
    *_, tr = case
    out = replay(tr, [{"type": "drop_instruction", "id": "R2"}])
    assert "[R2]" not in out.steps[0]["request"]["messages"][0]["content"]
    assert not any(
        s["kind"] == "tool.call" and s["request"]["tool"] == "recall" for s in out.steps
    )


def test_unfaithful_detects_injected_fault(case):
    task, inputs, env, tr = case
    bad = record(task, inputs, env, [{"type": "timeout", "step": 3}])
    assert unfaithful_steps(bad) == [3]


def test_held_deviation_survives_unrelated_restore():
    """Ceteris paribus: restoring a decoy must not silently undo the real fault."""
    case = next(
        c
        for c in seeded_failures(60, seed=3)
        if c["decoy"] and c["fault_type"] == "llm_wrong_arg"
    )
    decoy_iv = dict(candidates(case["trace"], case["reference_inputs"]))[case["decoy"]]
    out = replay(case["trace"], [decoy_iv])
    assert not passed(case["trace"].task, case["trace"].env, out.final)


@pytest.mark.parametrize("ftype", FAULT_TYPES)
def test_each_fault_type_can_be_seeded_and_attributed(ftype):
    cases = [
        c
        for c in seeded_failures(len(FAULT_TYPES) * 2, seed=11)
        if c["fault_type"] == ftype
    ]
    assert cases
    for c in cases:
        assert not passed(c["trace"].task, c["trace"].env, c["trace"].final)
        assert (
            counterfactual_attribution(c["trace"], c["reference_inputs"])["blame"]
            == c["ground_truth"]
        )


def test_overdetermined_needs_pair():
    cases = overdetermined_failures(4, seed=5)
    assert len(cases) == 4
    for c in cases:
        res = counterfactual_attribution(c["trace"], c["reference_inputs"])
        assert "+" in res["blame"] and res["blame"] == c["ground_truth"]


def test_baselines_return_candidate_ids(case):
    *_, tr = case
    for f in BASELINES.values():
        assert f(tr, tr.inputs).startswith(("step:", "input:", "unknown"))


def test_attribution_bench_counts():
    res = attribution_bench(18, seed=9)
    assert res["n_seeded_failures"] == 18
    assert sum(1 for _ in res["cases"]) == 18
    assert 0 <= res["accuracy"]["first_anomaly"] <= 1


def test_ablation_and_sweep_shapes():
    traces = clean_cases(5, seed=1)
    abl = ablation_bench(traces)
    assert abl["remove_tool:lookup_order"]["fail_rate"] == 1.0
    sw = fault_sweep(traces)
    assert sw["timeout:lookup_order"]["injected"] == 5


def test_committed_results_reproduce(tmp_path):
    from cae.bench import main

    attr, abl, sweep = main(["--out", str(tmp_path)])
    committed = json.load(open("results/attribution.json"))
    assert committed["accuracy"] == attr["accuracy"]
    assert committed["hits"] == attr["hits"]
