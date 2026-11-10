"""Local trace file layout: one JSONL file per run.

Line 1 is a header; every other line is one step. Steps use the same shape as
an llm-trace-replay cassette (`Recorder.cassette(run_id)`):

    {"step": 3, "kind": "llm.call" | "tool.call",
     "request":  {"messages": [...], "params": {...}}  |  {"tool": name, "args": {...}},
     "response": {"text": ..., "model": ..., "usage": {...}} | {"result": ...}}

so traces can be imported from that project's SQLite store without depending
on it (see `import_tracereplay_sqlite`). `req_hash` is the same function.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

TRACE_VERSION = "cae-trace/1"


def req_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Trace:
    run_id: str
    task: str
    inputs: dict  # agent inputs: instructions, memory, tools
    env: dict  # sandbox state needed to re-execute tools live
    steps: list[dict] = field(default_factory=list)
    final: str | None = None
    meta: dict = field(default_factory=dict)

    def header(self) -> dict:
        return {
            "version": TRACE_VERSION,
            "run_id": self.run_id,
            "task": self.task,
            "inputs": self.inputs,
            "env": self.env,
            "final": self.final,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(json.dumps(self.header()) + "\n")
            for s in self.steps:
                f.write(json.dumps(s) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        lines = Path(path).read_text().splitlines()
        h = json.loads(lines[0])
        if h.get("version") != TRACE_VERSION:
            raise ValueError(f"not a {TRACE_VERSION} file")
        return cls(
            h["run_id"],
            h["task"],
            h["inputs"],
            h["env"],
            [json.loads(x) for x in lines[1:] if x.strip()],
            h.get("final"),
            h.get("meta", {}),
        )


def import_tracereplay_sqlite(
    db_path: str, run_id: str, inputs: dict | None = None, env: dict | None = None
) -> Trace:
    """Read one run from an llm-trace-replay SQLite store (table `spans`)
    with sqlite3 only - no OpenTelemetry import needed."""
    conn = sqlite3.connect(db_path)
    try:
        root = conn.execute(
            "SELECT attributes FROM spans WHERE run_id=? AND name='agent.run'",
            (run_id,),
        ).fetchone()
        if root is None:
            raise KeyError(run_id)
        ra = json.loads(root[0])
        steps = []
        for name, step, attrs in conn.execute(
            "SELECT name, step, attributes FROM spans WHERE run_id=? AND step IS NOT NULL ORDER BY step",
            (run_id,),
        ):
            a = json.loads(attrs)
            steps.append(
                {
                    "step": step,
                    "kind": name,
                    "request": json.loads(a.get("replay.request", "null")),
                    "response": json.loads(a.get("replay.response", "null")),
                }
            )
    finally:
        conn.close()
    return Trace(
        run_id,
        ra.get("agent.task", ""),
        inputs or {},
        env or {},
        steps,
        ra.get("agent.final_answer"),
        {"imported_from": "llm-trace-replay"},
    )
