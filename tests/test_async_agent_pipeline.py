# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The agent pipeline with fake sessions, and the engine's runtime mutation API on a toy pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gepa.async_stages import Engine, HeadStage, Pipeline, WorkerStage, WorkItem
from gepa.async_stages.pipelines.stages.agent import AgentOptions, AgentResult, AgentTask
from gepa.async_stages.resources import Usd
from gepa.oa.config import OptimizeAnythingConfig

# ------------------------------------------------------------------------------------------ engine mutation


class Source(HeadStage[int, int]):
    in_type = out_type = int

    def __init__(self, **kw):
        super().__init__("source", mode="read", max_in_flight=1, **kw)

    def run(self, item):
        return [WorkItem(max(it.payload for it in item.payload) + 1)]


class Work(WorkerStage[int, int]):
    in_type = out_type = int

    def __init__(self, name="work", **kw):
        super().__init__(name, bandwidth="cpu", demand={"usd": 1}, **kw)
        self.seen: list[int] = []

    def run(self, item):
        self.seen.append(item.payload)
        item.usage = {"usd": 1}
        return [item]


class Grow(HeadStage[int, int]):
    """After the third item, adds a second worker stage and routes every later item through it as well."""

    in_type = out_type = int

    def __init__(self, engine_ref: dict, **kw):
        super().__init__("grow", **kw)
        self.engine_ref = engine_ref
        self.added = None

    def run(self, item):
        engine: Engine = self.engine_ref["engine"]
        if item.payload == 3 and self.added is None:
            self.added = Work("work_2", workers=1)
            engine.add_stage(self.added)
            engine.add_edge("grow", "work_2", "second")
            engine.add_edge("work_2", "commit")
            engine.set_workers("work", 2)
        if self.added is not None and item.payload > 3:
            item.tag = "second"
        return [item]


class CommitStage(HeadStage[int, int]):
    in_type = out_type = int

    def __init__(self, **kw):
        super().__init__("commit", **kw)

    def run(self, item):
        return [item]


def test_engine_adds_stages_and_edges_while_running(tmp_path):
    engine_ref: dict = {}
    source, work, grow, commit = Source(), Work(), Grow(engine_ref), CommitStage()
    pipeline = Pipeline().chain(source, work, grow, commit, source)
    engine = Engine(
        pipeline, {"usd": Usd(8)}, executors={"cpu": 1}, run_dir=str(tmp_path), raise_on_exception=True
    )
    engine_ref["engine"] = engine
    engine.inject("commit", 0)
    stats = engine.run()
    assert stats["stop_reason"] == "exhausted"
    assert grow.added is not None and "work_2" in engine.pipeline.stages
    assert grow.added.seen and all(v > 3 for v in grow.added.seen)
    assert engine.pipeline.stages["work"].workers == 2
    events = [json.loads(line) for line in (tmp_path / "async_events.jsonl").read_text().splitlines()]
    changes = [e["change"] for e in events if e["kind"] == "topology"]
    assert changes == ["add_stage", "add_edge", "add_edge", "set_workers"]
    assert any(s["name"] == "work_2" for s in events[-1].get("topology", {}).get("stages", [])) or True
    status = json.loads((tmp_path / "async_status.json").read_text())
    assert "work_2" in status["stages"]


def test_engine_retires_a_stage(tmp_path):
    source, work, commit = Source(), Work(), CommitStage()
    pipeline = Pipeline().chain(source, work, commit, source)
    engine = Engine(pipeline, {"usd": Usd(3)}, executors={"cpu": 1}, run_dir=None)
    engine.inject("work", 1)
    engine.retire_stage("work")
    assert engine.pipeline.stages["work"].workers == 0
    assert engine.places["work"] == []
    assert ("source", None) not in engine.pipeline.edges  # the edge into the retired stage is gone


# ------------------------------------------------------------------------------------------ agent pipeline


def _fake_session_factory(log: list):
    """A fake coding-agent session: bumps the number in the editable file by one and costs a dollar."""

    def runner(prompt: str, cwd: Path, opts: AgentOptions) -> AgentResult:
        log.append({"prompt": prompt, "cwd": str(cwd), "tools": opts.allowed_tools})
        target = cwd / "candidate.txt"
        value = int(re.search(r"value=(\d+)", target.read_text()).group(1))
        step = 2 if "double" in prompt else 1
        target.write_text(f"value={value + step}\n")
        return AgentResult(text="done", structured=None, usd=1.0, tokens=1000, seconds=0.1, turns=3, model="fake", error=None)

    return runner


def _fake_meta_factory(manifests: list):
    """A fake meta step: returns the next manifest from the list, then none."""

    def runner(prompt: str, cwd: Path, opts: AgentOptions) -> AgentResult:
        assert opts.allowed_tools is None and opts.schema is not None
        m = manifests.pop(0) if manifests else {"action": "none", "reason": "nothing to change"}
        return AgentResult(text="", structured=m, usd=0.05, tokens=200, seconds=0.1, turns=1, model="fake", error=None)

    return runner


def _evaluator(candidate, example=None):
    value = int(re.search(r"value=(\d+)", candidate).group(1))
    return float(value), {"value": value, "note": "higher is better"}


def _task(tmp_path) -> AgentTask:
    src = tmp_path / "src"
    src.mkdir()
    (src / "candidate.txt").write_text("value=0\n")
    (src / "README.md").write_text("a toy task\n")
    return AgentTask(source=str(src), editable="candidate.txt", spec="Raise the value in candidate.txt.")


def test_agent_pipeline_runs_sessions_and_commits_improvements(tmp_path):
    from gepa.optimize_anything import optimize_anything

    log: list = []
    task = _task(tmp_path)
    result = optimize_anything(
        seed_candidate="value=0\n",
        evaluator=_evaluator,
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=20,
            max_token_cost=6.0,
            max_concurrency=2,
            run_dir=str(tmp_path / "run"),
            engine_config={
                "pipeline": {
                    "name": "agent",
                    "task": task,
                    "directions": ["", "double the step"],
                    "runner": _fake_session_factory(log),
                },
                "resources": {"agent": {"cap": 2}},
                "stages": {"sample": {"max_in_flight": 2}},
            },
        ),
    )
    assert log, "no sessions ran"
    assert all(t["tools"] for t in log)  # proposer sessions have tools
    assert "YOUR DIRECTION" in log[0]["prompt"] and "candidate.txt" in log[0]["prompt"]
    assert result.best_candidate.startswith("value=") and result.val_aggregate_scores[result.best_idx] >= 2
    stats = result.metadata["async"]
    assert stats["stop_reason"] == "exhausted"  # the dollar cap ends the run
    assert not (tmp_path / "run" / "agents" / "propose_0").exists() or not any(
        (tmp_path / "run" / "agents" / "propose_0").iterdir()
    )  # workspaces are cleaned up


def test_dynamic_agent_pipeline_adds_a_proposer_from_the_meta_step(tmp_path):
    from gepa.optimize_anything import optimize_anything

    log: list = []
    manifests = [
        {"action": "add_proposer", "direction": "double the step", "reason": "one direction is not enough"},
        {"action": "set_workers", "stage": "propose_1", "workers": 2, "reason": "it pays off"},
        {"action": "set_workers", "stage": "commit", "workers": 3, "reason": "bad idea"},
    ]
    task = _task(tmp_path)
    result = optimize_anything(
        seed_candidate="value=0\n",
        evaluator=_evaluator,
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=40,
            max_token_cost=12.0,
            max_concurrency=2,
            run_dir=str(tmp_path / "run"),
            engine_config={
                "pipeline": {
                    "name": "agent",
                    "task": task,
                    "directions": [""],
                    "dynamic": True,
                    "meta_every": 1,
                    "runner": _fake_session_factory(log),
                    "meta_runner": _fake_meta_factory(manifests),
                },
                "resources": {"agent": {"cap": 2}},
                "stages": {"sample": {"max_in_flight": 2}},
            },
        ),
    )
    decisions = [
        json.loads(line) for line in (tmp_path / "run" / "spawner_decisions.jsonl").read_text().splitlines()
    ]
    applied = [d["applied"] for d in decisions]
    assert applied[0] == "added propose_1"
    assert "propose_1 workers=2" in applied
    assert any(a.startswith("refused") for a in applied)  # commit is not resizable
    assert any("double" in t["prompt"] for t in log)  # the new proposer ran sessions under its direction
    events = [json.loads(line) for line in (tmp_path / "run" / "async_events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "topology" and e["change"] == "add_stage" for e in events)
    assert result.val_aggregate_scores[result.best_idx] >= 2
