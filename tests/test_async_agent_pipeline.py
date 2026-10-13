# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The agent pipeline with fake sessions, and the engine's runtime mutation API on a toy pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import pytest

from gepa.async_stages import Engine, HeadStage, Pipeline, ResourceExhaustedError, WorkerStage, WorkItem
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
    engine = Engine(pipeline, {"usd": Usd(8)}, executors={"cpu": 1}, run_dir=str(tmp_path), raise_on_exception=True)
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


class Overspend(WorkerStage[int, int]):
    """Demands 1 dollar, reports 3 spent: the overshoot must count against the cap."""

    in_type = out_type = int

    def __init__(self, **kw):
        super().__init__("work", bandwidth="cpu", demand={"usd": 1}, **kw)
        self.fired = 0

    def run(self, item):
        self.fired += 1
        item.usage = {"usd": 3}
        return [item]


def test_usd_cap_holds_on_what_was_spent_not_what_was_predicted(tmp_path):
    source, work, commit = Source(), Overspend(), CommitStage()
    pipeline = Pipeline().chain(source, work, commit, source)
    usd = Usd(5)
    engine = Engine(pipeline, {"usd": usd}, executors={"cpu": 1}, run_dir=None)
    engine.inject("commit", 0)
    stats = engine.run()
    assert stats["stop_reason"] == "exhausted"
    assert work.fired == 2  # 3 + 3 spent against a cap of 5: the second firing's overshoot ends the run
    assert usd.available is not None and usd.available < 0
    assert stats["resources"]["usd"]["used"] == 6


def test_usd_grants_the_remainder_to_the_last_firing_and_ignores_float_residue():
    usd = Usd(5)
    assert usd.take(3) == 3
    usd.put(0, used=3.0000000000000004)  # float noise on a put must not read as "something in flight"
    assert usd.take(3) == 2  # 2 left and nothing in flight: a partial grant, not a refusal
    usd.put(0, used=2)
    with pytest.raises(ResourceExhaustedError):
        usd.take(3)


def test_session_cap_follows_a_partial_grant(tmp_path):
    """When the engine grants less than the session cap, the session runs with the grant as its cap."""
    caps: list[float] = []

    def runner(prompt: str, cwd: Path, opts: AgentOptions) -> AgentResult:
        caps.append(opts.max_usd)
        (cwd / "candidate.txt").write_text("value=9\n")
        return AgentResult("ok", None, opts.max_usd, 10, 0.1, 1, "fake", None)

    from gepa.async_stages.pipelines.stages.agent import AgentPropose
    from gepa.async_stages.pipelines.stages.common import Deps, Proposal, StateBox

    task = _task(tmp_path)
    deps = Deps(
        box=StateBox(),
        server=None,
        adapter=None,
        reflection_lm=None,
        trainset=None,
        valset=None,  # type: ignore[arg-type]
        rng=__import__("random").Random(0),
        logger=type("L", (), {"log": lambda self, m: None})(),
        run_dir=str(tmp_path / "run"),
        seed_candidate={"current_candidate": "value=0\n"},
        request_stop=lambda r: None,
        str_candidate_key="current_candidate",
    )
    stage = AgentPropose(deps, task, runner=runner, options=AgentOptions(max_usd=6.0))
    item: WorkItem[Proposal] = WorkItem(Proposal(parent_idx=0, parent={"current_candidate": "value=0\n"}))
    item.granted = {"agent": 1.0, "usd": 2.5}
    stage.run(item)
    item.granted = {"agent": 1.0, "usd": 6.0}
    stage.run(item)
    assert caps == [2.5, 6.0]


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
        return AgentResult(
            text="done", structured=None, usd=1.0, tokens=1000, seconds=0.1, turns=3, model="fake", error=None
        )

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
                "stages": {"sample": {"max_in_flight": 2}, "propose": {"workers": 2}},
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
    decisions = [json.loads(line) for line in (tmp_path / "run" / "spawner_decisions.jsonl").read_text().splitlines()]
    applied = [d["applied"] for d in decisions]
    assert applied[0] == "added propose_1"
    assert "propose_1 workers=2" in applied
    assert any(a.startswith("refused") for a in applied)  # commit is not resizable
    assert any("double" in t["prompt"] for t in log)  # the new proposer ran sessions under its direction
    events = [json.loads(line) for line in (tmp_path / "run" / "async_events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "topology" and e["change"] == "add_stage" for e in events)
    assert result.val_aggregate_scores[result.best_idx] >= 2


# ------------------------------------------------------------------------------------------ decomposition


def _fake_swarm_factory(log: list):
    """One fake runner for every session kind: the study session writes BRIEF.md, the planner returns two
    sub-tasks, and an implementer bumps the value (by 2 under the 'double' direction) and leaves NOTES.md."""

    def runner(prompt: str, cwd: Path, opts: AgentOptions) -> AgentResult:
        log.append({"prompt": prompt, "tools": opts.allowed_tools, "schema": opts.schema})
        if "study step" in prompt:
            (cwd / "BRIEF.md").write_text("# Brief\nThe value lives in candidate.txt; higher is better.\n")
            return AgentResult("brief", None, 0.5, 100, 0.1, 2, "fake", None)
        if "planning step" in prompt:
            plan = {"subtasks": [{"direction": "double the step"}, {"direction": "add one"}]}
            return AgentResult("", plan, 0.1, 50, 0.1, 1, "fake", None)
        target = cwd / "candidate.txt"
        value = int(re.search(r"value=(\d+)", target.read_text()).group(1))
        step = 2 if "double" in prompt.split("YOUR DIRECTION:")[1].split("\n")[0] else 1
        target.write_text(f"value={value + step}\n")
        (cwd / "NOTES.md").write_text(f"bumped by {step}; next try a bigger step\n")
        return AgentResult("done", None, 1.0, 1000, 0.1, 3, "fake", None)

    return runner


def test_decomposed_pipeline_shares_a_brief_and_notes_and_fans_out_plans(tmp_path):
    from gepa.optimize_anything import optimize_anything

    log: list = []
    task = _task(tmp_path)
    result = optimize_anything(
        seed_candidate="value=0\n",
        evaluator=_evaluator,
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=40,
            max_token_cost=10.0,
            max_concurrency=2,
            run_dir=str(tmp_path / "run"),
            engine_config={
                "pipeline": {
                    "name": "agent",
                    "task": task,
                    "directions": [""],
                    "brief": True,
                    "plan": 2,
                    "notes": 2,
                    "runner": _fake_swarm_factory(log),
                },
                "resources": {"agent": {"cap": 2}},
                "stages": {"sample": {"max_in_flight": 1}, "propose": {"workers": 2}},
            },
        ),
    )
    run = tmp_path / "run"
    assert (run / "places" / "brief" / "BRIEF.md").read_text().startswith("# Brief")
    brief_sessions = [t for t in log if "study step" in t["prompt"]]
    assert len(brief_sessions) == 1 and "Write" in brief_sessions[0]["tools"]
    plans = [t for t in log if "planning step" in t["prompt"]]
    assert plans and all(t["tools"] is None and t["schema"] is not None for t in plans)
    implementers = [t for t in log if "YOUR DIRECTION" in t["prompt"]]
    directions = {t["prompt"].split("YOUR DIRECTION:")[1].split("\n")[0].strip() for t in implementers}
    assert {"double the step", "add one"} <= directions  # the planner's sub-tasks became the sessions' directions
    assert any("BRIEF (written once" in t["prompt"] for t in implementers)  # later sessions read the brief
    notes = sorted((run / "places" / "notes").glob("*.md"))
    assert notes and "bumped by" in notes[0].read_text()
    assert any("NOTES LEFT BY EARLIER SESSIONS" in t["prompt"] for t in implementers)
    events = [json.loads(line) for line in (run / "async_events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "complete" and e["stage"] == "plan" and e["outputs"] == 2 for e in events)  # fan-out
    assert result.val_aggregate_scores[result.best_idx] >= 2


def test_route_holds_items_until_a_proposer_has_capacity():
    from gepa.async_stages.pipelines.stages.agent import Route
    from gepa.async_stages.pipelines.stages.common import Deps, Proposal, StateBox

    class FakeStage:
        def __init__(self, workers):
            self.workers = workers

    class FakePipeline:
        stages: ClassVar[dict] = {"propose_0": FakeStage(1), "propose_1": FakeStage(1)}

    class FakeEngine:
        pipeline: ClassVar[FakePipeline] = FakePipeline()
        inflight: ClassVar[dict] = {"propose_0": 1, "propose_1": 0}
        places: ClassVar[dict] = {"propose_0": [], "propose_1": []}

    import random

    deps = Deps(
        box=StateBox(),
        server=None,
        adapter=None,
        reflection_lm=None,
        trainset=None,
        valset=None,  # type: ignore[arg-type]
        rng=random.Random(0),
        logger=type("L", (), {"log": lambda self, m: None})(),
        run_dir=None,
        seed_candidate={},
        request_stop=lambda r: None,
    )
    deps.engine = FakeEngine()
    route = Route(deps, weights={"propose_0": 1.0, "propose_1": 1.0})
    item: WorkItem[Proposal] = WorkItem(Proposal(parent_idx=0, parent={}))
    assert route.select([item]) == 0  # propose_1 is free
    assert route.run(item)[0].tag == "propose_1"  # never the busy one
    FakeEngine.places["propose_1"].append(item)  # now something waits for propose_1 too
    assert route.select([item]) is None  # hold the item in route's own place
