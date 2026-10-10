# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Engine-level tests on a toy pipeline: no GEPA, no server, a fake evaluator.

The toy loop mirrors the shipped shape: a read-place source that draws from a pool, a worker stage that "evaluates"
(drains a budget and uses a bandwidth), a head gate that tags accepted / rejected, and a head commit that feeds the
pool back. Every test asserts the engine's accounting invariants at the end of the run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from gepa.async_stages import (
    Engine,
    HeadStage,
    Pipeline,
    PipelineError,
    Pool,
    ResourceExhaustedError,
    WorkerStage,
    WorkItem,
)


@dataclass
class Toy:
    parent: int
    value: float = 0.0
    score: float | None = None
    notes: list[str] = field(default_factory=list)


class Source(HeadStage[Toy, Toy]):
    """Reads the pool (committed items) and creates one proposal per firing.

    For a read stage ``in_type`` is the type of the items delivered into its place; the body receives the list.
    """

    in_type = out_type = Toy

    def __init__(self, **kw):
        super().__init__("source", mode="read", **kw)
        self.draws = 0

    def run(self, item):
        self.draws += 1
        best = max(item.payload, key=lambda it: it.payload.score or 0.0)
        return [WorkItem(Toy(parent=best.payload.parent + 1, value=float(self.draws)))]


class Evaluate(WorkerStage[Toy, Toy]):
    in_type = out_type = Toy

    def __init__(self, cost: int = 2, delay: float = 0.0, fail_on: set[float] = frozenset(), **kw):
        super().__init__("evaluate", bandwidth="evaluator", **kw)
        self.cost = cost
        self.delay = delay
        self.fail_on = fail_on
        self.calls = 0
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    def demand(self, item):
        return {"evals": self.cost, "evaluator": self.cost}

    def run(self, item):
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.calls += self.cost
        if self.delay:
            time.sleep(self.delay)
        if item.payload.value in self.fail_on:
            with self._lock:
                self._active -= 1
            item.usage = {"evals": self.cost}  # a failed evaluation was still paid for
            raise RuntimeError("boom")
        item.payload.score = item.payload.value
        item.usage = {"evals": self.cost}  # evaluator units come back; evals are consumed
        with self._lock:
            self._active -= 1
        return [item]


class Gate(HeadStage[Toy, Toy]):
    in_type = out_type = Toy

    def __init__(self, accept_if, **kw):
        super().__init__("gate", **kw)
        self.accept_if = accept_if

    def run(self, item):
        item.tag = "accepted" if self.accept_if(item.payload) else "rejected"
        return [item]


class Commit(HeadStage[Toy, Toy]):
    in_type = out_type = Toy

    def __init__(self, stop_at: int | None = None, engine_ref: dict | None = None, **kw):
        super().__init__("commit", **kw)
        self.commits = 0
        self.stop_at = stop_at
        self.engine_ref = engine_ref if engine_ref is not None else {}

    def run(self, item):
        self.commits += 1
        if self.stop_at is not None and self.commits >= self.stop_at and "engine" in self.engine_ref:
            self.engine_ref["engine"].request_stop("target")
        return [item]


def build(*, evals: int, evaluator: int, in_flight: int = 2, accept_if=lambda t: True, stop_at=None, **eval_kw):
    source = Source(demand={"in_flight": 1}, hold=["in_flight"])
    evaluate = Evaluate(workers=4, **eval_kw)
    ref: dict = {}
    gate = Gate(accept_if)
    commit = Commit(stop_at=stop_at, engine_ref=ref)
    pipeline = Pipeline().chain(source, evaluate, gate).edge(gate, commit, "accepted").edge(commit, source)
    resources = {
        "evals": Pool("evals", evals, unit="evaluation"),
        "evaluator": Pool("evaluator", evaluator),
        "in_flight": Pool("in_flight", in_flight),
    }
    engine = Engine(pipeline, resources, executors={"evaluator": evaluator}, write_event_log=False)
    ref["engine"] = engine
    seed = engine.inject("source", Toy(parent=0, value=0.0, score=0.0))
    seed.outcome = "seed"
    return engine, source, evaluate, gate, commit, resources


def assert_settled(engine: Engine, resources: dict):
    """Every unit taken was put back or reported used; nothing is left in flight or held."""
    for name, pool in resources.items():
        assert pool.in_flight == 0, f"{name}: {pool.in_flight} units still in flight"
        assert engine._in_flight_units.get(name, 0.0) == pytest.approx(0.0), name
    assert all(n == 0 for n in engine.inflight.values())
    for name, stage in engine.pipeline.stages.items():
        if stage.mode == "consume":
            assert not engine.places[name], f"{name} still holds items"


def test_budget_drains_then_the_run_stops_exhausted():
    engine, source, evaluate, gate, commit, resources = build(evals=10, evaluator=8)
    stats = engine.run()
    assert stats["stop_reason"] == "exhausted"
    assert evaluate.calls == 10  # five evaluations of two, the budget exactly
    assert commit.commits == 5
    assert resources["evals"].available == 0
    assert_settled(engine, resources)
    # The refused item ended with outcome budget and its held unit came back.
    assert stats["outcomes"].get("budget", 0) >= 1
    assert resources["in_flight"].available == 2


def test_bandwidth_at_its_cap_is_not_exhaustion():
    engine, source, evaluate, gate, commit, resources = build(evals=12, evaluator=2, in_flight=6, delay=0.02)
    stats = engine.run()
    assert stats["stop_reason"] == "exhausted"  # by the evals budget, not the evaluator
    assert evaluate.peak <= 1  # cost 2 against a cap of 2: one evaluation at a time
    assert_settled(engine, resources)


def test_held_units_pace_the_source():
    engine, source, evaluate, gate, commit, resources = build(evals=40, evaluator=8, in_flight=1, delay=0.01)
    engine.run()
    # One proposal alive at a time: every draw but the last (refused by the budget) reached a commit, and the
    # source never ran ahead of the loop.
    assert source.draws == commit.commits + 1
    assert evaluate.peak == 1
    assert_settled(engine, resources)


def test_rejected_items_end_and_release_their_hold():
    engine, source, evaluate, gate, commit, resources = build(
        evals=20, evaluator=8, in_flight=3, accept_if=lambda t: t.value % 2 == 0
    )
    stats = engine.run()
    assert stats["outcomes"]["rejected"] >= 1
    assert commit.commits >= 1
    assert stats["outcomes"]["rejected"] + commit.commits == evaluate.calls // 2
    assert_settled(engine, resources)


def test_requested_stop_drains_in_flight_work():
    engine, source, evaluate, gate, commit, resources = build(
        evals=1000, evaluator=8, in_flight=4, stop_at=3, delay=0.01
    )
    stats = engine.run()
    assert stats["stop_reason"] == "target"
    assert commit.commits >= 3
    assert_settled(engine, resources)


def test_worker_error_is_fatal_by_default_and_settles():
    engine, source, evaluate, gate, commit, resources = build(evals=100, evaluator=8, fail_on={2.0})
    with pytest.raises(RuntimeError, match="boom"):
        engine.run()
    assert_settled(engine, resources)


def test_worker_error_is_isolated_when_not_raising():
    source = Source(demand={"in_flight": 1}, hold=["in_flight"])
    evaluate = Evaluate(workers=2, fail_on={2.0})
    gate = Gate(lambda t: True)
    commit = Commit()
    pipeline = Pipeline().chain(source, evaluate, gate).edge(gate, commit, "accepted").edge(commit, source)
    resources = {"evals": Pool("evals", 12), "evaluator": Pool("evaluator", 4), "in_flight": Pool("in_flight", 2)}
    engine = Engine(pipeline, resources, executors={"evaluator": 4}, write_event_log=False, raise_on_exception=False)
    engine.inject("source", Toy(parent=0, value=0.0, score=0.0))
    stats = engine.run()
    assert stats["outcomes"].get("error", 0) >= 1
    assert stats["stop_reason"] == "exhausted"
    assert_settled(engine, resources)


def test_pipeline_validation():
    class A(HeadStage[Toy, Toy]):
        in_type = out_type = Toy

        def run(self, item):
            return [item]

    class B(HeadStage[list, list]):
        in_type = out_type = list

        def run(self, item):
            return [item]

    with pytest.raises(PipelineError, match="payload type"):
        Pipeline().chain(A("a"), B("b")).validate()
    p = Pipeline().chain(A("a"), A("a2"))
    with pytest.raises(PipelineError, match="not in the pipeline"):
        p.validate(stage_names_in_config=["zzz"])


def test_pool_exhaustion_semantics():
    pool = Pool("p", 3)
    assert pool.take(2)
    assert not pool.take(2)  # not now: two are out and may come back
    pool.put(2)
    assert pool.take(3)
    pool.put(0, used=3)  # all consumed
    with pytest.raises(ResourceExhaustedError):
        pool.take(1)
    with pytest.raises(ResourceExhaustedError):
        Pool("q", 2).take(5)  # larger than the cap can ever be
    clock = Pool("c", 2, rate=1000.0)
    assert clock.take(2)
    time.sleep(0.01)
    assert clock.take(1)  # refilled from the clock, never exhausted
