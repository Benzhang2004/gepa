# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end tests of the ``async`` backend through optimize_anything, on a counting evaluator and a fake LM."""

from __future__ import annotations

import pytest

from gepa.oa.config import OptimizeAnythingConfig


class _CountingLM:
    """Deterministic reflection LM: every call proposes a longer candidate; tracks cost like gepa.lm.LM."""

    def __init__(self):
        self.calls = 0
        self.total_cost = 0.0

    def __call__(self, prompt, *args, **kwargs):
        self.calls += 1
        self.total_cost += 0.001
        return f"```\nimproved candidate v{self.calls}\n```"


def _evaluator_factory(calls: list):
    def evaluator(candidate, example=None):
        calls.append((candidate, example))
        # Every later proposal is a little better, so strict improvement keeps accepting.
        version = int(candidate.rsplit("v", 1)[1]) if "improved candidate v" in candidate else 0
        return (1.0 if version else 0.2) + 0.01 * version, {"note": f"scored {example}"}

    return evaluator


def _run(tmp_path, *, max_evals: int, evals_mode: str = "bulk", in_flight: int = 2, extra_stages=None):
    from gepa.optimize_anything import optimize_anything

    calls: list = []
    lm = _CountingLM()
    data = [f"ex{i}" for i in range(4)]
    stages = {
        "sample": {"minibatch_size": 2, "max_in_flight": in_flight},
        "rollout": {"workers": 2},
        "propose": {"workers": 2},
        "screen": {"workers": 2},
        "validate": {"workers": 2, "select": "score"},
    }
    if extra_stages:
        stages.update(extra_stages)
    result = optimize_anything(
        seed_candidate="seed",
        evaluator=_evaluator_factory(calls),
        dataset=data,
        valset=data,
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=max_evals,
            max_concurrency=4,
            run_dir=str(tmp_path / "run"),
            engine_config={
                "reflection_lm": lm,
                "resources": {"evals": {"mode": evals_mode}, "reflection_lm": {"cap": 2}},
                "stages": stages,
                "write_event_log": True,
            },
        ),
    )
    return result, calls, lm


@pytest.mark.parametrize("evals_mode", ["bulk", "passthrough"])
def test_async_backend_runs_the_reflective_loop_within_the_budget(tmp_path, evals_mode):
    result, calls, lm = _run(tmp_path, max_evals=30, evals_mode=evals_mode)
    assert len(calls) <= 30  # the strict ceiling holds
    assert len(result.candidates) >= 2  # the seed and at least one committed child
    assert result.best_candidate.startswith("improved candidate")
    assert result.val_aggregate_scores[result.best_idx] > result.val_aggregate_scores[0]
    stats = result.metadata["async"]
    assert stats["stop_reason"] in ("exhausted", "quiescent")
    res = stats["resources"]
    assert res["evals"]["used"] == len(calls)
    assert res["evaluator"]["taken"] == res["evaluator"]["put_back"]  # a bandwidth comes back whole
    assert lm.calls >= 1
    assert result.metadata["adapter_cost"] == pytest.approx(lm.total_cost)


def test_async_backend_one_proposal_at_a_time_commits_every_improvement(tmp_path):
    result, calls, lm = _run(tmp_path, max_evals=24, in_flight=1)
    # seed 4 + per proposal: parent rollout (cached after the seed, 0) + child screen 2 + validation 2 (2 cached)
    assert len(calls) <= 24
    assert len(result.candidates) >= 3


def test_async_backend_with_the_staleness_guard(tmp_path):
    result, calls, lm = _run(tmp_path, max_evals=30, in_flight=4, extra_stages={"guard": {"max_gap": 0}})
    stats = result.metadata["async"]
    assert "stale" in stats["outcomes"] or len(result.candidates) >= 2
    assert len(calls) <= 30


def test_unknown_engine_config_key_fails_fast():
    from gepa.oa.engines.async_backend import AsyncBackend

    with pytest.raises(TypeError, match="unknown engine_config keys"):
        AsyncBackend(OptimizeAnythingConfig(engine="async", max_evals=4, engine_config={"budgets": {}}))


class _SlotLM:
    """Answers a reflect prompt with S slots and a generate prompt with a fenced text; counts both."""

    def __init__(self, slots: int = 3):
        self.slots = slots
        self.reflects = 0
        self.generates = 0
        self.total_cost = 0.0

    def __call__(self, prompt, *args, **kwargs):
        self.total_cost += 0.001
        if "Strategy to implement" in prompt:
            self.generates += 1
            return f"```\nimproved candidate v{self.generates}\n```"
        self.reflects += 1
        blocks = [
            f"### Slot {k}\nParent: 1\nComponent: current_candidate\nBrief: strategy {k} for reflection {self.reflects}"
            for k in range(1, self.slots + 1)
        ]
        return "\n\n".join(blocks)


def test_sibling_proposals_fans_out_one_generator_per_slot(tmp_path):
    from gepa.optimize_anything import optimize_anything

    calls: list = []
    lm = _SlotLM(slots=3)
    data = [f"ex{i}" for i in range(4)]
    result = optimize_anything(
        seed_candidate="seed",
        evaluator=_evaluator_factory(calls),
        dataset=data,
        valset=data,
        objective="make it longer",
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=60,
            max_concurrency=4,
            run_dir=str(tmp_path / "run"),
            engine_config={
                "pipeline": "sibling_proposals",
                "reflection_lm": lm,
                "resources": {"reflection_lm": {"cap": 3}},
                "stages": {
                    "sample": {"parents": 1, "minibatches": 2, "minibatch_size": 2, "max_in_flight": 1},
                    "reflect": {"slots": 3},
                    "generate": {"workers": 3},
                    "screen": {"workers": 3},
                    "validate": {"workers": 2, "select": "score"},
                },
                "write_event_log": False,
            },
        ),
    )
    assert lm.reflects >= 1
    assert lm.generates == 3 * lm.reflects  # one writer per slot, every reflection
    assert len(calls) <= 60
    assert len(result.candidates) >= 2
    stats = result.metadata["async"]
    assert stats["stages"]["generate"]["completed"] == lm.generates
