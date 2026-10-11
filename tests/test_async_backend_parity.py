# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Parity check: the ``reflective`` pipeline with one proposal alive at a time reproduces the gepa backend's
sequence of sampling, gating, validation and commits on a seeded toy task with a deterministic fake proposer.

A check on the engine's wiring, not a claim about the method and not prompt parity.
"""

from __future__ import annotations

from gepa.oa.config import OptimizeAnythingConfig


class _LM:
    """Proposes a candidate that depends only on the call count, so both engines see the same sequence."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []
        self.total_cost = 0.0

    def __call__(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        return f"```\ncandidate {self.calls}\n```"


def _evaluator(candidate, example=None):
    # Odd candidates improve, even ones regress, so the gate rejects some and accepts others.
    if candidate.startswith("candidate "):
        n = int(candidate.split()[1])
        base = 1.0 + 0.1 * n if n % 2 == 1 else 0.05
    else:
        base = 0.3
    return base + 0.01 * (hash(str(example)) % 3), {"note": str(example)}


def _run(engine: str, tmp_path, max_evals: int):
    from gepa.optimize_anything import optimize_anything

    lm = _LM()
    data = [f"ex{i}" for i in range(4)]
    if engine == "gepa":
        engine_config = {
            "engine": {"seed": 0, "parallel": False, "cache_evaluation": False, "frontier_type": "instance"},
            "reflection": {"reflection_lm": lm, "reflection_minibatch_size": 2},
        }
    else:
        engine_config = {
            "reflection_lm": lm,
            "seed": 0,
            "cache_evaluation": False,
            "frontier_type": "instance",
            "stages": {"sample": {"minibatch_size": 2, "max_in_flight": 1}},
            "write_event_log": False,
        }
    result = optimize_anything(
        seed_candidate="seed",
        evaluator=_evaluator,
        dataset=data,
        valset=data,
        config=OptimizeAnythingConfig(
            engine=engine,
            max_evals=max_evals,
            max_concurrency=1,
            run_dir=str(tmp_path / engine),
            engine_config=engine_config,
        ),
    )
    return result, lm


def test_one_proposal_at_a_time_reproduces_the_gepa_backend(tmp_path):
    gepa_result, gepa_lm = _run("gepa", tmp_path, max_evals=40)
    async_result, async_lm = _run("async", tmp_path, max_evals=40)
    assert async_result.candidates == gepa_result.candidates
    assert async_result.val_aggregate_scores == gepa_result.val_aggregate_scores
    assert async_result.parents == gepa_result.parents
    # Same reflective datasets and template, call for call. The async backend may make one more reflection at the
    # tail, since by design it keeps proposing until a resource is exhausted, where the gepa backend stops before an
    # iteration it cannot commit.
    n = gepa_lm.calls
    assert n <= async_lm.calls <= n + 1
    assert async_lm.prompts[:n] == gepa_lm.prompts
