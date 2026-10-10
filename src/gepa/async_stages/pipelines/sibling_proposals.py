# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The analogue of GEPA's parallel proposals with one shared reflection: sample (P parents, N minibatches each) ->
rollout -> reflect (one call, S strategy slots) -> generate (one writer per slot) -> guard -> screen -> gate ->
validate -> commit, closed by an edge from commit back into sample's read place."""

from __future__ import annotations

from typing import Any

from gepa.async_stages.pipeline import Pipeline
from gepa.async_stages.pipelines.reflective import split_entry
from gepa.async_stages.pipelines.stages import Commit, Deps, Gate, Screen, StalenessGuard, Validate, validate_select
from gepa.async_stages.pipelines.stages.multi import Generate, Reflect, RolloutMulti, SampleMulti


def build(stage_opts: dict[str, dict[str, Any]], defaults: dict[str, Any], deps: Deps) -> Pipeline:
    def o(name: str) -> dict[str, Any]:
        sched, opts = split_entry(stage_opts.get(name, {}))
        kw = dict(sched)
        kw.update(opts)
        return kw

    sample_kw = o("sample")
    sample_kw.setdefault("demand", {"proposals_in_flight": 1})
    sample_kw.setdefault("hold", ["proposals_in_flight"])
    shape = {
        "parents": sample_kw.get("parents", 1),
        "minibatches": sample_kw.get("minibatches", 1),
        "minibatch_size": sample_kw.get("minibatch_size", 3),
    }
    sample = SampleMulti(deps, **sample_kw)
    rollout = RolloutMulti(deps, **shape, **o("rollout"))
    reflect = Reflect(deps, **o("reflect"))
    generate = Generate(deps, **o("generate"))
    guard = StalenessGuard(deps, **o("guard")) if "guard" in stage_opts else None
    screen = Screen(deps, minibatch_size=shape["minibatches"] * shape["minibatch_size"], **o("screen"))
    gate = Gate(deps, **o("gate"))
    validate_kw = o("validate")
    select = validate_kw.pop("select", "fifo")
    validate = Validate(deps, select=validate_select(select) if isinstance(select, str) else select, **validate_kw)
    commit = Commit(deps, **o("commit"))

    pipeline = Pipeline()
    chain = [sample, rollout, reflect, generate]
    if guard is not None:
        chain.append(guard)
    chain.append(screen)
    pipeline.chain(*chain)
    pipeline.chain(screen, gate)
    pipeline.edge(gate, validate, "accepted")
    pipeline.chain(validate, commit)
    pipeline.edge(commit, sample)
    return pipeline
