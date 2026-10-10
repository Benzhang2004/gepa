# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The simple async pipeline with the default proposer: sample -> rollout -> propose -> guard -> screen -> gate ->
validate -> commit, closed by an edge from commit back into sample's read place."""

from __future__ import annotations

from typing import Any

from gepa.async_stages.pipeline import Pipeline
from gepa.async_stages.pipelines.resources import ProposalsInFlight, ReflectionLM
from gepa.async_stages.pipelines.stages import (
    Commit,
    Deps,
    Gate,
    Propose,
    Rollout,
    SampleParent,
    Screen,
    StalenessGuard,
    Validate,
    validate_select,
)

#: Keys the engine owns in a stage's config entry; everything else goes to the stage's constructor.
SCHEDULING_KEYS = ("workers", "max_workers", "max_queue", "select", "demand", "hold")


def split_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sched = {k: v for k, v in entry.items() if k in SCHEDULING_KEYS}
    opts = {k: v for k, v in entry.items() if k not in SCHEDULING_KEYS}
    return sched, opts


def build(
    stage_opts: dict[str, dict[str, Any]],
    resource_opts: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    deps: Deps,
) -> Pipeline:
    def o(name: str) -> dict[str, Any]:
        sched, opts = split_entry(stage_opts.get(name, {}))
        kw = dict(sched)
        kw.update(opts)
        return kw

    sample_kw = o("sample")
    sample_kw.setdefault("demand", {"proposals_in_flight": 1})
    sample_kw.setdefault("hold", ["proposals_in_flight"])
    sample = SampleParent(deps, **sample_kw)
    minibatch_size = sample_kw.get("minibatch_size", 3)
    rollout = Rollout(deps, minibatch_size=minibatch_size, **o("rollout"))
    propose = Propose(deps, **o("propose"))
    guard_kw = o("guard")
    guard = StalenessGuard(deps, **guard_kw) if "guard" in stage_opts else None
    screen = Screen(deps, minibatch_size=minibatch_size, **o("screen"))
    gate = Gate(deps, **o("gate"))
    validate_kw = o("validate")
    select = validate_kw.pop("select", "fifo")
    validate = Validate(deps, select=validate_select(select) if isinstance(select, str) else select, **validate_kw)
    commit = Commit(deps, **o("commit"))

    pipeline = Pipeline().resource(
        ReflectionLM(**resource_opts.get("reflection_lm", {})),
        ProposalsInFlight(**resource_opts.get("proposals_in_flight", {})),
    )
    chain = [sample, rollout, propose]
    if guard is not None:
        chain.append(guard)
    chain += [screen]
    pipeline.chain(*chain)
    pipeline.chain(screen, gate)
    pipeline.edge(gate, validate, "accepted")
    pipeline.chain(validate, commit)
    pipeline.edge(commit, sample)
    return pipeline
