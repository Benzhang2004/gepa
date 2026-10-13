# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The agent pipeline: proposals are coding-agent sessions, one proposer stage per direction, and (``dynamic``) a
spawner that reshapes the pipeline while it runs.

sample -> rollout -> route -> propose_<k> -> screen -> gate -> gate_ledger -> validate -> commit -> tee -> sample,
with tee also sending every ``meta_every``-th commit to the spawner, which may add a proposer (a new direction),
resize a stage, reweight or retire a proposer.

Stage config: ``propose`` is a template applied to every proposer stage (``propose_0``, ``propose_1``, ...).
Pipeline-wide options (``engine_config["pipeline"]``): ``task`` (an :class:`AgentTask`, required), ``directions``
(list of strings; one proposer each; ``[""]`` by default), ``dynamic`` (bool, default False), ``meta_every``
(commits between meta steps, default 2), ``agent_options`` (dict for :class:`AgentOptions`), ``meta_options``,
``runner`` (tests inject a fake session runner), ``keep_workspaces``. Resources: ``agent`` (``{"cap": n}``
concurrent sessions, default 3) and ``meta`` (default 1).
"""

from __future__ import annotations

from typing import Any

from gepa.async_stages.pipeline import Pipeline, PipelineError
from gepa.async_stages.pipelines.reflective import split_entry
from gepa.async_stages.pipelines.stages import (
    Commit,
    Deps,
    Gate,
    Rollout,
    SampleParent,
    Screen,
    Validate,
    validate_select,
)
from gepa.async_stages.pipelines.stages.agent import (
    AgentOptions,
    AgentPropose,
    AgentSessions,
    AgentTask,
    LedgerGate,
    MetaSessions,
    Route,
    Spawner,
    Tee,
    run_claude,
)
from gepa.async_stages.pipelines.stages.decompose import Brief, Places, Planner


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

    task = defaults.get("task")
    if not isinstance(task, AgentTask):
        raise PipelineError("the agent pipeline needs pipeline={'name': 'agent', 'task': AgentTask(...)}")
    directions: list[str] = list(defaults.get("directions") or [""])
    runner = defaults.get("runner") or run_claude
    keep_workspaces = bool(defaults.get("keep_workspaces", False))

    places = Places(deps.run_dir)
    notes_k = int(defaults.get("notes", 3))
    plan_k = int(defaults.get("plan") or 0)

    sample_kw = o("sample")
    sample_kw.setdefault("max_in_flight", 4)
    sample_kw.setdefault("minibatch_size", 1)
    sample = SampleParent(deps, **sample_kw)
    minibatch_size = sample_kw["minibatch_size"]
    rollout = Rollout(deps, minibatch_size=minibatch_size, **o("rollout"))
    route = Route(deps, **o("route"))

    # "propose" in the stage config is a template for every proposer (propose_0, propose_1, ...), not a stage of
    # its own: consume it here so the backend's validation does not look for a stage by that name.
    propose_template = stage_opts.pop("propose", {})

    def new_proposer(name: str, direction: str) -> AgentPropose:
        sched, opts = split_entry(propose_template)
        kw = dict(sched)
        kw.update(opts)
        kw.setdefault("workers", 1)
        return AgentPropose(
            deps,
            task,
            name=name,
            direction=direction,
            runner=runner,
            options=AgentOptions(**dict(defaults.get("agent_options") or {})),
            keep_workspaces=keep_workspaces,
            places=places,
            notes_k=notes_k,
            **kw,
        )

    proposers = [new_proposer(f"propose_{i}", d) for i, d in enumerate(directions)]
    screen = Screen(deps, minibatch_size=minibatch_size, **o("screen"))
    gate = Gate(deps, **o("gate"))
    gate_ledger = LedgerGate(deps, name="gate_ledger", what="gate")
    validate_kw = o("validate")
    select = validate_kw.pop("select", "fifo")
    validate = Validate(deps, select=validate_select(select) if isinstance(select, str) else select, **validate_kw)
    commit = Commit(deps, **o("commit"))
    commit_ledger = LedgerGate(deps, name="commit_ledger", what="commit")
    tee = Tee(deps, every=int(defaults.get("meta_every", 2)), brief=bool(defaults.get("brief")))

    pipeline = Pipeline().resource(
        AgentSessions(**resource_opts.get("agent", {"cap": 3})),
        MetaSessions(**resource_opts.get("meta", {"cap": 1})),
    )
    if plan_k:
        meta_runner = defaults.get("meta_runner") or runner
        plan_opts: dict[str, Any] = dict(defaults.get("plan_options") or {})
        plan_opts.setdefault("max_usd", 1.0)
        plan_opts.setdefault("timeout", 600.0)
        planner = Planner(
            deps, task, places, k=plan_k, runner=meta_runner, options=AgentOptions(**plan_opts), **o("plan")
        )
        pipeline.chain(sample, rollout, planner, route)
    else:
        pipeline.chain(sample, rollout, route)
    if defaults.get("brief"):
        brief_opts: dict[str, Any] = dict(defaults.get("brief_options") or {})
        brief_opts.setdefault("max_usd", 3.0)
        brief_opts.setdefault("timeout", 900.0)
        brief = Brief(deps, task, places, runner=runner, options=AgentOptions(**brief_opts), **o("brief"))
        pipeline.edge(tee, brief, "brief")
    for stage in proposers:
        pipeline.edge(route, stage, stage.name)
        pipeline.edge(stage, screen)
        route.weights[stage.name] = 1.0
    pipeline.chain(screen, gate)
    pipeline.edge(gate, gate_ledger, "accepted")
    pipeline.edge(gate, gate_ledger, "rejected")
    pipeline.edge(gate_ledger, validate, "accepted")
    pipeline.chain(validate, commit, commit_ledger, tee, sample)
    if defaults.get("dynamic"):
        meta_opts: dict[str, Any] = dict(defaults.get("meta_options") or {})
        meta_opts.setdefault("max_usd", 1.0)
        meta_opts.setdefault("timeout", 600.0)
        meta_options = AgentOptions(**meta_opts)
        spawner_kw = o("spawner")
        spawner = Spawner(
            deps,
            task,
            route,
            new_proposer=new_proposer,
            downstream="screen",
            runner=defaults.get("meta_runner") or runner,
            options=meta_options,
            **spawner_kw,
        )
        pipeline.edge(tee, spawner, "meta")
    return pipeline
