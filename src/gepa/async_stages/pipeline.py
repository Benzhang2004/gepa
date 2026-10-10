# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""A pipeline: stages, the edges that join a body's output to a downstream stage's place, and the resources the
stages draw on.

Edges may be labelled; a body tags an item to pick among several outgoing edges, and an untagged item takes the
default edge. Cycles are allowed and expected: an optimization pipeline is a loop. A pipeline declares the
resources its stages demand beyond the ones the engine shares with every pipeline (the optimize_anything budget
and server limits); the engine holds them for the run. The validator checks that edges name stages that exist,
that the payload types on either end agree, that every demand names a declared or shared resource, and that
every config entry names a stage or resource of the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from gepa.async_stages.resources import Resource
from gepa.async_stages.stage import Stage


class PipelineError(ValueError):
    pass


class Pipeline:
    def __init__(self, stages: Iterable[Stage[Any, Any]] = ()):
        self.stages: dict[str, Stage[Any, Any]] = {}
        #: (source stage name, label) -> destination stage name; label ``None`` is the default edge.
        self.edges: dict[tuple[str, str | None], str] = {}
        #: The resources this pipeline declares, by name (beyond the engine's shared ones).
        self.resources: dict[str, Resource] = {}
        for stage in stages:
            self.add(stage)

    def resource(self, *resources: Resource) -> Pipeline:
        for res in resources:
            if res.name in self.resources and self.resources[res.name] is not res:
                raise PipelineError(f"two resources named {res.name!r}")
            self.resources[res.name] = res
        return self

    def add(self, stage: Stage[Any, Any]) -> Pipeline:
        if stage.name in self.stages and self.stages[stage.name] is not stage:
            raise PipelineError(f"two stages named {stage.name!r}")
        self.stages[stage.name] = stage
        return self

    def edge(self, src: Stage[Any, Any], dst: Stage[Any, Any], label: str | None = None) -> Pipeline:
        self.add(src)
        self.add(dst)
        key = (src.name, label)
        if key in self.edges and self.edges[key] != dst.name:
            raise PipelineError(f"{src.name!r} already has an edge for label {label!r}")
        self.edges[key] = dst.name
        return self

    def chain(self, *stages: Stage[Any, Any]) -> Pipeline:
        """One default edge from each stage to the next."""
        for a, b in pairwise(stages):
            self.edge(a, b)
        if len(stages) == 1:
            self.add(stages[0])
        return self

    def successor(self, src: str, tag: str | None) -> Stage[Any, Any] | None:
        """The stage an item leaving ``src`` with ``tag`` goes to; ``None`` when there is no such edge."""
        dst = self.edges.get((src, tag))
        if dst is None and tag is not None:
            return None
        if dst is None:
            return None
        return self.stages[dst]

    def validate(
        self,
        stage_names_in_config: Iterable[str] = (),
        resource_names_in_config: Iterable[str] = (),
        shared: Iterable[str] = (),
    ) -> None:
        """``shared`` names the resources the engine provides to every pipeline (evals, evaluator, usd)."""
        for (src, label), dst in self.edges.items():
            if src not in self.stages or dst not in self.stages:
                raise PipelineError(f"edge {src!r} -> {dst!r} (label {label!r}) names a stage that does not exist")
            out_t, in_t = self.stages[src].out_type, self.stages[dst].in_type
            if out_t is not None and in_t is not None and not issubclass(out_t, in_t):
                raise PipelineError(f"edge {src!r} -> {dst!r}: payload type {out_t.__name__} is not a {in_t.__name__}")
        unknown = [n for n in stage_names_in_config if n not in self.stages]
        if unknown:
            raise PipelineError(f"config names stages that are not in the pipeline: {unknown}")
        shared_names = set(shared)
        clash = sorted(set(self.resources) & shared_names)
        if clash:
            raise PipelineError(f"pipeline redefines shared resources: {clash}")
        known = set(self.resources) | shared_names
        for stage in self.stages.values():
            for name in stage.nominal_demand:
                if name not in known:
                    raise PipelineError(
                        f"stage {stage.name!r} demands resource {name!r}, which the pipeline does not declare "
                        f"and the engine does not share (declared: {sorted(self.resources)}, shared: {sorted(shared_names)})"
                    )
        unknown = [n for n in resource_names_in_config if n not in known]
        if unknown:
            raise PipelineError(f"config names resources that are not in the pipeline: {unknown}")
