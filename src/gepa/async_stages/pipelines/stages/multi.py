# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The multi-stage proposer of ``sibling_proposals``: one reflection over P x N rollouts writes S strategy slots,
and one generator per slot implements it on the slot's base parent. The two prompts are drafted from the task's
objective and background; reflection and generation share the ``reflection_lm`` bandwidth in v1."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa.async_stages.pipelines.stages.common import Deps, Proposal
from gepa.async_stages.pipelines.stages.evaluate import _batch, _evaluate, _misses, _put_cache
from gepa.async_stages.pipelines.stages.search import _selector
from gepa.async_stages.stage import HeadStage, WorkerStage, WorkItem
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import TRAINSET_CACHE_SPLIT, new_iteration_id
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler


@dataclass
class ParentRollouts:
    parent_idx: int
    parent: dict[str, str]
    minibatch_ids: list[Any]
    minibatch: list[Any]
    eval: EvaluationBatch | None = None
    pending_ids: list[Any] = field(default_factory=list)
    results: list[tuple[float, dict[str, Any]]] | None = None


@dataclass
class MultiProposal:
    """P parents, each rolled out on N minibatches (the union of the N is stored per parent)."""

    parents: list[ParentRollouts]
    slots: list[dict[str, Any]] = field(default_factory=list)
    reflect_meta: dict[str, Any] = field(default_factory=dict)


class SampleMulti(HeadStage[Proposal, MultiProposal]):
    """Reads the pool: P parents (the selector, re-drawn; repeats allowed), N minibatches each, one item."""

    in_type = Proposal
    out_type = MultiProposal

    def __init__(
        self,
        deps: Deps,
        *,
        parents: int = 1,
        minibatches: int = 1,
        minibatch_size: int = 3,
        candidate_selector: str | CandidateSelector = "pareto",
        batch_sampler: BatchSampler | None = None,
        **kw: Any,
    ):
        super().__init__("sample", mode="read", **kw)
        self.deps = deps
        self.parents = parents
        self.minibatches = minibatches
        self.selector = _selector(candidate_selector, deps.rng)
        self.batch_sampler = batch_sampler or EpochShuffledBatchSampler(minibatch_size=minibatch_size, rng=deps.rng)

    def run(self, item: WorkItem[Any]) -> list[WorkItem[MultiProposal]]:
        state = self.deps.box.get()
        drawn: list[ParentRollouts] = []
        for _ in range(self.parents):
            idx = self.selector.select_candidate_idx(state)
            ids: list[Any] = []
            for _ in range(self.minibatches):
                state.i += 1
                state.full_program_trace.append({"i": state.i, "iteration_id": new_iteration_id(), "engine": "async"})
                for eid in self.batch_sampler.next_minibatch_ids(self.deps.trainset, state):
                    if eid not in ids:
                        ids.append(eid)
            drawn.append(ParentRollouts(idx, state.program_candidates[idx], ids, self.deps.trainset.fetch(ids)))
        self.deps.logger.log(
            f"Sampled parents {[d.parent_idx for d in drawn]} with {[len(d.minibatch_ids) for d in drawn]} examples"
        )
        return [WorkItem(MultiProposal(parents=drawn))]


class RolloutMulti(WorkerStage[MultiProposal, MultiProposal]):
    """Every parent on its examples, with traces; cache misses only, nominal demand P x N x minibatch."""

    in_type = out_type = MultiProposal

    def __init__(self, deps: Deps, *, parents: int = 1, minibatches: int = 1, minibatch_size: int = 3, **kw: Any):
        n = parents * minibatches * minibatch_size
        slots = n if deps.evaluator_cap is None else min(n, deps.evaluator_cap)
        kw.setdefault("demand", {"evaluator": slots, "evals": n})
        super().__init__("rollout", bandwidth="evaluator", **kw)
        self.deps = deps

    def demand(self, item: WorkItem[MultiProposal]) -> dict[str, float]:
        for pr in item.payload.parents:
            _cached, pr.pending_ids = _misses(self.deps, pr.parent, pr.minibatch_ids, TRAINSET_CACHE_SPLIT)
        return dict(self._demand)

    def run(self, item: WorkItem[MultiProposal]) -> list[WorkItem[MultiProposal]]:
        total = 0
        for pr in item.payload.parents:
            by_id = dict(zip(pr.minibatch_ids, pr.minibatch, strict=True))
            pr.results = _evaluate(self.deps, pr.parent, [by_id[i] for i in pr.pending_ids])
            total += len(pr.results)
        item.usage = {"evals": total}
        return [item]

    def after(self, item: WorkItem[MultiProposal]) -> None:
        state = self.deps.box.state
        for pr in item.payload.parents:
            fresh = dict(zip(pr.pending_ids, pr.results or [], strict=True))
            cached, _ = _misses(self.deps, pr.parent, pr.minibatch_ids, TRAINSET_CACHE_SPLIT)
            rows: list[tuple[float, dict[str, Any]]] = []
            for eid in pr.minibatch_ids:
                if eid in fresh:
                    rows.append(fresh[eid])
                else:
                    entry = cached[eid]
                    rows.append((entry.score, entry.output[2] if isinstance(entry.output, tuple) else {}))
            pr.eval = _batch(pr.parent, rows, traces=True)
            _put_cache(
                self.deps,
                pr.parent,
                pr.pending_ids,
                _batch(pr.parent, [fresh[i] for i in pr.pending_ids], traces=True),
                TRAINSET_CACHE_SPLIT,
            )
            if state is not None and pr.pending_ids:
                state.increment_evals(len(pr.pending_ids))
            pr.pending_ids, pr.results = [], None


REFLECT_PROMPT = """You are helping optimize {components} in a system.{objective}{background}

Below are {n_parents} current version(s) of it (numbered parents) and, for each, evaluation feedback from several
rollouts. Read all of the evidence together.

{evidence}

Your task: propose {slots} distinct high-level strategies for improving the component. Each strategy must pursue a
different hypothesis about what to change (a different failure pattern to fix, a different approach, or a different
trade-off), not a rewording of the same idea. Each names the parent it builds on, and the component it changes.
Do NOT write the new text itself; a separate writer will implement each strategy.

Answer with exactly this format, one block per strategy:

### Slot 1
Parent: <parent number>
Component: <component name>
Brief: <what to change, why, and which failures it targets; two to six sentences>

### Slot 2
...
"""

GENERATE_PROMPT = """You are improving the component `{component}` of a system.{objective}{background}

Current text of the component:
```
{current}
```

Strategy to implement:
{brief}

Write the complete new text of the component that implements this strategy. Keep everything in the current text
that the strategy does not ask to change. Provide the full new text within ``` blocks, and nothing else.
"""

_SLOT = re.compile(r"^\s*#{1,6}\s*Slot\s*(\d+)\b.*$", re.IGNORECASE | re.MULTILINE)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _format_evidence(deps: Deps, parents: Sequence[ParentRollouts], components: list[str]) -> str:
    parts: list[str] = []
    for n, pr in enumerate(parents, start=1):
        assert pr.eval is not None
        parts.append(f"## Parent {n}")
        for comp in components:
            parts.append(f"### Component `{comp}` (current text)\n```\n{pr.parent.get(comp, '')}\n```")
        dataset = deps.adapter.make_reflective_dataset(pr.parent, pr.eval, components)
        for comp, rows in dataset.items():
            parts.append(f"### Feedback for `{comp}` ({len(rows)} rollouts)")
            for k, row in enumerate(rows, start=1):
                parts.append(f"#### Rollout {k}")
                for key, value in row.items():
                    parts.append(f"{key}: {value}")
    return "\n".join(parts)


def parse_slots(raw: str, n_parents: int, components: list[str], max_slots: int) -> list[dict[str, Any]]:
    heads = list(_SLOT.finditer(raw))
    slots: list[dict[str, Any]] = []
    for i, h in enumerate(heads):
        body = raw[h.end() : heads[i + 1].start() if i + 1 < len(heads) else len(raw)]
        parent = re.search(r"Parent:\s*(\d+)", body)
        comp = re.search(r"Component:\s*`?([\w\-. ]+)`?", body)
        brief = re.search(r"Brief:\s*(.*)", body, re.DOTALL)
        p = int(parent.group(1)) - 1 if parent else 0
        c = comp.group(1).strip() if comp else components[0]
        if c not in components:
            c = components[0]
        text = brief.group(1).strip() if brief else body.strip()
        if not text:
            continue
        slots.append({"parent": min(max(p, 0), n_parents - 1), "component": c, "brief": text})
        if len(slots) >= max_slots:
            break
    return slots


def parse_generated(raw: str) -> str | None:
    fences = _FENCE.findall(raw)
    if fences:
        return fences[-1].strip() or None
    text = raw.strip()
    return text or None


class _RefitsUsd:
    """Re-fit the per-call USD estimate from the LM's metered total, as the propose stage does. Reflect and generate
    share the one metered total, so the call count is shared through ``deps.lm_calls``."""

    deps: Deps
    _demand: dict[str, float]

    def after(self, item: WorkItem[Any]) -> None:
        total = getattr(self.deps.lm_cost_source, "total_cost", None)
        if isinstance(total, int | float):
            self.deps.lm_calls["n"] += 1
            if total > 0:
                self._demand["usd"] = total / self.deps.lm_calls["n"]


class Reflect(_RefitsUsd, WorkerStage[MultiProposal, Proposal]):
    """One reflection call over every rollout; fans out one proposal per slot (base parent + brief)."""

    in_type = MultiProposal
    out_type = Proposal

    def __init__(self, deps: Deps, *, slots: int = 4, usd_first_guess: float = 0.01, **kw: Any):
        kw.setdefault("demand", {"reflection_lm": 1, "usd": usd_first_guess})
        super().__init__("reflect", bandwidth="reflection_lm", **kw)
        self.deps = deps
        self.slots = slots

    def run(self, item: WorkItem[MultiProposal]) -> list[WorkItem[Proposal]]:
        mp = item.payload
        lm = getattr(self.deps.reflection_lm, "lm", None)
        assert lm is not None, "the sibling_proposals pipeline needs a reflection LM"
        components = list(mp.parents[0].parent.keys())
        objective = f"\n\nObjective: {self.deps.objective}" if self.deps.objective else ""
        background = f"\n\nBackground:\n{self.deps.background}" if self.deps.background else ""
        prompt = REFLECT_PROMPT.format(
            components=", ".join(f"`{c}`" for c in components),
            objective=objective,
            background=background,
            n_parents=len(mp.parents),
            evidence=_format_evidence(self.deps, mp.parents, components),
            slots=self.slots,
        )
        raw = str(lm(prompt))
        mp.reflect_meta = {"prompt": prompt, "raw": raw}
        mp.slots = parse_slots(raw, len(mp.parents), components, self.slots)
        item.usage = {"usd": self._demand.get("usd", 0.0)}
        outputs: list[WorkItem[Proposal]] = []
        for k, slot in enumerate(mp.slots):
            base = mp.parents[slot["parent"]]
            proposal = Proposal(
                parent_idx=base.parent_idx,
                parent=base.parent,
                minibatch_ids=list(base.minibatch_ids),
                minibatch=list(base.minibatch),
                parent_eval=base.eval,
                components=[slot["component"]],
                proposal_meta={"slot": k, "brief": slot["brief"], "reflect_raw": raw},
            )
            outputs.append(WorkItem(proposal))
        if not outputs:
            item.tag = "no_proposal"
        return outputs


class Generate(_RefitsUsd, WorkerStage[Proposal, Proposal]):
    """Implements one slot's brief on its base parent: a complete new text for the named component."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, usd_first_guess: float = 0.01, **kw: Any):
        kw.setdefault("demand", {"reflection_lm": 1, "usd": usd_first_guess})
        super().__init__("generate", bandwidth="reflection_lm", **kw)
        self.deps = deps

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        lm = getattr(self.deps.reflection_lm, "lm", None)
        assert lm is not None
        component = (p.components or list(p.parent))[0]
        objective = f"\n\nObjective: {self.deps.objective}" if self.deps.objective else ""
        background = f"\n\nBackground:\n{self.deps.background}" if self.deps.background else ""
        prompt = GENERATE_PROMPT.format(
            component=component,
            objective=objective,
            background=background,
            current=p.parent.get(component, ""),
            brief=p.proposal_meta.get("brief", ""),
        )
        raw = str(lm(prompt))
        text = parse_generated(raw)
        p.proposal_meta["generate_prompt"] = prompt
        p.proposal_meta["generate_raw"] = raw
        item.usage = {"usd": self._demand.get("usd", 0.0)}
        if not text or text == p.parent.get(component):
            item.tag = "no_proposal"
            return [item]
        child = dict(p.parent)
        child[component] = text
        p.child = child
        return [item]
