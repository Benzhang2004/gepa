# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Evaluation stages: rollout (the parent on its minibatch), screen (the child on it), validate (the child on the
valset). All three call the eval server directly, against the ``evals`` reservation the engine took at dispatch.
Their demand is nominal and constant per stage (the minibatch, the valset) so the engine can predict it; what a
firing actually evaluated, the cache misses, is what ``used`` reports, and the rest comes back. Cache reads happen
in ``demand`` and cache writes in ``after``, both on the engine thread, so the GEPAState cache is never touched
from a worker."""

from __future__ import annotations

from typing import Any

from gepa.async_stages.pipelines.stages.common import Deps, Proposal
from gepa.async_stages.stage import WorkerStage, WorkItem
from gepa.core.adapter import EvaluationBatch
from gepa.core.state import TRAINSET_CACHE_SPLIT, VALSET_CACHE_SPLIT, ValsetEvaluation


def _misses(deps: Deps, candidate: dict[str, str], ids: list[Any], split: str) -> tuple[dict[Any, Any], list[Any]]:
    state = deps.box.state
    cache = state.evaluation_cache if state is not None else None
    if cache is None:
        return {}, list(ids)
    cached, todo = cache.get_batch(candidate, ids, split=split)
    return cached, list(todo)


def _evaluate(deps: Deps, candidate: dict[str, str], examples: list[Any]) -> list[tuple[float, dict[str, Any]]]:
    if not examples:
        return []
    server_candidate = deps.to_server_candidate(candidate)
    pairs = [(server_candidate, ex) for ex in examples]
    return deps.server.evaluate_batch(
        pairs, reserved=True, isolate_errors=deps.isolate_errors, sequential=deps.sequential_evals
    )


def _objective_scores(candidate: dict[str, str], info: dict[str, Any]) -> dict[str, float]:
    """Per-objective scores from a side_info, as the optimize_anything adapter extracts them (top-level ``scores``
    plus ``<component>_specific_info.scores``); an empty dict when the evaluator gives none."""
    out: dict[str, float] = {}
    if isinstance(info.get("scores"), dict):
        out.update(info["scores"])
    for name in candidate:
        specific = info.get(f"{name}_specific_info")
        if isinstance(specific, dict) and isinstance(specific.get("scores"), dict):
            out.update({f"{name}::{k}": v for k, v in specific["scores"].items()})
    return out


def _batch(candidate: dict[str, str], results: list[tuple[float, dict[str, Any]]], *, traces: bool) -> EvaluationBatch:
    scores = [float(s) for s, _ in results]
    infos = [dict(i) for _, i in results]
    objective = [_objective_scores(candidate, i) for i in infos]
    return EvaluationBatch(
        outputs=[(s, candidate, i) for s, i in zip(scores, infos, strict=True)],
        scores=scores,
        trajectories=infos if traces else None,
        objective_scores=objective,
        num_metric_calls=len(results),
    )


def _put_cache(deps: Deps, candidate: dict[str, str], ids: list[Any], batch: EvaluationBatch, split: str) -> None:
    state = deps.box.state
    if state is None or state.evaluation_cache is None or not ids:
        return
    obj = list(batch.objective_scores) if batch.objective_scores else None
    state.evaluation_cache.put_batch(candidate, ids, batch.outputs, batch.scores, obj, split=split)


class _MinibatchEval(WorkerStage[Proposal, Proposal]):
    """Shared shape of rollout and screen: evaluate one candidate on the minibatch, cache misses only."""

    in_type = out_type = Proposal
    split = TRAINSET_CACHE_SPLIT

    def __init__(self, name: str, deps: Deps, *, minibatch_size: int = 3, **kw: Any):
        # Nominal cost, constant per stage: the minibatch. Cache hits come back as unused at completion.
        kw.setdefault("demand", {"evaluator": minibatch_size, "evals": minibatch_size})
        super().__init__(name, bandwidth="evaluator", **kw)
        self.deps = deps

    def candidate(self, p: Proposal) -> dict[str, str]:
        raise NotImplementedError

    def demand(self, item: WorkItem[Proposal]) -> dict[str, float]:
        p = item.payload
        _cached, todo = _misses(self.deps, self.candidate(p), p.minibatch_ids, self.split)
        p.pending_ids = todo  # decided here, on the engine thread, where the cache may be read
        return dict(self._demand)

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        by_id = dict(zip(p.minibatch_ids, p.minibatch, strict=True))
        p.results = _evaluate(self.deps, self.candidate(p), [by_id[i] for i in p.pending_ids])
        item.usage = {"evals": len(p.results)}
        return [item]

    def after(self, item: WorkItem[Proposal]) -> None:
        p = item.payload
        candidate = self.candidate(p)
        fresh = dict(zip(p.pending_ids, p.results or [], strict=True))
        cached, _ = _misses(self.deps, candidate, p.minibatch_ids, self.split)
        results: list[tuple[float, dict[str, Any]]] = []
        for eid in p.minibatch_ids:
            if eid in fresh:
                results.append(fresh[eid])
            else:
                entry = cached[eid]
                results.append((entry.score, entry.output[2] if isinstance(entry.output, tuple) else {}))
        batch = _batch(candidate, results, traces=True)
        fresh_batch = _batch(candidate, [fresh[i] for i in p.pending_ids], traces=True)
        _put_cache(self.deps, candidate, p.pending_ids, fresh_batch, self.split)
        self.store(p, batch)
        p.metric_calls += len(p.pending_ids)
        state = self.deps.box.state
        if state is not None and p.pending_ids:
            state.increment_evals(len(p.pending_ids))
        p.pending_ids, p.results = [], None

    def store(self, p: Proposal, batch: EvaluationBatch) -> None:
        raise NotImplementedError


class Rollout(_MinibatchEval):
    """The parent on its minibatch, with traces for reflection."""

    def __init__(self, deps: Deps, **kw: Any):
        super().__init__("rollout", deps, **kw)

    def candidate(self, p: Proposal) -> dict[str, str]:
        return p.parent

    def store(self, p: Proposal, batch: EvaluationBatch) -> None:
        p.parent_eval = batch


class Screen(_MinibatchEval):
    """The child on the same minibatch."""

    def __init__(self, deps: Deps, **kw: Any):
        super().__init__("screen", deps, **kw)

    def candidate(self, p: Proposal) -> dict[str, str]:
        assert p.child is not None
        return p.child

    def store(self, p: Proposal, batch: EvaluationBatch) -> None:
        p.child_eval = batch


class Validate(WorkerStage[Proposal, Proposal]):
    """The child (or the seed) on the whole valset; only cache misses are evaluated."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, **kw: Any):
        n = len(deps.valset)  # nominal cost: the whole valset; cache hits come back as unused at completion
        kw.setdefault("demand", {"evaluator": n, "evals": n})
        super().__init__("validate", bandwidth="evaluator", **kw)
        self.deps = deps

    def demand(self, item: WorkItem[Proposal]) -> dict[str, float]:
        p = item.payload
        assert p.child is not None
        p.val_ids = list(self.deps.valset.all_ids())
        p.val_cached, p.val_todo = _misses(self.deps, p.child, p.val_ids, VALSET_CACHE_SPLIT)
        return dict(self._demand)

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        assert p.child is not None
        p.val_results = _evaluate(self.deps, p.child, self.deps.valset.fetch(p.val_todo))
        item.usage = {"evals": len(p.val_results)}
        return [item]

    def after(self, item: WorkItem[Proposal]) -> None:
        p = item.payload
        assert p.child is not None
        fresh = _batch(p.child, p.val_results or [], traces=False)
        _put_cache(self.deps, p.child, p.val_todo, fresh, VALSET_CACHE_SPLIT)
        outputs_by: dict[Any, Any] = {}
        scores_by: dict[Any, float] = {}
        objective_by: dict[Any, Any] | None = None
        for vid, entry in p.val_cached.items():
            outputs_by[vid], scores_by[vid] = entry.output, entry.score
            if entry.objective_scores is not None:
                objective_by = objective_by or {}
                objective_by[vid] = entry.objective_scores
        obj = list(fresh.objective_scores) if fresh.objective_scores else None
        for j, vid in enumerate(p.val_todo):
            outputs_by[vid], scores_by[vid] = fresh.outputs[j], fresh.scores[j]
            if obj is not None:
                objective_by = objective_by or {}
                objective_by[vid] = obj[j]
        p.valset_evaluation = ValsetEvaluation(
            outputs_by_val_id=outputs_by, scores_by_val_id=scores_by, objective_scores_by_val_id=objective_by
        )
        p.metric_calls += len(p.val_todo)
        state = self.deps.box.state
        if state is not None and p.val_todo:
            state.increment_evals(len(p.val_todo))
        p.val_results = None
