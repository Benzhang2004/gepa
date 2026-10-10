# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The search stages of the shipped pipelines: sample (reads the pool), propose (reflection), the staleness guard,
the gate (acceptance) and commit (the pool's only writer). GEPA's selectors, samplers, acceptance criteria and
state are imported here, never by the engine."""

from __future__ import annotations

from typing import Any

from gepa.async_stages.pipelines.stages.common import Deps, Proposal
from gepa.async_stages.stage import HeadStage, WorkerStage, WorkItem, by_key
from gepa.core.state import VALSET_CACHE_SPLIT, EvaluationCache, initialize_gepa_state, new_iteration_id
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import CandidateSelector, ReflectionComponentSelector
from gepa.strategies.acceptance import AcceptanceCriterion, ImprovementOrEqualAcceptance, StrictImprovementAcceptance
from gepa.strategies.batch_sampler import BatchSampler, EpochShuffledBatchSampler
from gepa.strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
    TopKParetoCandidateSelector,
)
from gepa.strategies.component_selector import AllReflectionComponentSelector, RoundRobinReflectionComponentSelector
from gepa.strategies.eval_policy import FullEvaluationPolicy


def _selector(name: str | CandidateSelector, rng: Any) -> CandidateSelector:
    if not isinstance(name, str):
        return name
    table = {
        "pareto": lambda: ParetoCandidateSelector(rng=rng),
        "current_best": lambda: CurrentBestCandidateSelector(),
        "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(epsilon=0.1, rng=rng),
        "top_k_pareto": lambda: TopKParetoCandidateSelector(k=5, rng=rng),
    }
    if name not in table:
        raise ValueError(f"unknown candidate_selector {name!r}")
    return table[name]()


class SampleParent(HeadStage[Proposal, Proposal]):
    """Reads the pool: picks a parent with the candidate selector, draws a minibatch, creates one proposal.

    A read-place stage: its place holds every committed candidate (the seed first); the body ignores the list
    itself and reads the GEPAState the pipeline keeps.
    """

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        *,
        candidate_selector: str | CandidateSelector = "pareto",
        minibatch_size: int = 3,
        batch_sampler: BatchSampler | None = None,
        **kw: Any,
    ):
        super().__init__("sample", mode="read", **kw)
        self.deps = deps
        self.selector = _selector(candidate_selector, deps.rng)
        self.batch_sampler = batch_sampler or EpochShuffledBatchSampler(minibatch_size=minibatch_size, rng=deps.rng)

    def run(self, item: WorkItem[Any]) -> list[WorkItem[Proposal]]:
        state = self.deps.box.get()
        parent_idx = self.selector.select_candidate_idx(state)
        state.i += 1
        state.full_program_trace.append({"i": state.i, "iteration_id": new_iteration_id(), "engine": "async"})
        ids = list(self.batch_sampler.next_minibatch_ids(self.deps.trainset, state))
        self.deps.logger.log(
            f"Iteration {state.i + 1}: Selected program {parent_idx} score: {state.program_full_scores_val_set[parent_idx]}"
        )
        proposal = Proposal(
            parent_idx=parent_idx,
            parent=state.program_candidates[parent_idx],
            minibatch_ids=ids,
            minibatch=self.deps.trainset.fetch(ids),
        )
        return [WorkItem(proposal)]


class Propose(WorkerStage[Proposal, Proposal]):
    """Builds the reflective dataset from the parent's rollouts and asks the reflection LM for a child."""

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        *,
        module_selector: str | ReflectionComponentSelector = "round_robin",
        usd_first_guess: float = 0.01,
        **kw: Any,
    ):
        super().__init__("propose", bandwidth="reflection_lm", **kw)
        self.deps = deps
        if isinstance(module_selector, str):
            table: dict[str, type[ReflectionComponentSelector]] = {
                "round_robin": RoundRobinReflectionComponentSelector,
                "all": AllReflectionComponentSelector,
            }
            self.module_selector: ReflectionComponentSelector = table[module_selector]()
        else:
            self.module_selector = module_selector
        self._usd_estimate = usd_first_guess

    def demand(self, item: WorkItem[Proposal]) -> dict[str, float]:
        p = item.payload
        if p.components is None and p.parent_eval is not None and p.parent_eval.trajectories:
            # Component selection reads and advances the state's round-robin pointer: engine thread only.
            p.components = self.module_selector(
                self.deps.box.get(), p.parent_eval.trajectories, p.parent_eval.scores, p.parent_idx, p.parent
            )
        return {"reflection_lm": 1, "usd": self._usd_estimate}

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        if p.parent_eval is None or not p.parent_eval.trajectories or not p.components:
            item.tag = "no_trajectories"
            return [item]
        lm = self.deps.reflection_lm
        assert lm is not None, "the shipped propose stage needs a reflection LM"
        dataset = self.deps.adapter.make_reflective_dataset(p.parent, p.parent_eval, p.components)
        proposal, _next = lm.reflect(p.parent, dataset, p.components)
        new_texts = dict(proposal.new_texts or {})
        p.proposal_meta = {"prompts": dict(proposal.prompts), "raw_lm_outputs": dict(proposal.raw_lm_outputs)}
        item.usage = {"usd": self._usd_estimate}  # the call was made whether or not it proposed anything
        if not new_texts or all(p.parent.get(k) == v for k, v in new_texts.items()):
            item.tag = "no_proposal"
            return [item]
        child = dict(p.parent)
        child.update(new_texts)
        p.child = child
        return [item]

    def after(self, item: WorkItem[Proposal]) -> None:
        # Re-fit the per-call estimate from the LM's metered total, when it has one.
        source = self.deps.lm_cost_source
        total = getattr(source, "total_cost", None)
        if isinstance(total, int | float):
            self.deps.lm_calls["n"] += 1
            if total > 0:
                self._usd_estimate = total / self.deps.lm_calls["n"]


class StalenessGuard(HeadStage[Proposal, Proposal]):
    """Drops a proposal whose stamp of the commit counter is more than ``max_gap`` behind its current value."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, max_gap: int = 4, watch: str = "commit", **kw: Any):
        super().__init__("guard", **kw)
        self.deps = deps
        self.max_gap = max_gap
        self.watch = watch

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        gap = item.gap(self.watch, self.deps.counters.get(self.watch, 0))
        if gap > self.max_gap:
            item.tag = "stale"
            return [item]
        return [item]


class Gate(HeadStage[Proposal, Proposal]):
    """Acceptance against the parent on the same minibatch; tags ``accepted`` or ``rejected``."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, acceptance: str | AcceptanceCriterion = "strict_improvement", **kw: Any):
        super().__init__("gate", **kw)
        self.deps = deps
        if isinstance(acceptance, str):
            table: dict[str, type[AcceptanceCriterion]] = {
                "strict_improvement": StrictImprovementAcceptance,
                "improvement_or_equal": ImprovementOrEqualAcceptance,
            }
            self.acceptance: AcceptanceCriterion = table[acceptance]()
        else:
            self.acceptance = acceptance

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        state = self.deps.box.get()
        assert p.child is not None and p.child_eval is not None and p.parent_eval is not None

        def sub(b: Any) -> SubsampleEvaluation:
            return SubsampleEvaluation(
                scores=b.scores,
                outputs=b.outputs,
                objective_scores=list(b.objective_scores) if b.objective_scores else None,
                trajectories=b.trajectories,
            )

        proposal = CandidateProposal(
            candidate=p.child,
            parent_program_ids=[p.parent_idx],
            subsample_indices=p.minibatch_ids,
            subsample_scores_before=p.parent_eval.scores,
            subsample_scores_after=p.child_eval.scores,
            eval_before=sub(p.parent_eval),
            eval_after=sub(p.child_eval),
            tag="reflective_mutation",
            metadata=p.proposal_meta,
        )
        if p.child in state.program_candidates:
            item.tag = "duplicate"
            return [item]
        accepted = self.acceptance.should_accept(proposal, state)
        before, after = sum(p.parent_eval.scores), sum(p.child_eval.scores)
        if accepted:
            self.deps.logger.log(f"Accepted candidate (subsample score {before} -> {after}); running full eval.")
        else:
            self.deps.logger.log(f"New subsample score {after} is not better than old score {before}, skipping")
        item.tag = "accepted" if accepted else "rejected"
        return [item]


def validate_select(mode: str):
    """Selection rules for the validate place: ``fifo``, ``score`` (highest minibatch score first) or
    ``improvement`` (largest gain over the parent first)."""
    if mode == "fifo":
        from gepa.async_stages.stage import fifo

        return fifo

    def key(item: WorkItem[Proposal]) -> float:
        p = item.payload
        after = sum(p.child_eval.scores) if p.child_eval is not None else float("-inf")
        if mode == "score":
            return after
        before = sum(p.parent_eval.scores) if p.parent_eval is not None else 0.0
        return after - before

    return by_key(key)


class Commit(HeadStage[Proposal, Proposal]):
    """The pool's only writer: adds a validated candidate to the GEPAState (initializing it for the seed)."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, frontier_type: str | None = None, **kw: Any):
        super().__init__("commit", **kw)
        self.deps = deps
        self.frontier_type = frontier_type or deps.frontier_type
        self.policy = FullEvaluationPolicy()

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        assert p.child is not None and p.valset_evaluation is not None
        deps = self.deps
        if deps.box.state is None:
            state = initialize_gepa_state(
                run_dir=deps.run_dir,
                logger=deps.logger,
                seed_candidate=p.child,
                seed_valset_evaluation=p.valset_evaluation,
                track_best_outputs=deps.track_best_outputs,
                frontier_type=self.frontier_type,  # type: ignore[arg-type]
                evaluation_cache=EvaluationCache() if deps.cache_evaluation else None,
            )
            if state.evaluation_cache is not None:
                ids = list(p.valset_evaluation.scores_by_val_id)
                obj_by = p.valset_evaluation.objective_scores_by_val_id
                state.evaluation_cache.put_batch(
                    p.child,
                    ids,
                    [p.valset_evaluation.outputs_by_val_id[i] for i in ids],
                    [p.valset_evaluation.scores_by_val_id[i] for i in ids],
                    [obj_by[i] for i in ids] if obj_by else None,
                    split=VALSET_CACHE_SPLIT,
                )
            # initialize_gepa_state already counts the seed's validation in total_num_evals. It also loads a saved
            # state when run_dir holds one, in which case the pool resumes from that run.
            if len(state.program_candidates) > 1 or state.total_num_evals != p.metric_calls:
                deps.logger.log(
                    f"Resumed a saved pool from {deps.run_dir}: {len(state.program_candidates)} candidates, "
                    f"{state.total_num_evals} evaluations counted so far"
                )
            deps.box.state = state
            new_idx = 0
        else:
            state = deps.box.state
            new_idx = state.update_state_with_new_program(
                parent_program_idx=[p.parent_idx],
                new_program=p.child,
                valset_evaluation=p.valset_evaluation,
                run_dir=deps.run_dir,
                num_metric_calls_by_discovery_of_new_program=state.total_num_evals,
                iteration_id=new_iteration_id(),
            )
        score = self.policy.get_valset_score(new_idx, state)
        best = self.policy.get_best_program(state)
        deps.logger.log(f"Committed program {new_idx} (valset {score}); best is {best}")
        if deps.run_dir is not None:
            state.save(deps.run_dir, use_cloudpickle=deps.use_cloudpickle)
        if deps.stop_at_score is not None and score >= deps.stop_at_score:
            deps.request_stop("stop_at_score")
        return [item]
