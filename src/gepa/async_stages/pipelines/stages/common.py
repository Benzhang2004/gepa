# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The payload the shipped pipelines pass between stages, and what the stages share."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM


@dataclass
class Proposal:
    """One proposal, filled in as it travels: parent and minibatch at sample, rollouts after rollout, child after
    propose, screen scores after screen, validation scores after validate."""

    parent_idx: int
    parent: dict[str, str]
    minibatch_ids: list[Any] = field(default_factory=list)
    minibatch: list[Any] = field(default_factory=list)
    parent_eval: EvaluationBatch | None = None
    components: list[str] | None = None
    child: dict[str, str] | None = None
    proposal_meta: dict[str, Any] = field(default_factory=dict)
    child_eval: EvaluationBatch | None = None
    #: validation bookkeeping
    val_ids: list[Any] = field(default_factory=list)
    val_cached: dict[Any, Any] = field(default_factory=dict)
    val_todo: list[Any] = field(default_factory=list)
    val_results: list[tuple[float, dict[str, Any]]] | None = None
    valset_evaluation: ValsetEvaluation | None = None
    #: evaluations this proposal has spent so far (for the state's discovery count)
    metric_calls: int = 0
    is_seed: bool = False
    #: scratch for the evaluation stages: the (candidate, example) pairs the body sent and what came back
    pending_ids: list[Any] = field(default_factory=list)
    results: list[tuple[float, dict[str, Any]]] | None = None


class StateBox:
    """Holds the pipeline's GEPAState; empty until the seed is committed."""

    def __init__(self, state: GEPAState | None = None):
        self.state = state

    def get(self) -> GEPAState:
        if self.state is None:
            raise RuntimeError("the pool is empty: the seed has not been committed yet")
        return self.state


@dataclass
class Deps:
    """What the shipped stages need; the backend builds it and hands it to the pipeline's ``build``."""

    box: StateBox
    server: Any  # gepa.oa.eval_server.EvalServer
    adapter: GEPAAdapter  # make_reflective_dataset only
    reflection_lm: ReflectionLM | None
    trainset: DataLoader
    valset: DataLoader
    rng: random.Random
    logger: LoggerProtocol
    run_dir: str | None
    seed_candidate: dict[str, str]
    request_stop: Callable[[str], None]
    #: per-stage completion counters, set by the backend from the engine (the staleness guard reads them)
    counters: Mapping[str, int] = field(default_factory=dict)
    str_candidate_key: str | None = None
    stop_at_score: float | None = None
    frontier_type: str = "hybrid"
    track_best_outputs: bool = True
    use_cloudpickle: bool = True
    isolate_errors: bool = False
    sequential_evals: bool = False
    lm_cost_source: Any = None  # object with total_cost, for adapter_cost
    cache_evaluation: bool = True

    def to_server_candidate(self, candidate: dict[str, str]) -> Any:
        """A candidate as the eval server expects it: the bare text in str-candidate mode."""
        if self.str_candidate_key is not None:
            return candidate[self.str_candidate_key]
        return candidate
