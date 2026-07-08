"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

import math
import random
from abc import abstractmethod
from typing import Protocol, runtime_checkable

from gepa.core.data_loader import DataId, DataInst, DataLoader
from gepa.core.state import GEPAState, ProgramIdx


@runtime_checkable
class EvaluationPolicy(Protocol[DataId, DataInst]):  # type: ignore
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances.

    Implementations may additionally define ``get_seed_eval_batch(loader) -> list[DataId]``
    to control which validation ids the seed candidate is evaluated on before the
    optimization state exists. When absent, the engine evaluates the seed on the
    full valset. It is intentionally not part of the protocol so existing
    structural implementations remain valid.
    """

    @abstractmethod
    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Select examples for evaluation for a program"""
        ...

    @abstractmethod
    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        ...


def _best_program_by_average(state: GEPAState) -> ProgramIdx:
    """Pick the program whose evaluated validation scores achieve the highest average.

    Ties are broken in favor of the program with more evaluated validation examples.
    """
    best_idx, best_score, best_coverage = -1, float("-inf"), -1
    for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
        coverage = len(scores)
        avg = sum(scores.values()) / coverage if coverage else float("-inf")
        if avg > best_score or (avg == best_score and coverage > best_coverage):
            best_score = avg
            best_idx = program_idx
            best_coverage = coverage
    return best_idx


def _standard_error(scores: list[float]) -> float:
    """Standard error of the mean; infinite when fewer than two samples exist."""
    n = len(scores)
    if n < 2:
        return float("inf")
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return math.sqrt(variance / n)


class FullEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Policy that evaluates all validation instances every time."""

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        return _best_program_by_average(state)

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


class SubsampleEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Evaluate every candidate on a shared random subsample of the valset (#103).

    The policy keeps one seeded shuffle of the validation ids and evaluates each
    candidate — the seed candidate included — on a prefix of that order. Sharing
    the prefix across candidates means:

    - each candidate's average is an unbiased estimate of its full-valset average,
      because the prefix is a uniform random sample of the valset;
    - candidates are compared on the same examples (common random numbers), which
      makes paired comparisons far less noisy than disjoint subsamples; and
    - per-instance Pareto fronts stay balanced: no candidate accumulates
      uncontested front entries on ids that other candidates were never scored on,
      which would otherwise skew Pareto-based parent sampling.

    ``subsample_size`` is either an int (absolute number of validation examples per
    evaluation) or a float in (0, 1] (fraction of the current valset). With a
    fractional size, valsets with at most ``min_size`` examples are evaluated in
    full and larger valsets never use fewer than ``min_size`` examples, so
    subsampling only engages where it actually saves budget.

    If the valset grows mid-run, new ids are shuffled and appended to the order, so
    earlier prefixes remain prefixes and previously evaluated candidates stay
    comparable.
    """

    def __init__(self, subsample_size: int | float = 0.2, *, min_size: int = 32, seed: int = 0):
        if isinstance(subsample_size, float):
            if not 0.0 < subsample_size <= 1.0:
                raise ValueError(f"Fractional subsample_size must be in (0, 1], got {subsample_size}")
        elif subsample_size < 1:
            raise ValueError(f"Integer subsample_size must be >= 1, got {subsample_size}")
        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}")
        self.subsample_size = subsample_size
        self.min_size = min_size
        self._rng = random.Random(seed)
        self._order: list[DataId] = []
        self._known_ids: set[DataId] = set()

    def _eval_size(self, num_ids: int) -> int:
        if isinstance(self.subsample_size, int):
            return min(self.subsample_size, num_ids)
        if num_ids <= self.min_size:
            return num_ids
        return max(self.min_size, math.ceil(self.subsample_size * num_ids))

    def _ordered_ids(self, loader: DataLoader[DataId, DataInst]) -> list[DataId]:
        """Return all current ids in the stable shuffled order, appending new ids."""
        all_ids = list(loader.all_ids())
        new_ids = [val_id for val_id in all_ids if val_id not in self._known_ids]
        if new_ids:
            self._rng.shuffle(new_ids)
            self._order.extend(new_ids)
            self._known_ids.update(new_ids)
        if len(self._order) != len(all_ids):
            current = set(all_ids)
            return [val_id for val_id in self._order if val_id in current]
        return list(self._order)

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Return the shared subsample prefix for the current valset."""
        ordered = self._ordered_ids(loader)
        return ordered[: self._eval_size(len(ordered))]

    def get_seed_eval_batch(self, loader: DataLoader[DataId, DataInst]) -> list[DataId]:
        """Evaluate the seed candidate on the same shared prefix as later candidates."""
        ordered = self._ordered_ids(loader)
        return ordered[: self._eval_size(len(ordered))]

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        return _best_program_by_average(state)

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the average score of the program over its evaluated validation ids."""
        return state.get_program_average_val_subset(program_idx)[0]


class UCBEvaluationPolicy(SubsampleEvaluationPolicy):
    """Experimental explore/exploit evaluation scheduler with confidence-aware selection (#34).

    Splits the metric-call budget into two phases:

    - **Exploration** — while less than ``exploration_fraction`` of
      ``total_metric_calls`` has been spent, candidates are evaluated on a small
      shared subsample (``subsample_size``, floored at ``min_size``). Error bars
      are wide, but many candidates can be tried cheaply.
    - **Exploitation** — once the threshold is crossed, every new candidate is
      evaluated on the full valset, shifting the remaining budget toward
      high-confidence measurements of late (typically strongest) candidates.

    ``get_best_program`` ranks candidates by a lower confidence bound
    ``mean - z * standard_error`` instead of the raw mean, so a sparsely evaluated
    candidate only wins if its average beats well-measured candidates even after
    the uncertainty penalty. Candidates with fewer than two evaluated examples have
    unbounded uncertainty and are ranked by mean only after all finite lower bounds.

    When ``total_metric_calls`` is unknown (``None``), the policy stays in the
    exploration regime and relies solely on the conservative selection rule.
    """

    def __init__(
        self,
        total_metric_calls: int | None = None,
        *,
        exploration_fraction: float = 0.7,
        subsample_size: int | float = 0.1,
        min_size: int = 16,
        z: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(subsample_size, min_size=min_size, seed=seed)
        if not 0.0 <= exploration_fraction <= 1.0:
            raise ValueError(f"exploration_fraction must be in [0, 1], got {exploration_fraction}")
        if z < 0.0:
            raise ValueError(f"z must be >= 0, got {z}")
        self.total_metric_calls = total_metric_calls
        self.exploration_fraction = exploration_fraction
        self.z = z

    def _in_exploitation(self, state: GEPAState) -> bool:
        if self.total_metric_calls is None:
            return False
        return state.total_num_evals >= self.exploration_fraction * self.total_metric_calls

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Small shared subsample during exploration, full valset during exploitation."""
        ordered = self._ordered_ids(loader)
        if self._in_exploitation(state):
            return ordered
        return ordered[: self._eval_size(len(ordered))]

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Rank candidates by the lower confidence bound of their average val score."""
        best_idx = -1
        best_key = (float("-inf"), float("-inf"), -1)
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            if coverage == 0:
                continue
            mean = sum(scores.values()) / coverage
            standard_error = _standard_error(list(scores.values()))
            if math.isfinite(standard_error):
                lcb = mean - self.z * standard_error
            else:
                lcb = mean if self.z == 0.0 else float("-inf")
            key = (lcb, mean, coverage)
            if key > best_key:
                best_key = key
                best_idx = program_idx
        if best_idx < 0:
            return _best_program_by_average(state)
        return best_idx


class DynamicHoldoutEvaluationPolicy(EvaluationPolicy[DataId, DataInst]):
    """Evaluate each candidate on every example not yet used for training (#125-style pooling).

    Designed for the combined train+val workflow where the same examples — with the
    same ids — serve as both trainset and valset (e.g. ``valset=None`` in
    :func:`gepa.optimize`, which reuses the train loader). Each candidate is scored
    on exactly the examples that have not appeared in any reflection minibatch so
    far, so validation always measures behavior on data the reflective proposer has
    not exploited yet, even though every example is eventually trained on.

    The trained set is recovered from ``state.full_program_trace``, where the
    reflective proposer records each iteration's minibatch ids (including rejected
    proposals — reflection saw those examples too), so it survives run resumption.
    The seed candidate is evaluated on the full pool (nothing is trained yet), and
    once every example has been trained on the policy falls back to full-pool
    evaluation, since no uncontaminated example remains.

    Caveats: candidates are scored on different (shrinking) evaluation sets, so
    aggregate scores are unbiased population estimates only while minibatch
    sampling is random; and the trainset and valset must share the same id space
    for the exclusion to be meaningful.
    """

    def get_eval_batch(
        self, loader: DataLoader[DataId, DataInst], state: GEPAState, target_program_idx: ProgramIdx | None = None
    ) -> list[DataId]:
        """Return all ids not yet seen in a reflection minibatch, or all ids once none remain."""
        trained = self._trained_ids(state)
        all_ids = list(loader.all_ids())
        untrained = [val_id for val_id in all_ids if val_id not in trained]
        return untrained if untrained else all_ids

    def get_seed_eval_batch(self, loader: DataLoader[DataId, DataInst]) -> list[DataId]:
        """The seed is evaluated before any training, so the whole pool is untrained."""
        return list(loader.all_ids())

    @staticmethod
    def _trained_ids(state: GEPAState) -> set:
        """Union of all reflection-minibatch ids recorded in the optimization trace."""
        trained: set = set()
        for trace_entry in state.full_program_trace:
            ids = trace_entry.get("subsample_ids")
            if ids:
                trained.update(ids)
            for task_ids in trace_entry.get("all_subsample_ids") or []:
                trained.update(task_ids)
        return trained

    def get_best_program(self, state: GEPAState) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        return _best_program_by_average(state)

    def get_valset_score(self, program_idx: ProgramIdx, state: GEPAState) -> float:
        """Return the average score of the program over its evaluated validation ids."""
        return state.get_program_average_val_subset(program_idx)[0]


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
    "SubsampleEvaluationPolicy",
    "DynamicHoldoutEvaluationPolicy",
    "UCBEvaluationPolicy",
]
