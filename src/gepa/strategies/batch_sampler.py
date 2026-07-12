# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from typing import Protocol

from gepa.core.adapter import DataInst
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState


class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]: ...


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """
    Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None):
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.last_trainset_size = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]):
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        base_idx = state.i * self.minibatch_size
        curr_epoch = 0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)

        needs_refresh = not self.shuffled_ids or trainset_size != self.last_trainset_size or curr_epoch > self.epoch
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]


class WorstFirstBatchSampler(BatchSampler[DataId, DataInst]):
    """Sample reflection minibatches biased toward the worst-performing examples (#34).

    Designed for the combined train+val pool workflow (trainset and valset share
    ids, e.g. ``valset=None``): each example's signal is the best score any
    candidate has achieved on it so far (``state.pareto_front_valset``). Examples
    where even the best candidate scores low have the most remaining headroom, so
    reflection trains on them first — the score-deficit counterpart to weighting
    by cross-candidate variance (PR #338's LearnabilityBatchSampler).

    Each example is weighted by its deficit ``best_pool_score - best_score``;
    ids with no recorded score yet get the largest observed deficit (unexplored,
    assume the worst). ``temperature`` controls concentration:

    - ``0.0`` — deterministic: the ``minibatch_size`` worst examples, tie-broken
      toward the least-sampled so static scores still rotate through ties
    - ``1.0`` (default) — sample proportional to deficit
    - ``> 1`` — flatten toward uniform (more easy examples mixed in)
    - ``< 1`` — sharpen toward the very worst

    ``uniform_mix`` spreads that fraction of probability mass uniformly, so
    solved examples (deficit 0) keep a nonzero chance of being revisited and the
    optimizer does not drift away from them. While no example has a recorded
    score (or all scores are equal), sampling is uniform without replacement.
    """

    def __init__(
        self,
        minibatch_size: int,
        temperature: float = 1.0,
        uniform_mix: float = 0.1,
        rng: random.Random | None = None,
    ):
        if minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError(f"uniform_mix must be in [0, 1], got {uniform_mix}")
        self.minibatch_size = minibatch_size
        self.temperature = temperature
        self.uniform_mix = uniform_mix
        self.rng = rng if rng is not None else random.Random(0)
        self._sample_counts: Counter = Counter()

    def _deficits(self, all_ids: list[DataId], state: GEPAState) -> dict[DataId, float] | None:
        """Per-id score deficit vs. the best-scoring example; None if uninformative."""
        front = state.pareto_front_valset
        best_scores = {data_id: front[data_id] for data_id in all_ids if data_id in front}
        if not best_scores:
            return None
        s_max = max(best_scores.values())
        s_min = min(best_scores.values())
        if s_max <= s_min:
            return None
        # Ids never evaluated default to the observed minimum: unexplored, assume the worst.
        return {data_id: s_max - best_scores.get(data_id, s_min) for data_id in all_ids}

    def next_minibatch_ids(self, loader: DataLoader[DataId, DataInst], state: GEPAState) -> list[DataId]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample a minibatch from an empty loader.")
        k = min(self.minibatch_size, len(all_ids))

        deficits = self._deficits(all_ids, state)
        if deficits is None:
            batch = self.rng.sample(all_ids, k)
        elif self.temperature == 0.0:
            order = {data_id: position for position, data_id in enumerate(all_ids)}
            ranked = sorted(all_ids, key=lambda i: (-deficits[i], self._sample_counts[i], order[i]))
            batch = ranked[:k]
        else:
            weights = {data_id: deficit ** (1.0 / self.temperature) for data_id, deficit in deficits.items()}
            total = sum(weights.values())
            n = len(all_ids)
            probs = {
                data_id: (1.0 - self.uniform_mix) * (weight / total) + self.uniform_mix / n
                for data_id, weight in weights.items()
            }
            remaining = list(all_ids)
            batch: list[DataId] = []
            for _ in range(k):
                candidate_probs = [probs[data_id] for data_id in remaining]
                chosen = self.rng.choices(remaining, weights=candidate_probs, k=1)[0]
                batch.append(chosen)
                remaining.remove(chosen)

        self._sample_counts.update(batch)
        return batch
