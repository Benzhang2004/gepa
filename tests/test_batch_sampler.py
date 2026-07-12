import random
from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


def test_epoch_sampler_refreshes_when_loader_expands():
    loader = ListDataLoader(["a", "b", "c", "d"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    first_batch = sampler.next_minibatch_ids(loader, state)
    assert len(first_batch) == 2
    assert len(sampler.shuffled_ids) == 4
    assert sampler.last_trainset_size == 4

    state.i += 1
    loader.add_items(["e", "f"])

    second_batch = sampler.next_minibatch_ids(loader, state)
    assert len(second_batch) == 2
    assert sampler.last_trainset_size == 6
    assert len(sampler.shuffled_ids) == 6
    assert {4, 5}.issubset(set(sampler.shuffled_ids))


def test_epoch_sampler_errors_when_loader_empty():
    loader = ListDataLoader([])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, state)


def make_scored_state(best_scores: dict[int, float]) -> SimpleNamespace:
    """State stub exposing the per-example Pareto-front scores the sampler reads."""
    return SimpleNamespace(i=0, pareto_front_valset=dict(best_scores))


class TestWorstFirstBatchSampler:
    def test_temperature_zero_picks_worst_examples(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c", "d", "e"])
        state = make_scored_state({0: 1.0, 1: 0.2, 2: 0.9, 3: 0.1, 4: 0.5})
        sampler = WorstFirstBatchSampler(minibatch_size=2, temperature=0.0)

        assert sampler.next_minibatch_ids(loader, state) == [3, 1]

    def test_temperature_zero_rotates_ties_by_sample_count(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c"])
        state = make_scored_state({0: 0.0, 1: 0.0, 2: 1.0})
        sampler = WorstFirstBatchSampler(minibatch_size=1, temperature=0.0)

        first = sampler.next_minibatch_ids(loader, state)
        second = sampler.next_minibatch_ids(loader, state)
        assert first == [0]
        assert second == [1]  # least-sampled tie-break, not the same id again

    def test_unknown_ids_treated_as_worst(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c", "d"])
        # id 3 has never been evaluated; ids 0-2 have scores.
        state = make_scored_state({0: 1.0, 1: 0.6, 2: 0.4})
        sampler = WorstFirstBatchSampler(minibatch_size=2, temperature=0.0)

        batch = sampler.next_minibatch_ids(loader, state)
        assert set(batch) == {2, 3}

    def test_uniform_fallback_without_scores(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c", "d"])
        state = make_scored_state({})
        sampler = WorstFirstBatchSampler(minibatch_size=3, rng=random.Random(0))

        batch = sampler.next_minibatch_ids(loader, state)
        assert len(batch) == 3
        assert len(set(batch)) == 3

    def test_uniform_fallback_when_all_scores_equal(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c"])
        state = make_scored_state({0: 0.5, 1: 0.5, 2: 0.5})
        sampler = WorstFirstBatchSampler(minibatch_size=2, rng=random.Random(0))

        batch = sampler.next_minibatch_ids(loader, state)
        assert len(set(batch)) == 2

    def test_sampling_skews_toward_deficit_but_keeps_uniform_floor(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        loader = ListDataLoader(["a", "b", "c"])
        # id 0 solved (deficit 0), id 2 worst.
        state = make_scored_state({0: 1.0, 1: 0.5, 2: 0.0})
        sampler = WorstFirstBatchSampler(minibatch_size=1, temperature=1.0, uniform_mix=0.1, rng=random.Random(0))

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(2000):
            counts[sampler.next_minibatch_ids(loader, state)[0]] += 1

        assert counts[2] > counts[1] > counts[0]
        assert counts[0] > 0  # uniform_mix keeps solved examples in rotation

    def test_invalid_parameters(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        with pytest.raises(ValueError):
            WorstFirstBatchSampler(minibatch_size=0)
        with pytest.raises(ValueError):
            WorstFirstBatchSampler(minibatch_size=3, temperature=-0.1)
        with pytest.raises(ValueError):
            WorstFirstBatchSampler(minibatch_size=3, uniform_mix=1.5)

    def test_empty_loader_raises(self):
        from gepa.strategies.batch_sampler import WorstFirstBatchSampler

        with pytest.raises(ValueError):
            WorstFirstBatchSampler(minibatch_size=2).next_minibatch_ids(ListDataLoader([]), make_scored_state({}))


def _run_pool_optimization(tmp_path, train_batches, **kwargs):
    import gepa
    from gepa.core.adapter import EvaluationBatch

    class RecordingAdapter:
        def __init__(self):
            self.propose_new_texts = self._propose_new_texts

        def evaluate(self, batch, candidate, capture_traces=False):
            weight = int(candidate["system_prompt"].split("=")[-1])
            scores = [min(1.0, (weight + 1) / item["difficulty"]) for item in batch]
            if capture_traces:
                train_batches.append([item["difficulty"] for item in batch])
            trajectories = [{"score": s} for s in scores] if capture_traces else None
            return EvaluationBatch(outputs=[{}] * len(batch), scores=scores, trajectories=trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return dict.fromkeys(components_to_update, [{"score": s} for s in eval_batch.scores])

        def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            weight = int(candidate["system_prompt"].split("=")[-1])
            return dict.fromkeys(components_to_update, f"weight={weight + 1}")

    # Difficulties 2..11; the seed (weight=0) scores 1/difficulty, so the hardest
    # examples are difficulties 11, 10, 9.
    pool = [{"difficulty": d} for d in range(2, 12)]
    return gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=pool,
        adapter=RecordingAdapter(),
        reflection_lm=None,
        max_metric_calls=40,
        run_dir=str(tmp_path / "run"),
        val_evaluation_policy="dynamic_holdout",
        seed=0,
        **kwargs,
    )


def test_worst_first_trains_hardest_pool_examples_first(tmp_path):
    """Combined pool + dynamic_holdout + deterministic worst-first sampling.

    The seed's full-pool evaluation populates per-example best scores; the first
    reflection minibatch must then be exactly the k worst-scoring examples.
    """
    from gepa.strategies.batch_sampler import WorstFirstBatchSampler

    train_batches = []
    _run_pool_optimization(
        tmp_path,
        train_batches,
        batch_sampler=WorstFirstBatchSampler(minibatch_size=3, temperature=0.0),
    )

    assert train_batches, "no reflection minibatch was sampled"
    assert train_batches[0] == [11, 10, 9]


def test_worst_first_string_literal_wiring(tmp_path):
    train_batches = []
    _run_pool_optimization(
        tmp_path,
        train_batches,
        batch_sampler="worst_first",
        reflection_minibatch_size=3,
    )

    # train_batches also records child and batched-val evaluations (the batch
    # evaluate path captures traces for all of them); the first entry is always
    # the first reflection minibatch.
    assert train_batches
    assert len(train_batches[0]) == 3
