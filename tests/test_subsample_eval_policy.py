import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ListDataLoader
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.eval_policy import SubsampleEvaluationPolicy, UCBEvaluationPolicy


def make_state(subscores_per_program: list[dict[int, float]]) -> GEPAState:
    """Build a GEPAState with the given per-program valset subscores."""
    base_scores = subscores_per_program[0]
    base_evaluation = ValsetEvaluation(
        outputs_by_val_id={val_id: {"out": val_id} for val_id in base_scores},
        scores_by_val_id=dict(base_scores),
        objective_scores_by_val_id=None,
    )
    state = GEPAState({"system_prompt": "seed"}, base_evaluation)
    state.num_full_ds_evals = 1
    state.total_num_evals = len(base_scores)
    for idx, scores in enumerate(subscores_per_program[1:], start=1):
        state.program_candidates.append({"system_prompt": f"prog{idx}"})
        state.prog_candidate_val_subscores.append(dict(scores))
    return state


class WeightAdapter:
    """Adapter whose candidates are integer weights; scores rise with the weight."""

    def __init__(self):
        self.propose_new_texts = self._propose_new_texts

    def evaluate(self, batch, candidate, capture_traces=False):
        weight = int(candidate["system_prompt"].split("=")[-1])
        outputs = [{"weight": weight} for _ in batch]
        scores = [min(1.0, (weight + 1) / item["difficulty"]) for item in batch]
        trajectories = [{"score": score} for score in scores] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [{"score": score} for score in eval_batch.scores]
        return dict.fromkeys(components_to_update, records)

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        weight = int(candidate["system_prompt"].split("=")[-1])
        return dict.fromkeys(components_to_update, f"weight={weight + 1}")


def run_weight_optimization(valset_size: int, max_metric_calls: int, **optimize_kwargs):
    trainset = [{"difficulty": d} for d in (2, 3, 4)]
    valset = [{"difficulty": 2 + (i % 5)} for i in range(valset_size)]
    return gepa.optimize(
        seed_candidate={"system_prompt": "weight=0"},
        trainset=trainset,
        valset=valset,
        adapter=WeightAdapter(),
        reflection_lm=None,
        max_metric_calls=max_metric_calls,
        **optimize_kwargs,
    )


class TestSubsampleEvaluationPolicy:
    def test_fractional_size_with_floor(self):
        policy = SubsampleEvaluationPolicy(0.2, min_size=32, seed=0)
        state = make_state([{0: 1.0}])
        assert len(policy.get_eval_batch(ListDataLoader([{}] * 10), state)) == 10
        assert len(SubsampleEvaluationPolicy(0.2).get_eval_batch(ListDataLoader([{}] * 32), state)) == 32
        assert len(SubsampleEvaluationPolicy(0.2).get_eval_batch(ListDataLoader([{}] * 100), state)) == 32
        assert len(SubsampleEvaluationPolicy(0.2).get_eval_batch(ListDataLoader([{}] * 500), state)) == 100

    def test_integer_size(self):
        state = make_state([{0: 1.0}])
        assert len(SubsampleEvaluationPolicy(10).get_eval_batch(ListDataLoader([{}] * 100), state)) == 10
        assert len(SubsampleEvaluationPolicy(10).get_eval_batch(ListDataLoader([{}] * 5), state)) == 5

    def test_batches_are_shared_and_nested(self):
        policy = SubsampleEvaluationPolicy(0.2, min_size=32, seed=7)
        loader = ListDataLoader([{"i": i} for i in range(100)])
        state = make_state([{0: 1.0}])

        seed_batch = policy.get_seed_eval_batch(loader)
        first = policy.get_eval_batch(loader, state)
        second = policy.get_eval_batch(loader, state)
        assert seed_batch == first == second
        assert len(first) == 32
        assert len(set(first)) == 32

        # Growing the valset appends to the order: the old prefix is preserved.
        loader.add_items([{"i": i} for i in range(100, 200)])
        grown = policy.get_eval_batch(loader, state)
        assert len(grown) == 40  # ceil(0.2 * 200)
        assert grown[: len(first)] == first

    def test_deterministic_across_instances(self):
        loader = ListDataLoader([{"i": i} for i in range(100)])
        state = make_state([{0: 1.0}])
        batch_a = SubsampleEvaluationPolicy(0.2, seed=3).get_eval_batch(loader, state)
        batch_b = SubsampleEvaluationPolicy(0.2, seed=3).get_eval_batch(loader, state)
        assert batch_a == batch_b

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            SubsampleEvaluationPolicy(0.0)
        with pytest.raises(ValueError):
            SubsampleEvaluationPolicy(1.5)
        with pytest.raises(ValueError):
            SubsampleEvaluationPolicy(0)
        with pytest.raises(ValueError):
            SubsampleEvaluationPolicy(0.2, min_size=0)

    def test_get_best_program_prefers_higher_average(self):
        state = make_state(
            [
                {0: 0.5, 1: 0.5},
                {0: 1.0, 1: 0.5},
            ]
        )
        assert SubsampleEvaluationPolicy(0.5).get_best_program(state) == 1


class TestUCBEvaluationPolicy:
    def test_explore_then_exploit_schedule(self):
        policy = UCBEvaluationPolicy(
            total_metric_calls=1000, exploration_fraction=0.7, subsample_size=0.1, min_size=16, seed=0
        )
        loader = ListDataLoader([{"i": i} for i in range(100)])
        state = make_state([{0: 1.0}])

        state.total_num_evals = 0
        assert len(policy.get_eval_batch(loader, state)) == 16

        state.total_num_evals = 699
        assert len(policy.get_eval_batch(loader, state)) == 16

        state.total_num_evals = 700
        assert len(policy.get_eval_batch(loader, state)) == 100

    def test_unknown_budget_stays_in_exploration(self):
        policy = UCBEvaluationPolicy(total_metric_calls=None, subsample_size=0.1, min_size=16)
        loader = ListDataLoader([{"i": i} for i in range(100)])
        state = make_state([{0: 1.0}])
        state.total_num_evals = 10**9
        assert len(policy.get_eval_batch(loader, state)) == 16

    def test_lower_confidence_bound_selection(self):
        # Program 0: mean 0.6 with zero variance over 50 examples -> LCB 0.6.
        # Program 1: mean 0.75 but noisy over 4 examples -> SE 0.25 -> LCB 0.5.
        # Program 2: a single (lucky) perfect example -> unbounded uncertainty.
        state = make_state(
            [
                dict.fromkeys(range(50), 0.6),
                {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                {0: 1.0},
            ]
        )
        policy = UCBEvaluationPolicy(total_metric_calls=1000, z=1.0)
        assert policy.get_best_program(state) == 0

        # The plain-average policy would have picked the noisy program instead.
        assert SubsampleEvaluationPolicy().get_best_program(state) == 2

    def test_result_best_idx_honors_policy_selection(self):
        from gepa.core.result import GEPAResult

        state = make_state(
            [
                dict.fromkeys(range(50), 0.6),
                {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                {0: 1.0},
            ]
        )
        policy = UCBEvaluationPolicy(total_metric_calls=1000, z=1.0)
        result = GEPAResult.from_state(state, val_evaluation_policy=policy)
        assert result.best_idx == 0

        # Round-trips through serialization.
        assert GEPAResult.from_dict(result.to_dict()).best_idx == 0

        # Without a policy, best_idx falls back to the raw argmax.
        assert GEPAResult.from_state(state).best_idx == 2

    def test_lcb_falls_back_to_mean_among_underexplored(self):
        state = make_state(
            [
                {0: 0.2},
                {0: 0.9},
            ]
        )
        policy = UCBEvaluationPolicy(total_metric_calls=1000, z=1.0)
        assert policy.get_best_program(state) == 1


class TestOptimizeIntegration:
    def test_default_policy_subsamples_large_valset(self, tmp_path):
        result = run_weight_optimization(
            valset_size=80,
            max_metric_calls=150,
            run_dir=str(tmp_path / "run"),
        )
        # 0.2 * 80 = 16, floored at min_size=32: every candidate (seed included)
        # is evaluated on exactly 32 shared validation examples.
        assert len(result.val_subscores) >= 2
        coverages = [len(scores) for scores in result.val_subscores]
        assert all(coverage == 32 for coverage in coverages)

        # All candidates share the same subsample ids.
        id_sets = [frozenset(scores.keys()) for scores in result.val_subscores]
        assert len(set(id_sets)) == 1

    def test_default_policy_full_eval_on_small_valset(self, tmp_path):
        result = run_weight_optimization(
            valset_size=10,
            max_metric_calls=60,
            run_dir=str(tmp_path / "run"),
        )
        assert all(len(scores) == 10 for scores in result.val_subscores)

    def test_full_eval_literal_restores_old_behavior(self, tmp_path):
        result = run_weight_optimization(
            valset_size=40,
            max_metric_calls=150,
            run_dir=str(tmp_path / "run"),
            val_evaluation_policy="full_eval",
        )
        assert len(result.val_subscores) >= 2
        assert all(len(scores) == 40 for scores in result.val_subscores)

    def test_ucb_literal_runs(self, tmp_path):
        result = run_weight_optimization(
            valset_size=60,
            max_metric_calls=200,
            run_dir=str(tmp_path / "run"),
            val_evaluation_policy="ucb",
        )
        assert len(result.val_subscores) >= 2
        coverages = [len(scores) for scores in result.val_subscores]
        # Exploration-phase candidates are evaluated on min_size=16 examples.
        assert coverages[0] == 16


class TestTrainExclusivePolicy:
    def test_excludes_all_recorded_minibatch_ids(self):
        from gepa.strategies.eval_policy import DynamicHoldoutEvaluationPolicy

        state = make_state([{0: 1.0}])
        state.full_program_trace.append({"i": 0, "subsample_ids": [1, 3]})
        # Multi-task iterations record every task's minibatch under all_subsample_ids.
        state.full_program_trace.append({"i": 1, "subsample_ids": [5], "all_subsample_ids": [[5], [7, 8]]})
        loader = ListDataLoader([{"i": i} for i in range(10)])
        policy = DynamicHoldoutEvaluationPolicy()

        assert policy.get_eval_batch(loader, state) == [0, 2, 4, 6, 9]
        assert policy.get_seed_eval_batch(loader) == list(range(10))

    def test_falls_back_to_full_pool_when_everything_trained(self):
        from gepa.strategies.eval_policy import DynamicHoldoutEvaluationPolicy

        state = make_state([{0: 1.0}])
        state.full_program_trace.append({"i": 0, "subsample_ids": list(range(10))})
        loader = ListDataLoader([{"i": i} for i in range(10)])
        policy = DynamicHoldoutEvaluationPolicy()

        assert policy.get_eval_batch(loader, state) == list(range(10))

    def test_optimize_with_combined_train_val_pool(self, tmp_path):
        # valset=None reuses the trainset loader, giving the combined pool this
        # policy is designed for.
        trainset = [{"difficulty": 2 + (i % 5)} for i in range(20)]
        result = gepa.optimize(
            seed_candidate={"system_prompt": "weight=0"},
            trainset=trainset,
            adapter=WeightAdapter(),
            reflection_lm=None,
            max_metric_calls=80,
            run_dir=str(tmp_path / "run"),
            val_evaluation_policy="dynamic_holdout",
        )

        coverages = [len(scores) for scores in result.val_subscores]
        # The seed sees the full untrained pool; every later candidate is evaluated
        # only on examples its reflection minibatches have not touched.
        assert coverages[0] == 20
        assert len(coverages) >= 2
        assert all(coverage < 20 for coverage in coverages[1:])

        # The untrained pool only shrinks, so evaluation sets are nested.
        id_sets = [set(scores.keys()) for scores in result.val_subscores]
        for earlier, later in zip(id_sets, id_sets[1:], strict=False):
            assert later <= earlier
