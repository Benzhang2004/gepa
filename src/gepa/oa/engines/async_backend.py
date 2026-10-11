"""The ``async`` backend: a pipeline of independently scaled stages driven by :class:`gepa.async_stages.Engine`.

``OptimizeAnythingConfig.engine_config`` keys:

- ``pipeline``: ``{"name": <registered name or module path>, ...pipeline-wide defaults}`` (default ``reflective``).
- ``resources``: one entry per resource, by name, passed to its constructor. The backend provides the three that
  come from optimize_anything: ``evals`` (``{"mode": "bulk" | "passthrough", "cap": n}``; the server's budget
  tracker), ``evaluator`` (``{"cap": n}``, at most the server's ``max_concurrency``) and ``usd`` (``{"cap": n}``,
  default ``max_token_cost``). Every other entry configures a resource the chosen pipeline declares
  (``reflection_lm`` in the shipped ones, ``{"cap": n}``). A stage may not demand more of a resource than its cap.
- ``stages``: one entry per stage of the pipeline: the engine's scheduling keys (``workers``, ``max_workers``,
  ``max_queue``, ``max_in_flight`` for a read stage, ``select``, ``demand``) and the stage's own constructor
  options. The shipped pipelines bound proposals alive with ``{"sample": {"max_in_flight": n}}`` (default 8).
- ``reflection_lm``: a model name or a callable; ``seed``; ``write_event_log``; ``isolate_errors``;
  ``cache_evaluation`` (default True: a GEPAState evaluation cache, so demands are cache misses).
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gepa.async_stages.engine import Engine
from gepa.async_stages.pipelines import resolve
from gepa.async_stages.pipelines.stages import Deps, Proposal, StateBox
from gepa.async_stages.resources import Evals, Evaluator, Usd
from gepa.async_stages.stage import WorkerStage
from gepa.oa.engine import Result

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task

_STR_KEY = "current_candidate"


class AsyncBackend:
    name = "async"

    def __init__(self, config: OptimizeAnythingConfig) -> None:
        self.run_dir = config.run_dir
        self.stop_at_score = config.stop_at_score
        self.max_token_cost = config.max_token_cost
        cfg = dict(config.engine_config)
        pipeline = cfg.pop("pipeline", "reflective")
        self.pipeline_cfg: dict[str, Any] = {"name": pipeline} if isinstance(pipeline, str) else dict(pipeline)
        self.resources_cfg: dict[str, dict[str, Any]] = {k: dict(v) for k, v in cfg.pop("resources", {}).items()}
        self.stages_cfg: dict[str, dict[str, Any]] = {k: dict(v) for k, v in cfg.pop("stages", {}).items()}
        self.reflection_lm = cfg.pop("reflection_lm", None)
        self.seed = int(cfg.pop("seed", 0))
        self.write_event_log = bool(cfg.pop("write_event_log", True))
        self.isolate_errors = bool(cfg.pop("isolate_errors", False))
        self.sequential_evals = bool(cfg.pop("sequential_evals", False))
        self.frontier_type = cfg.pop("frontier_type", "hybrid")
        self.cache_evaluation = bool(cfg.pop("cache_evaluation", True))
        if cfg:
            raise TypeError(f"unknown engine_config keys for the async backend: {sorted(cfg)}")

    # ------------------------------------------------------------------

    def run(self, task: Task, server: EvalServer) -> Result:
        from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
        from gepa.core.data_loader import ensure_loader
        from gepa.core.result import GEPAResult
        from gepa.gepa_launcher import _build_reflection_prompt_template, optimize_anything_reflection_prompt_template
        from gepa.logging.logger import Logger, StdOutLogger
        from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM

        if task.seed_candidate is None:
            raise ValueError("the async backend needs a seed candidate")
        str_mode = not isinstance(task.seed_candidate, dict)
        seed: dict[str, str] = (
            {_STR_KEY: cast(str, task.seed_candidate)}
            if str_mode
            else {str(k): str(v) for k, v in cast("dict[str, str]", task.seed_candidate).items()}
        )
        train = list(task.train_set) if task.train_set else [None]
        val = list(task.val_set) if task.val_set else train
        trainset = ensure_loader(train)
        valset = trainset if val is train else ensure_loader(val)

        run_dir = self.run_dir
        if run_dir is None:
            run_dir = (
                str(server.output_dir / "async_state") if server.output_dir is not None else tempfile.mkdtemp("-async")
            )
        os.makedirs(run_dir, exist_ok=True)
        logger: Any = Logger(os.path.join(run_dir, "run_log.txt")) if run_dir else StdOutLogger()

        # Reflection LM: a model name becomes a cost-tracking LM; a callable is used as is.
        lm_callable: Any = None
        cost_source: Any = None
        if isinstance(self.reflection_lm, str):
            from gepa.lm import LM

            lm_callable = LM(self.reflection_lm)
            cost_source = lm_callable
        elif self.reflection_lm is not None:
            lm_callable = self.reflection_lm
            cost_source = self.reflection_lm if hasattr(self.reflection_lm, "total_cost") else None
        # Same rule as the launcher: objective/background build a template; neither means the default one.
        if task.objective or task.background:
            template: str = _build_reflection_prompt_template(task.objective or None, task.background or None)
        else:
            template = optimize_anything_reflection_prompt_template
        reflection_lm = StatelessReflectionLM(lm_callable, template, logger) if lm_callable is not None else None

        def evaluator(candidate: Any, example: Any = None, **kw: Any) -> tuple[float, Any, dict[str, Any]]:
            score, info = server.evaluate(candidate, example, **kw)
            return score, None, info

        adapter = OptimizeAnythingAdapter(
            evaluator=evaluator,
            objective=task.objective or None,
            background=task.background or None,
            cache_mode="off",
            parallel=False,
        )

        box = StateBox()
        engine_ref: dict[str, Engine] = {}
        deps = Deps(
            box=box,
            server=server,
            adapter=adapter,
            reflection_lm=reflection_lm,
            trainset=trainset,
            valset=valset,
            rng=random.Random(self.seed),
            logger=logger,
            run_dir=run_dir,
            seed_candidate=seed,
            request_stop=lambda reason: engine_ref["engine"].request_stop(reason),
            str_candidate_key=_STR_KEY if str_mode else None,
            stop_at_score=self.stop_at_score,
            frontier_type=self.frontier_type,
            isolate_errors=self.isolate_errors,
            sequential_evals=self.sequential_evals,
            lm_cost_source=cost_source,
            cache_evaluation=self.cache_evaluation,
            evaluator_cap=min(
                server.max_concurrency,
                int(self.resources_cfg.get("evaluator", {}).get("cap", server.max_concurrency)),
            ),
            objective=task.objective or "",
            background=task.background or "",
        )

        # The resources every pipeline shares, from optimize_anything; the pipeline declares the rest.
        evals_cfg = self.resources_cfg.get("evals", {})
        evaluator_cap = min(
            server.max_concurrency, int(self.resources_cfg.get("evaluator", {}).get("cap", server.max_concurrency))
        )
        shared: dict[str, Any] = {
            "evals": Evals(server.budget, mode=evals_cfg.get("mode", "bulk"), cap=evals_cfg.get("cap")),
            "evaluator": Evaluator(evaluator_cap),
            "usd": Usd(self.resources_cfg.get("usd", {}).get("cap", self.max_token_cost)),
        }
        resource_opts = {k: v for k, v in self.resources_cfg.items() if k not in shared}

        build = resolve(self.pipeline_cfg["name"])
        pipeline_defaults = {k: v for k, v in self.pipeline_cfg.items() if k != "name"}
        pipeline = build(self.stages_cfg, resource_opts, pipeline_defaults, deps)
        pipeline.validate(
            stage_names_in_config=list(self.stages_cfg),
            resource_names_in_config=list(resource_opts),
            shared=list(shared),
        )
        resources = {**shared, **pipeline.resources}
        # One thread pool per bandwidth, as wide as the stages that draw on it can be in flight.
        executors: dict[str, int] = {}
        for stage in pipeline.stages.values():
            if isinstance(stage, WorkerStage) and stage.bandwidth:
                executors[stage.bandwidth] = executors.get(stage.bandwidth, 0) + max(1, stage.workers)

        engine = Engine(
            pipeline,
            resources,
            executors=executors,
            run_dir=run_dir,
            write_event_log=self.write_event_log,
            raise_on_exception=not self.isolate_errors,
            gap_watch="commit" if "commit" in pipeline.stages else None,
        )
        engine_ref["engine"] = engine
        deps.counters = engine.counters
        # The seed's validation is an evaluation like any other: it enters at validate and commit seeds the pool.
        engine.inject("validate", Proposal(parent_idx=-1, parent=seed, child=seed, is_seed=True))
        if isinstance(logger, Logger):
            with logger:
                stats = engine.run()
        else:
            stats = engine.run()

        state = box.state
        adapter_cost = float(getattr(cost_source, "total_cost", 0.0) or 0.0)
        if state is None:
            return Result(
                best_candidate=cast(str, server.best_candidate),
                best_score=server.best_score,
                total_evals=server.budget.used,
                eval_log=server.eval_log,
                metadata={"async": stats, "adapter_cost": adapter_cost},
            )
        gepa_result = GEPAResult.from_state(
            state, run_dir=run_dir, seed=self.seed, str_candidate_key=_STR_KEY if str_mode else None
        )
        best: Any = gepa_result.best_candidate
        if isinstance(best, dict) and str_mode:
            best = best.get(_STR_KEY, "")
        return Result(
            best_candidate=cast(str, best),
            best_score=gepa_result.val_aggregate_scores[gepa_result.best_idx],
            total_evals=server.budget.used,
            eval_log=server.eval_log,
            metadata={"gepa_result": gepa_result, "async": stats, "adapter_cost": adapter_cost},
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        return


__all__ = ["AsyncBackend"]
