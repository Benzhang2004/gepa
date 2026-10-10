# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stage-decoupled asynchronous optimization: a pipeline of stages driven by an engine that allocates resources.

Registered with optimize_anything as the ``"async"`` backend (:mod:`gepa.oa.engines.async_backend`).
"""

from gepa.async_stages.engine import Engine
from gepa.async_stages.pipeline import Pipeline, PipelineError
from gepa.async_stages.resources import Evals, Evaluator, Resource, ResourceExhaustedError, Usd
from gepa.async_stages.stage import HeadStage, Stage, WorkerStage, WorkItem, by_key, fifo

__all__ = [
    "Engine",
    "Evals",
    "Evaluator",
    "HeadStage",
    "Pipeline",
    "PipelineError",
    "Resource",
    "ResourceExhaustedError",
    "Stage",
    "Usd",
    "WorkItem",
    "WorkerStage",
    "by_key",
    "fifo",
]
