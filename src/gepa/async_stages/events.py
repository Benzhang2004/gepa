# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Append-only event log and run statistics. Written by the engine thread only."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class StageStats:
    fired: int = 0
    completed: int = 0
    failed: int = 0
    busy_seconds: float = 0.0
    wait_seconds: float = 0.0
    max_queue_depth: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "completed": self.completed,
            "failed": self.failed,
            "busy_seconds": round(self.busy_seconds, 4),
            "mean_service_seconds": round(self.busy_seconds / max(self.completed, 1), 4),
            "mean_wait_seconds": round(self.wait_seconds / max(self.completed, 1), 4),
            "max_queue_depth": self.max_queue_depth,
        }


@dataclass
class ResourceStats:
    taken: float = 0.0
    put_back: float = 0.0
    used: float = 0.0
    refused: int = 0
    _busy_seconds: float = 0.0
    _last: float = field(default_factory=time.monotonic)

    def as_dict(self, wall: float) -> dict[str, Any]:
        return {
            "taken": self.taken,
            "put_back": self.put_back,
            "used": self.used,
            "refused": self.refused,
            "mean_in_flight": round(self._busy_seconds / max(wall, 1e-9), 3),
        }


class EventLog:
    def __init__(self, run_dir: str | None, enabled: bool = True):
        self._t0 = time.monotonic()
        self._fh: TextIO | None = None
        if enabled and run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            self._fh = open(os.path.join(run_dir, "async_events.jsonl"), "a", encoding="utf-8")
        self.stages: dict[str, StageStats] = {}
        self.resources: dict[str, ResourceStats] = {}
        self.outcomes: Counter[str] = Counter()
        self.gaps: dict[str, Counter[int]] = {}
        self.stop_reason: str | None = None
        self.stranded: dict[str, float] = {}
        self._last_tick = self._t0
        self._inflight_seconds = 0.0

    def now(self) -> float:
        return time.monotonic() - self._t0

    def stage(self, name: str) -> StageStats:
        return self.stages.setdefault(name, StageStats())

    def resource(self, name: str) -> ResourceStats:
        return self.resources.setdefault(name, ResourceStats())

    def tick(self, inflight_by_stage: dict[str, int], inflight_by_resource: dict[str, float]) -> None:
        t = time.monotonic()
        dt = t - self._last_tick
        self._last_tick = t
        self._inflight_seconds += dt * sum(inflight_by_stage.values())
        for name, n in inflight_by_resource.items():
            self.resource(name)._busy_seconds += dt * n

    def emit(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps({"t": round(self.now(), 4), "kind": kind, **fields}, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def summary(self) -> dict[str, Any]:
        wall = max(self.now(), 1e-9)
        return {
            "wall_seconds": round(wall, 4),
            "stop_reason": self.stop_reason,
            "mean_items_in_flight": round(self._inflight_seconds / wall, 3),
            "outcomes": dict(self.outcomes),
            "gaps": {stage: {str(k): v for k, v in sorted(c.items())} for stage, c in self.gaps.items()},
            "stranded": dict(self.stranded),
            "stages": {name: s.as_dict() for name, s in self.stages.items()},
            "resources": {name: r.as_dict(wall) for name, r in self.resources.items()},
        }
