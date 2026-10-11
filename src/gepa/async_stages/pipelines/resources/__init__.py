# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The resource portfolio: what the shipped pipelines declare beyond the three that come from optimize_anything
(``evals``, ``evaluator``, ``usd``; see :mod:`gepa.async_stages.resources`). A pipeline file instantiates these as
it instantiates stages, and the engine holds them for the run. Each implements the ``Resource`` protocol with its
own logic."""

from __future__ import annotations

from gepa.async_stages.resources import ResourceExhaustedError


class ReflectionLM:
    """Concurrent calls to the reflection model. Rented: a call's unit comes back when the call returns. A second
    model gets its own instance under another name (``ReflectionLM(cap=8, name="generate_lm")``)."""

    def __init__(self, cap: int = 4, *, name: str = "reflection_lm"):
        self.name = name
        self.cap = int(cap)
        self._out = 0.0

    @property
    def available(self) -> float:
        return self.cap - self._out

    def take(self, units: float) -> float:
        if units <= 0:
            return 0.0
        if units > self.cap:
            raise ResourceExhaustedError(f"{self.name}: {units} asked, cap is {self.cap}")
        if self._out + units <= self.cap:
            self._out += units
            return units
        return 0.0  # calls in flight return their units; never exhausted

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


__all__ = ["ReflectionLM"]
