# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The resource portfolio: what the shipped pipelines declare beyond the three that come from optimize_anything
(``evals``, ``evaluator``, ``usd``; see :mod:`gepa.async_stages.resources`). A pipeline file instantiates these as
it instantiates stages, and the engine holds them for the run. Each implements the ``Resource`` protocol with its
own logic."""

from __future__ import annotations


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
        want = min(units, self.cap)
        if self._out + want <= self.cap:
            self._out += want
            return want
        return 0.0  # calls in flight return their units; never exhausted

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


class ProposalsInFlight:
    """Proposals alive at once. The sample stage takes one unit per proposal and holds it to the end of the
    proposal, so at most ``cap`` are between sampling and commit and every draw reads the pool as it then is:
    the smoother form of the generation cap (a slot frees as each proposal ends, not at a commit)."""

    def __init__(self, cap: int = 8, *, name: str = "proposals_in_flight"):
        self.name = name
        self.cap = int(cap)
        self._out = 0.0

    @property
    def available(self) -> float:
        return self.cap - self._out

    def take(self, units: float) -> float:
        if units <= 0 or self._out + units > self.cap:
            return 0.0  # a proposal ending frees its slot; never exhausted
        self._out += units
        return units

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


__all__ = ["ProposalsInFlight", "ReflectionLM"]
