# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Resources: what the engine takes from at dispatch and puts back at completion.

A resource is a named supply of fungible units. A pipeline declares the resources its stages draw on (as it
declares its stages); the engine holds them for the run and is the only caller: it takes a stage's demand, all
or nothing, and puts back demand minus what the stage reported used. Whether a resource is a budget (drained:
the stage reports what it used) or a bandwidth (rented: nothing is reported used, everything comes back) is
decided by the stages that draw on it. Each resource implements the protocol with its own logic; there is no
generic pool. A resource whose supply depends on time (a per-minute quota) does its time check inside ``take``.

``take`` returns the whole demand, ``0`` for "not now", and raises :class:`ResourceExhaustedError` for "never
again" (a demand above the cap is one such case; the engine refuses it before the run when it is declared): the
resource is the one that knows which, since it sees whether anything of it is still out and could come back.

This module holds the protocol and the three resources that come from optimize_anything: ``evals`` (the eval
budget, the one resource the engine prescribes), ``evaluator`` (the server's concurrency) and ``usd``
(``max_token_cost``). The pipeline-level ones ship in :mod:`gepa.async_stages.pipelines.resources`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class ResourceExhaustedError(Exception):
    """This resource can never satisfy a demand again."""


@runtime_checkable
class Resource(Protocol):
    name: str

    @property
    def available(self) -> float | None: ...  # None = unlimited

    def take(self, units: float) -> float: ...  # units taken; 0 = not now

    def put(self, units: float, used: float = 0.0) -> None: ...

    def open(self) -> None: ...  # run start

    def close(self) -> None: ...  # run end


class Evaluator:
    """Concurrent evaluations: the eval server's ``max_concurrency``, mirrored so the engine never dispatches an
    evaluation that would only block a thread. Rented: everything comes back at completion. A demand above the
    cap can never be met and raises; a stage that evaluates more examples than the cap declares the cap and the
    server serializes inside it."""

    name = "evaluator"

    def __init__(self, cap: int, *, name: str = "evaluator"):
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
        return 0.0  # the units out come back; never exhausted

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


class Usd:
    """LLM spend under optimize_anything's ``max_token_cost``: a budget drained by what the LLM stages report used.
    ``cap=None`` is unbounded. Exhausted when a demand cannot be met and nothing is out that could come back."""

    name = "usd"

    def __init__(self, cap: float | None, *, name: str = "usd"):
        self.name = name
        self.cap = None if cap is None else float(cap)
        self._available = float("inf") if cap is None else float(cap)
        self._out = 0.0

    @property
    def available(self) -> float | None:
        return None if self.cap is None else self._available

    def take(self, units: float) -> float:
        if units <= 0:
            return 0.0
        if self.cap is None:
            self._out += units
            return units
        if units > self.cap:
            raise ResourceExhaustedError(f"{self.name}: {units} asked, cap is {self.cap}")
        if units <= self._available:
            self._available -= units
            self._out += units
            return units
        if self._out == 0:
            raise ResourceExhaustedError(f"{self.name}: {units} asked, {self._available} left and nothing in flight")
        return 0.0

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)
        if self.cap is not None:
            self._available = min(self.cap, self._available + units)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


class Evals:
    """The eval budget: optimize_anything's ``BudgetTracker``, a reservation ledger (``reserve`` / ``commit`` /
    ``release``). The one resource the engine prescribes, since every backend answers to the same budget.

    ``bulk`` (the default): ``open`` reserves, before anything runs, the budget the entire run needs (the ``cap``
    the user set, or everything the tracker still has), so no backend sharing the server can race this engine
    for it; take and put are then local against that lease, the server books each evaluation that runs against
    the reservation (the evaluation stages call it with ``reserved=True``), and ``close`` releases what was
    leased but never run. ``passthrough``: every take is forwarded as a reserve and every put as a release of the
    unused part, as they happen, for a server shared with engines running at the same time.
    """

    name = "evals"

    def __init__(self, tracker: Any, *, mode: str = "bulk", cap: int | None = None):
        if mode not in ("bulk", "passthrough"):
            raise ValueError(f"unknown evals mode {mode!r}")
        self.tracker = tracker
        self.mode = mode
        self.cap = cap
        self._lease: float | None = None  # bulk: the units leased for the run; None = unlimited
        self._available = float("inf")
        self._out = 0.0  # taken and neither put back nor reported used
        self._used = 0.0

    def open(self) -> None:
        if self.mode != "bulk":
            return
        remaining = self.tracker.remaining
        if remaining is None and self.cap is None:
            return
        lease = remaining if self.cap is None else (self.cap if remaining is None else min(self.cap, remaining))
        self.tracker.reserve(int(lease))
        self._lease = self._available = float(lease)

    def close(self) -> None:
        if self.mode == "bulk" and self._lease is not None:
            # The server committed each evaluation that ran against the lease; give back what is still reserved,
            # never more than the tracker holds (another engine may share it).
            unused = min(int(self._lease - self._used), int(self.tracker.reserved))
            if unused > 0:
                self.tracker.release(unused)
            self._lease = None

    @property
    def available(self) -> float | None:
        if self.mode == "passthrough":
            return self.tracker.remaining
        return None if self._lease is None else self._available

    def take(self, units: float) -> float:
        if units <= 0:
            return 0.0
        if self.mode == "bulk":
            if self._lease is None:
                self._out += units
                return units
            if units > self._lease:
                raise ResourceExhaustedError(f"evals: {units} asked, the lease is {self._lease}")
            if units <= self._available:
                self._available -= units
                self._out += units
                return units
            if self._out == 0:
                raise ResourceExhaustedError(f"evals: {units} asked, {self._available} left and nothing in flight")
            return 0.0
        from gepa.oa.budget import BudgetExhausted

        try:
            self.tracker.reserve(int(units))
        except BudgetExhausted:
            if self.tracker.remaining == 0 and self._out == 0:
                raise ResourceExhaustedError("evals: the eval budget is spent") from None
            return 0.0
        self._out += units
        return units

    def put(self, units: float, used: float = 0.0) -> None:
        self._used += used
        self._out = max(0.0, self._out - units - used)
        if self.mode == "bulk":
            if self._lease is not None:
                self._available = min(self._lease, self._available + units)
            return
        if units > 0:
            self.tracker.release(int(units))
