# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Resources: pools of fungible units the engine takes from at dispatch and puts back at completion.

Only the engine calls a resource. A stage says what it needs per item (its ``demand``) and what it used (its
``used``); the engine takes the demand, all or nothing, and puts back demand minus used. Whether a pool behaves
as a budget (drained: the stage reports what it used) or a bandwidth (refilled: nothing is reported used) is
therefore decided by the stages, not by the pool.

``take`` returns what it took (the whole demand, or the whole pool for a demand above its cap, which the holder
then serializes inside), ``0`` for "not now", and raises :class:`ResourceExhaustedError` for "never again": the pool
is the one that knows which, since it sees whether anything of it is still out and could come back.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol


class ResourceExhaustedError(Exception):
    """This resource can never satisfy a demand again."""


class Resource(Protocol):
    @property
    def available(self) -> float | None: ...  # None = unlimited

    def take(self, units: float) -> float: ...  # units taken; 0 = not now

    def put(self, units: float, used: float = 0.0) -> None: ...

    def open(self) -> None: ...

    def close(self) -> None: ...


class Pool:
    """The shipped resource: a cap, optionally refilled from a clock.

    ``rate`` is units per second: a token bucket that refills itself, up to ``cap``, when taken from.
    """

    def __init__(self, name: str, cap: float | None, *, unit: str = "unit", rate: float | None = None):
        self.name = name
        self.unit = unit
        self.cap = cap
        self.rate = rate
        self._available = float("inf") if cap is None else float(cap)
        self._in_flight = 0.0  # taken and neither put back nor reported used
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    @property
    def available(self) -> float | None:
        if self.cap is None:
            return None
        with self._lock:
            self._refill()
            return self._available

    @property
    def in_flight(self) -> float:
        return self._in_flight

    def _refill(self) -> None:
        if self.rate is None or self.cap is None:
            return
        now = time.monotonic()
        self._available = min(float(self.cap), self._available + (now - self._last_refill) * self.rate)
        self._last_refill = now

    def take(self, units: float) -> float:
        if units <= 0:
            return 0.0
        if self.cap is None:
            self._in_flight += units
            return units
        with self._lock:
            self._refill()
            need = min(units, float(self.cap))  # a demand above the cap takes the whole pool
            if need <= self._available:
                self._available -= need
                self._in_flight += need
                return need
            if self.rate is None and self._in_flight == 0:
                # Nothing is out that could come back and nothing refills: this demand can never be met.
                raise ResourceExhaustedError(
                    f"{self.name}: {need} {self.unit}s asked, {self._available} left and nothing in flight"
                )
            return 0.0

    def put(self, units: float, used: float = 0.0) -> None:
        with self._lock:
            self._in_flight = max(0.0, self._in_flight - units - used)
            if self.cap is not None:
                self._available = min(float(self.cap), self._available + units)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


class Evals(Pool):
    """The proxy of the optimize_anything eval budget: upstream is the eval server's ``BudgetTracker``.

    ``bulk`` (the default): ``open`` reserves, before anything runs, the budget the entire run needs (the ``cap``
    the user set, or everything the tracker still has), so no backend sharing the server can race this engine
    for it; take and put are then local, the server books each evaluation that runs against the reservation
    (the evaluation stages call it with ``reserved=True``), and ``close`` releases what was leased but never run.
    ``passthrough``: every take is forwarded as a reserve and every put as a release of the unused part, as they
    happen, for a server shared with engines running at the same time.
    """

    def __init__(self, tracker: Any, *, mode: str = "bulk", cap: int | None = None):
        if mode not in ("bulk", "passthrough"):
            raise ValueError(f"unknown evals mode {mode!r}")
        self.tracker = tracker
        self.mode = mode
        remaining = tracker.remaining
        lease = remaining if cap is None else (cap if remaining is None else min(cap, remaining))
        super().__init__("evals", lease, unit="evaluation")
        self._leased = 0.0
        self._used = 0.0

    def open(self) -> None:
        if self.mode == "bulk" and self.cap is not None:
            self.tracker.reserve(int(self.cap))
            self._leased = float(self.cap)

    def close(self) -> None:
        if self.mode == "bulk":
            # The server committed each evaluation that ran against the lease; give back what is still reserved,
            # never more than the tracker holds (another engine may share it).
            unused = min(int(self._leased - self._used), int(self.tracker.reserved))
            if unused > 0:
                self.tracker.release(unused)
            self._leased = 0.0

    @property
    def available(self) -> float | None:
        if self.mode == "passthrough":
            return self.tracker.remaining
        return super().available

    def take(self, units: float) -> float:
        if self.mode == "bulk":
            return super().take(units)
        from gepa.oa.budget import BudgetExhausted

        if units <= 0:
            return 0.0
        try:
            self.tracker.reserve(int(units))
        except BudgetExhausted:
            remaining = self.tracker.remaining
            if remaining == 0 and self._in_flight == 0:
                raise ResourceExhaustedError("evals: the eval budget is spent") from None
            return 0.0
        self._in_flight += units
        return units

    def put(self, units: float, used: float = 0.0) -> None:
        self._used += used
        if self.mode == "bulk":
            super().put(units, used)
            return
        self._in_flight = max(0.0, self._in_flight - units - used)
        if units > 0:
            self.tracker.release(int(units))
