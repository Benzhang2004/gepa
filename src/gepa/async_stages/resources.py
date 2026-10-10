# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Resources: pools of fungible units the engine takes from at dispatch and puts back at completion.

Only the engine calls a resource. A stage says what it needs per item (its ``demand``) and what it used (its
``used``); the engine takes the demand, all or nothing, and puts back demand minus used. Whether a pool behaves
as a budget (drained: the stage reports what it used) or a bandwidth (refilled: nothing is reported used) is
therefore decided by the stages, not by the pool.

``take`` answers ``False`` for "not now" and raises :class:`ResourceExhaustedError` for "never again": the pool is the
one that knows which, since it sees whether anything of it is still out and could come back.
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

    def take(self, units: float) -> bool: ...

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

    def take(self, units: float) -> bool:
        if units <= 0:
            return True
        if self.cap is None:
            self._in_flight += units
            return True
        with self._lock:
            self._refill()
            if units > self.cap:
                raise ResourceExhaustedError(f"{self.name}: {units} {self.unit}s asked, cap is {self.cap}")
            if units <= self._available:
                self._available -= units
                self._in_flight += units
                return True
            if self.rate is None and self._in_flight == 0 and self._available == 0:
                raise ResourceExhaustedError(f"{self.name}: nothing left and nothing in flight")
            return False

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

    ``bulk`` (the default): ``open`` reserves the whole allotment from the tracker, take and put are local, and
    ``close`` releases what was reserved but never run (the server commits each evaluation that runs, since the
    evaluation stages call it with ``reserved=True``). ``passthrough``: every take is forwarded as a reserve and
    every put as a release of the unused part, as they happen, for a server shared with engines running at the
    same time.
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
            unused = int(self._leased - self._used)
            if unused > 0:
                self.tracker.release(unused)
            self._leased = 0.0

    @property
    def available(self) -> float | None:
        if self.mode == "passthrough":
            return self.tracker.remaining
        return super().available

    def take(self, units: float) -> bool:
        if self.mode == "bulk":
            return super().take(units)
        from gepa.oa.budget import BudgetExhausted

        if units <= 0:
            return True
        try:
            self.tracker.reserve(int(units))
        except BudgetExhausted:
            remaining = self.tracker.remaining
            if remaining == 0 and self._in_flight == 0:
                raise ResourceExhaustedError("evals: the eval budget is spent") from None
            return False
        self._in_flight += units
        return True

    def put(self, units: float, used: float = 0.0) -> None:
        self._used += used
        if self.mode == "bulk":
            super().put(units, used)
            return
        self._in_flight = max(0.0, self._in_flight - units - used)
        if units > 0:
            self.tracker.release(int(units))
