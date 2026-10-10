# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Resources are caps the engine books against; this module holds the one upstream, the eval budget.

A resource is a name and a cap. The engine keeps what is available and what is in flight, takes a stage's demand
at dispatch (all or nothing) and puts back demand minus used at completion. Whether a resource behaves as a budget
(drained: the stage reports what it used) or a bandwidth (refilled: nothing is reported used) is decided by the
stages, not by the resource. An **upstream** is what stands behind a resource outside the engine, such as the eval
server's budget tracker behind ``evals``; only the engine calls it.
"""

from __future__ import annotations

from typing import Any, Protocol


class ResourceExhaustedError(Exception):
    """This resource can never satisfy a demand again."""


class Upstream(Protocol):
    """What stands behind a resource outside the engine. Only the engine calls it.

    ``open`` runs at the start: it returns the lease the engine books locally for the whole run, or ``None`` to be
    asked per take. ``take`` and ``put`` are called per dispatch and completion only while ``open`` returned
    ``None``: ``take`` returns the units granted, ``0`` for "not now", and raises :class:`ResourceExhaustedError`
    for "never again". ``close`` runs at the end with the units used over the run, to give back what was leased
    but never used.
    """

    def open(self) -> float | None: ...

    def take(self, units: float) -> float: ...

    def put(self, units: float, used: float) -> None: ...

    def close(self, used: float) -> None: ...


class Evals:
    """The upstream of ``evals``: the optimize_anything eval budget, a reservation ledger (``BudgetTracker``).

    ``bulk`` (the default): ``open`` reserves, before anything runs, the budget the entire run needs (the ``cap``
    the user set, or everything the tracker still has), so no backend sharing the server can race this engine
    for it; the engine then books locally against that lease, the server books each evaluation that runs against
    the reservation (the evaluation stages call it with ``reserved=True``), and ``close`` releases what was leased
    but never run. ``passthrough``: ``open`` leases nothing; every take is forwarded as a reserve and every put as
    a release of the unused part, as they happen, for a server shared with engines running at the same time.
    """

    def __init__(self, tracker: Any, *, mode: str = "bulk", cap: int | None = None):
        if mode not in ("bulk", "passthrough"):
            raise ValueError(f"unknown evals mode {mode!r}")
        self.tracker = tracker
        self.mode = mode
        self.cap = cap
        self._leased = 0.0
        self._out = 0.0  # passthrough: reserved at the tracker and neither released nor run

    def open(self) -> float | None:
        if self.mode != "bulk":
            return None
        remaining = self.tracker.remaining
        if remaining is None and self.cap is None:
            return None  # an unlimited budget: nothing to lease, every take is granted
        lease = remaining if self.cap is None else (self.cap if remaining is None else min(self.cap, remaining))
        self.tracker.reserve(int(lease))
        self._leased = float(lease)
        return self._leased

    def take(self, units: float) -> float:
        from gepa.oa.budget import BudgetExhausted

        if units <= 0:
            return 0.0
        try:
            self.tracker.reserve(int(units))
        except BudgetExhausted:
            if self.tracker.remaining == 0 and self._out == 0:
                raise ResourceExhaustedError("evals: the eval budget is spent") from None
            return 0.0
        self._out += units
        return units

    def put(self, units: float, used: float) -> None:
        self._out = max(0.0, self._out - units - used)
        if units > 0:
            self.tracker.release(int(units))

    def close(self, used: float) -> None:
        if self.mode == "bulk" and self._leased > 0:
            # The server committed each evaluation that ran against the lease; give back what is still reserved,
            # never more than the tracker holds (another engine may share it).
            unused = min(int(self._leased - used), int(self.tracker.reserved))
            if unused > 0:
                self.tracker.release(unused)
            self._leased = 0.0
