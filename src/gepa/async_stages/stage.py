# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stages and the item they pass along.

A stage is an input *place* followed by a *body*. The place is the stage's own buffer: a ``consume`` place is a
bounded queue the firing pops from; a ``read`` place is a pool the firing reads and leaves (the body builds a new
item from it). The body is ``run(item)``: on a :class:`HeadStage` it runs inline on the engine thread, serialized
with every other head body, so it may touch the pipeline's shared objects; on a :class:`WorkerStage` it runs on a
thread of the bandwidth the stage draws from, so it touches only the item and thread-safe clients.

A stage also declares what the engine must take from the resources before the body runs (``demand``) and reports
what it used afterwards (``used``); both are reads the engine makes on its own thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

P = TypeVar("P")
In = TypeVar("In")
Out = TypeVar("Out")

PlaceMode = Literal["consume", "read"]

#: A selection rule: given the items waiting in a consume place, the index of the one to fire next, or ``None``
#: when none should fire yet (a sibling rule waiting for the rest of a fan-out).
SelectRule = Callable[[Sequence["WorkItem[Any]"]], "int | None"]


@dataclass
class WorkItem(Generic[P]):
    """The only thing that travels between stages.

    The engine reads ``id``, ``held`` and ``tag`` for its decisions, and writes ``parent_id``, ``fanout``,
    ``stamps`` and ``outcome`` as bookkeeping for the pipeline and the event log. ``payload`` is the pipeline's.
    """

    payload: P
    id: int = 0
    parent_id: int | None = None
    fanout: int = 1
    #: Per-stage completion counters at creation, inherited along the parent chain.
    stamps: dict[str, int] = field(default_factory=dict)
    #: Units held on the item (a ``hold`` demand), by resource; served to later demands, put back when the item ends.
    held: dict[str, float] = field(default_factory=dict)
    #: The label of the edge the item leaves its stage by; ``None`` takes the default edge.
    tag: str | None = None
    outcome: str | None = None
    #: What the last body used, by resource; worker bodies annotate it for ``Stage.used`` to read.
    usage: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    enqueued_at: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    wait_seconds: dict[str, float] = field(default_factory=dict)

    def gap(self, stage: str, current: int) -> int:
        """How many completions of ``stage`` happened since this item's stamp of it."""
        return current - self.stamps.get(stage, current)


def fifo(items: Sequence[WorkItem[Any]]) -> int | None:
    return 0 if items else None


def by_key(key: Callable[[WorkItem[Any]], float]) -> SelectRule:
    """Highest key first; arrival order on ties."""

    def rule(items: Sequence[WorkItem[Any]]) -> int | None:
        if not items:
            return None
        return max(range(len(items)), key=lambda i: (key(items[i]), -items[i].enqueued_at))

    return rule


class Stage(Generic[In, Out]):
    """Base of the two stage kinds. Subclass :class:`HeadStage` or :class:`WorkerStage`, never this."""

    #: Payload types, used by the pipeline validator to check edges; ``None`` means unchecked. For a ``read``
    #: stage ``in_type`` is the type of the items delivered into its place; its body receives a ``WorkItem`` whose
    #: payload is the list of those items.
    in_type: type | None = None
    out_type: type | None = None

    def __init__(
        self,
        name: str,
        *,
        mode: PlaceMode = "consume",
        select: SelectRule = fifo,
        workers: int = 1,
        max_workers: int | None = None,
        max_queue: int | None = None,
        bandwidth: str | None = None,
        demand: Mapping[str, float] | None = None,
        hold: Sequence[str] = (),
    ):
        self.name = name
        self.mode: PlaceMode = mode
        self.select = select
        self.workers = workers
        self.max_workers = max_workers if max_workers is not None else workers
        self.max_queue = max_queue
        self.bandwidth = bandwidth
        self._demand = dict(demand or {})
        self.hold: frozenset[str] = frozenset(hold)

    # -- what the engine asks ---------------------------------------------

    def demand(self, item: WorkItem[In]) -> dict[str, float]:
        """Units per resource the engine must take before ``run``; the constructor's ``demand`` by default."""
        return dict(self._demand)

    def used(self, item: WorkItem[In]) -> dict[str, float]:
        """Units per resource the body consumed; the item's ``usage`` annotation by default."""
        return dict(item.usage)

    def after(self, item: WorkItem[In]) -> None:
        """Epilogue on the engine thread after a successful body, before ``used`` is read.

        A worker stage uses it for what must happen serialized with the head bodies (writing an evaluation
        cache, counting evaluations on a shared state); it must stay cheap.
        """
        return

    def carry(self, item: WorkItem[In], outputs: Sequence[WorkItem[Out]]) -> int | None:
        """On a fan-out, the index of the returned item that keeps the input's held units; ``None`` puts them back.

        A continuing item (the input returned as is) keeps its hold without asking.
        """
        return None

    def run(self, item: WorkItem[In]) -> list[WorkItem[Out]]:
        """The body. Returns 0 (drop), 1, or n (fan-out) items, each optionally tagged."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class HeadStage(Stage[In, Out]):
    """Body runs inline on the engine thread, serialized with every other head body."""


class WorkerStage(Stage[In, Out]):
    """Body runs on a thread of the stage's bandwidth, concurrently with other workers."""

    def __init__(self, name: str, *, bandwidth: str, **kwargs: Any):
        super().__init__(name, bandwidth=bandwidth, **kwargs)
