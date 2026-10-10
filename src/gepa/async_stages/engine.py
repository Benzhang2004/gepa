# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The engine: fires enabled stages, runs bodies, routes items along edges, and allocates resources.

A stage is enabled when its place offers an item under its rule, the engine can take the stage's whole demand
from the resources, and the places its edges lead to have room. Firing takes everything atomically or nothing;
completion routes the returned items along the edges and puts units back (demand minus what the stage reports
used). Head bodies run inline on the engine thread; worker bodies on a thread pool per bandwidth, under the
stage's own ``workers``.

The engine knows nothing about candidates, lineage or data. It reads three fields of an item for its decisions
(id, units held, tag) and writes four as bookkeeping (parent id, fan-out count, stamps, outcome).
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gepa.async_stages.events import EventLog
from gepa.async_stages.pipeline import Pipeline
from gepa.async_stages.resources import ResourceExhaustedError, Upstream
from gepa.async_stages.stage import Stage, WorkerStage, WorkItem


@dataclass
class _Firing:
    stage: Stage[Any, Any]
    item: WorkItem[Any]
    demand: dict[str, float]  # what was taken from the pools for this firing (after serving from held units)
    started: float


@dataclass
class _Completion:
    firing: _Firing
    outputs: list[WorkItem[Any]] | None
    error: BaseException | None
    seconds: float


class Engine:
    """Drives a pipeline. ``resources`` are the pools stages may name in their demand.

    ``executors`` maps a bandwidth name to a thread-pool size; a worker stage submits to the pool of its
    bandwidth. ``stage_overrides`` carries per-stage scheduling keys from config (``workers``, ``max_queue``).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        caps: Mapping[str, float | None],
        *,
        upstreams: Mapping[str, Upstream] | None = None,
        executors: Mapping[str, int],
        run_dir: str | None = None,
        write_event_log: bool = True,
        poll_interval_seconds: float = 0.02,
        raise_on_exception: bool = True,
        on_item_end: Callable[[WorkItem[Any], str], None] | None = None,
        gap_watch: str | None = None,
    ):
        self.pipeline = pipeline
        #: When set, every fire records how many completions of this stage happened since the item was created
        #: (its distribution per stage is ``stats["gaps"]``; "commit" in the shipped pipelines).
        self.gap_watch = gap_watch
        #: The resources: a cap per name (None = unbounded), booked by the engine itself. An upstream, when a
        #: resource has one, leases the cap at open or is asked per take (see ``resources.Upstream``).
        self.caps: dict[str, float | None] = dict(caps)
        self.upstreams: dict[str, Upstream] = dict(upstreams or {})
        for name in self.upstreams:
            self.caps.setdefault(name, None)
        self._available: dict[str, float | None] = {n: (None if c is None else float(c)) for n, c in self.caps.items()}
        self.run_dir = run_dir
        self.poll_interval = poll_interval_seconds
        self.raise_on_exception = raise_on_exception
        self.on_item_end = on_item_end
        self.events = EventLog(run_dir, enabled=write_event_log)
        self.stats: dict[str, Any] = {}

        self.places: dict[str, list[WorkItem[Any]]] = {name: [] for name in pipeline.stages}
        self.counters: dict[str, int] = dict.fromkeys(pipeline.stages, 0)
        self.inflight: dict[str, int] = dict.fromkeys(pipeline.stages, 0)
        self._executors: dict[str, ThreadPoolExecutor] = {}
        for name, size in executors.items():
            self._executors[name] = ThreadPoolExecutor(max_workers=max(1, size), thread_name_prefix=f"async-{name}")
        self._completions: queue.Queue[_Completion] = queue.Queue()
        self._next_id = 1
        self._stop_requested = False
        self._stopping = False
        self._stop_reason: str | None = None
        self._fatal: BaseException | None = None
        self._in_flight_units: dict[str, float] = dict.fromkeys(self.caps, 0.0)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def request_stop(self, reason: str = "requested") -> None:
        self._stop_requested = True
        if self._stop_reason is None:
            self._stop_reason = reason

    def inject(self, stage_name: str, payload: Any, *, tag: str | None = None) -> WorkItem[Any]:
        """Put a new item into a stage's place from outside (the backend seeds the pipeline this way)."""
        item = self._new_item(payload, parent=None)
        item.tag = tag
        self._deliver(self.pipeline.stages[stage_name], item)
        return item

    def run(self) -> dict[str, Any]:
        self.pipeline.validate()
        for name, upstream in self.upstreams.items():
            lease = upstream.open()
            if lease is not None:
                self.caps[name] = lease
                self._available[name] = float(lease)
        self.events.emit("run_start", stages=list(self.pipeline.stages), resources=list(self.caps))
        try:
            while True:
                self._drain_completions()
                if self._fatal is not None:
                    break
                if self._stop_requested and not self._stopping:
                    self._begin_stop(self._stop_reason or "requested")
                fired = self._fire_all()
                self.events.tick(dict(self.inflight), dict(self._in_flight_units))
                busy = any(self.inflight.values())
                if not busy and not fired:
                    if self._stopping:
                        break
                    if not self._anything_waiting():
                        self._begin_stop("quiescent")
                        break
                    # Items wait but nothing can fire: either a pool is drained (handled at take) or wiring.
                    self._begin_stop("quiescent")
                    break
                if not fired and busy:
                    self._wait_for_completion()
        finally:
            for ex in self._executors.values():
                ex.shutdown(wait=True, cancel_futures=False)
            self._drain_completions(force=True)
            # Items still waiting in a consume place cannot run any more: end them so what they hold comes back.
            for name, stage in self.pipeline.stages.items():
                if stage.mode != "consume":
                    continue
                for leftover in list(self.places[name]):
                    self._end_item(leftover, "unfinished")
                self.places[name].clear()
            for name, upstream in self.upstreams.items():
                upstream.close(self.events.resource(name).used)
            self.events.stop_reason = self._stop_reason
            self.stats = self.events.summary()
            self.stats["counters"] = dict(self.counters)
            self.events.emit("run_end", **{k: v for k, v in self.stats.items() if k not in ("stages", "resources")})
            self.events.close()
        if self._fatal is not None:
            raise self._fatal
        return self.stats

    # ------------------------------------------------------------------ loop pieces

    def _anything_waiting(self) -> bool:
        return any(self.places[n] for n, s in self.pipeline.stages.items() if s.mode == "consume")

    def _begin_stop(self, reason: str) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._stop_reason is None:
            self._stop_reason = reason
        self.events.emit("stop", reason=self._stop_reason)

    def _wait_for_completion(self) -> None:
        try:
            c = self._completions.get(timeout=self.poll_interval)
        except queue.Empty:
            return
        self._on_completion(c)

    def _drain_completions(self, force: bool = False) -> None:
        """Settle every completion in the queue. After a fatal error the loop stops firing, but what completes
        is still settled (``force``) so that every unit taken is accounted for."""
        while True:
            try:
                c = self._completions.get_nowait()
            except queue.Empty:
                return
            self._on_completion(c)
            if self._fatal is not None and not force:
                return

    def _fire_all(self) -> int:
        fired = 0
        # Downstream stages first so finished work leaves before new work enters; a fixed order is enough since
        # each stage is tried until it cannot fire.
        for name in reversed(list(self.pipeline.stages)):
            stage = self.pipeline.stages[name]
            if self._fatal is not None:
                return fired
            if self._stopping and stage.mode == "read":
                continue  # stopping: no new work is created, what exists drains
            while self._try_fire(stage):
                fired += 1
                if self._fatal is not None:
                    return fired
        return fired

    # ------------------------------------------------------------------ enabledness and firing

    def _room_downstream(self, stage: Stage[Any, Any]) -> bool:
        for (src, _label), dst in self.pipeline.edges.items():
            if src != stage.name:
                continue
            d = self.pipeline.stages[dst]
            if d.mode == "consume" and d.max_queue is not None and len(self.places[dst]) >= d.max_queue:
                return False
        return True

    def _try_fire(self, stage: Stage[Any, Any]) -> bool:
        if self.inflight[stage.name] >= stage.workers:
            return False
        place = self.places[stage.name]
        if not place:
            return False
        if not self._room_downstream(stage):
            return False
        if stage.mode == "read":
            item: WorkItem[Any] = WorkItem(payload=list(place))
            index = None
        else:
            index = stage.select(place)
            if index is None:
                return False
            item = place[index]
        demand = stage.demand(item)
        taken = self._take(stage, item, demand)
        if taken is None:
            return False
        if index is not None:
            del place[index]
        if stage.mode == "consume":
            item.wait_seconds[stage.name] = item.wait_seconds.get(stage.name, 0.0) + (
                time.monotonic() - item.enqueued_at
            )
        firing = _Firing(stage, item, taken, time.monotonic())
        self.inflight[stage.name] += 1
        st = self.events.stage(stage.name)
        st.fired += 1
        if self.gap_watch is not None and stage.mode == "consume":
            self.events.gaps.setdefault(stage.name, Counter())[
                item.gap(self.gap_watch, self.counters[self.gap_watch])
            ] += 1
        self.events.emit("fire", stage=stage.name, item=item.id, demand=taken, queue=len(place))
        if isinstance(stage, WorkerStage):
            bandwidth = stage.bandwidth or ""
            executor = self._executors.get(bandwidth)
            if executor is None:
                raise RuntimeError(f"stage {stage.name!r} draws from bandwidth {bandwidth!r}, which has no thread pool")
            fut = executor.submit(stage.run, item)
            fut.add_done_callback(lambda f, firing=firing: self._on_future(firing, f))
        else:
            try:
                outputs = stage.run(item)
                err: BaseException | None = None
            except Exception as e:
                outputs, err = None, e
            self._completions.put(_Completion(firing, outputs, err, time.monotonic() - firing.started))
        return True

    def _on_future(self, firing: _Firing, fut: Future[Any]) -> None:
        err = fut.exception()
        outputs = None if err else fut.result()
        self._completions.put(_Completion(firing, outputs, err, time.monotonic() - firing.started))

    def _take(
        self, stage: Stage[Any, Any], item: WorkItem[Any], demand: Mapping[str, float]
    ) -> dict[str, float] | None:
        """Take ``demand`` from the pools, serving held units first. All or nothing. ``None`` = not now."""
        taken: dict[str, float] = {}
        for name, units in demand.items():
            if units <= 0:
                continue
            need = units
            if name in item.held and name not in stage.hold:
                served = min(item.held[name], need)
                item.held[name] -= served
                if item.held[name] <= 0:
                    del item.held[name]
                need -= served
            if need <= 0:
                taken[name] = 0.0
                continue
            if name not in self.caps:
                raise RuntimeError(f"stage {stage.name!r} demands unknown resource {name!r}")
            try:
                got = self._take_units(name, need)
            except ResourceExhaustedError as e:
                self._rollback(taken, item)
                self.events.resource(name).refused += 1
                self._end_item_at(stage, item, "budget", reason=str(e))
                self._begin_stop("exhausted")
                return None
            if got <= 0:
                self._rollback(taken, item)  # not now: the loop asks again; only exhaustion counts as refused
                return None
            taken[name] = got
        for name, got in taken.items():  # count only a demand that was taken whole
            self.events.resource(name).taken += got
            self._in_flight_units[name] = self._in_flight_units.get(name, 0.0) + got
        return taken

    def _rollback(self, taken: Mapping[str, float], item: WorkItem[Any]) -> None:
        for name, units in taken.items():
            if units > 0:
                self._put_units(name, units, 0.0)  # never counted as taken or in flight: the demand was not met whole

    def _take_units(self, name: str, units: float) -> float:
        """Units taken from one resource: the whole demand, or the whole cap for a demand above it (the holder
        serializes inside); ``0`` = not now; raises ``ResourceExhaustedError`` when the demand can never be met,
        which is when it cannot be met now and nothing is in flight that could come back."""
        cap = self.caps[name]
        if cap is None:
            upstream = self.upstreams.get(name)
            return upstream.take(units) if upstream is not None else units
        available = self._available[name] or 0.0
        want = min(units, cap)
        if want <= available:
            self._available[name] = available - want
            return want
        if self._in_flight_units.get(name, 0.0) <= 0:
            raise ResourceExhaustedError(f"{name}: {want} asked, {available} left and nothing in flight")
        return 0.0

    def _put_units(self, name: str, units: float, used: float) -> None:
        """``units`` come back to the resource; ``used`` were spent and stay out."""
        cap = self.caps[name]
        if cap is None:
            upstream = self.upstreams.get(name)
            if upstream is not None:
                upstream.put(units, used)
            return
        self._available[name] = min(cap, (self._available[name] or 0.0) + units)

    # ------------------------------------------------------------------ completion and routing

    def _on_completion(self, c: _Completion) -> None:
        firing, stage, item = c.firing, c.firing.stage, c.firing.item
        self.inflight[stage.name] -= 1
        st = self.events.stage(stage.name)
        st.completed += 1
        st.busy_seconds += c.seconds
        st.wait_seconds += item.wait_seconds.get(stage.name, 0.0)
        item.stage_seconds[stage.name] = item.stage_seconds.get(stage.name, 0.0) + c.seconds
        if c.error is not None:
            st.failed += 1
            self.events.emit("error", stage=stage.name, item=item.id, error=repr(c.error))
            # A failed body may still have spent (an evaluation that raised was paid for): charge what it annotated.
            self._settle(stage, item, firing.demand, used=stage.used(item))
            if stage.mode == "consume":
                self._end_item(item, "error")
            if self.raise_on_exception:
                self._fatal = c.error
                self._begin_stop("error")
            return
        self.counters[stage.name] += 1
        if self._fatal is not None:
            # The run is aborting: book what this firing used and end the item; nothing is routed any more.
            self._settle(stage, item, firing.demand, used=stage.used(item))
            if stage.mode == "consume":
                self._end_item(item, "unfinished")
            return
        try:
            stage.after(item)
        except Exception as e:
            st.failed += 1
            self.events.emit("error", stage=stage.name, item=item.id, error=repr(e), where="after")
            self._settle(stage, item, firing.demand, used=stage.used(item))
            if stage.mode == "consume":
                self._end_item(item, "error")
            if self.raise_on_exception:
                self._fatal = e
                self._begin_stop("error")
            return
        used = stage.used(item)
        self._settle(stage, item, firing.demand, used)
        outputs = list(c.outputs or [])
        self.events.emit("complete", stage=stage.name, item=item.id, seconds=round(c.seconds, 4), outputs=len(outputs))
        if stage.mode == "read":
            # The read item is synthetic: what it holds belongs to the items the body created.
            for out in outputs:
                self._register_new(out, parent=None)
            if outputs:
                keep = stage.carry(item, outputs)
                target = outputs[keep] if keep is not None and 0 <= keep < len(outputs) else outputs[0]
                for name, units in item.held.items():
                    target.held[name] = target.held.get(name, 0.0) + units
                item.held = {}
            else:
                self._end_item(item, "nothing", log=False)
            for out in outputs:
                self._route(stage, out)
            return
        if not outputs:
            self._end_item(item, "dropped")
            return
        continuing = [o for o in outputs if o is item]
        children = [o for o in outputs if o is not item]
        if continuing and not children:
            self._route(stage, item)
            return
        # A fan-out (or a replacement): the input ends; children descend from it.
        keep = stage.carry(item, outputs)
        for out in outputs:
            if out is item:
                continue
            self._register_new(out, parent=item)
            out.fanout = len(outputs)
        if keep is not None and 0 <= keep < len(outputs) and outputs[keep] is not item:
            outputs[keep].held = dict(item.held)
            item.held = {}
        if continuing:
            self._route(stage, item)
        else:
            self._end_item(item, "fanned_out", log=False)
        for out in children:
            self._route(stage, out)

    def _settle(
        self, stage: Stage[Any, Any], item: WorkItem[Any], taken: Mapping[str, float], used: Mapping[str, float]
    ) -> None:
        """Put back demand minus used; keep held demands on the item."""
        for name, units in taken.items():
            if units <= 0:
                continue
            if name in stage.hold:
                item.held[name] = item.held.get(name, 0.0) + units  # stays in flight until the item ends
                continue
            u = float(used.get(name, 0.0))
            back = max(0.0, units - u)
            self._put_units(name, back, min(u, units))
            self.events.resource(name).put_back += back
            self.events.resource(name).used += min(u, units)
            self._in_flight_units[name] -= units

    def _held_units(self, name: str, delta: float) -> None:
        self._in_flight_units[name] = self._in_flight_units.get(name, 0.0) + delta

    def _route(self, stage: Stage[Any, Any], item: WorkItem[Any]) -> None:
        tag, item.tag = item.tag, None
        dst = self.pipeline.successor(stage.name, tag)
        if dst is None:
            self._end_item(item, tag or "finished")
            return
        self._deliver(dst, item)

    def _deliver(self, dst: Stage[Any, Any], item: WorkItem[Any]) -> None:
        if dst.mode == "read":
            # Delivery into a read place ends the item as work; what stays is content for the reading stage.
            self.places[dst.name].append(item)
            self._end_item(item, "accepted")
            return
        item.enqueued_at = time.monotonic()
        self.places[dst.name].append(item)
        depth = len(self.places[dst.name])
        st = self.events.stage(dst.name)
        st.max_queue_depth = max(st.max_queue_depth, depth)

    # ------------------------------------------------------------------ items

    def _new_item(self, payload: Any, parent: WorkItem[Any] | None) -> WorkItem[Any]:
        item: WorkItem[Any] = WorkItem(payload=payload)
        self._register_new(item, parent)
        return item

    def _register_new(self, item: WorkItem[Any], parent: WorkItem[Any] | None) -> None:
        item.id = self._next_id
        self._next_id += 1
        if parent is None:
            item.parent_id = None
            item.stamps = dict(self.counters)
        else:
            item.parent_id = parent.id
            item.stamps = dict(parent.stamps)
            item.created_at = parent.created_at
        self.events.emit("item", item=item.id, parent=item.parent_id)

    def _end_item_at(self, stage: Stage[Any, Any], item: WorkItem[Any], outcome: str, **fields: Any) -> None:
        """End an item that is still in ``stage``'s place (a refused demand)."""
        if stage.mode == "consume":
            place = self.places[stage.name]
            for i, it in enumerate(place):
                if it is item:
                    del place[i]
                    break
        for name in item.held:
            self.events.stranded[name] = self.events.stranded.get(name, 0.0)
        self._end_item(item, outcome, **fields)

    def _end_item(self, item: WorkItem[Any], outcome: str, log: bool = True, **fields: Any) -> None:
        for name, units in list(item.held.items()):
            self._put_units(name, units, 0.0)
            self.events.resource(name).put_back += units
            self._held_units(name, -units)
        item.held = {}
        item.outcome = outcome
        if log:
            self.events.outcomes[outcome] += 1
            self.events.emit(
                "end", item=item.id, outcome=outcome, latency=round(time.monotonic() - item.created_at, 4), **fields
            )
        if self.on_item_end is not None:
            try:
                self.on_item_end(item, outcome)
            except Exception as e:
                self.events.emit("error", stage="on_item_end", item=item.id, error=repr(e))

    # ------------------------------------------------------------------ introspection

    def gap(self, item: WorkItem[Any], stage: str) -> int:
        return item.gap(stage, self.counters[stage])

    def describe(self) -> str:
        lines = []
        for name, stage in self.pipeline.stages.items():
            lines.append(f"{name}: {len(self.places[name])} waiting, {self.inflight[name]} in flight / {stage.workers}")
        for name in self.caps:
            lines.append(f"{name}: available {self.available(name)}, in flight {self.in_flight(name)}")
        return "\n".join(lines)

    def available(self, name: str) -> float | None:
        """Units of a resource not taken right now; ``None`` for an unbounded one."""
        return self._available[name]

    def in_flight(self, name: str) -> float:
        """Units of a resource taken and neither put back nor reported used, held units included."""
        return self._in_flight_units.get(name, 0.0)


def format_exception(err: BaseException) -> str:
    return "".join(traceback.format_exception(type(err), err, err.__traceback__))
