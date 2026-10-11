# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""The engine: fires enabled stages, runs bodies, routes items along edges, and allocates resources.

A stage is enabled when its place offers an item under its rule, the engine can take the stage's whole demand
from the resources, and the places its edges lead to have room. Firing takes everything atomically or nothing;
completion routes the returned items along the edges and puts units back (demand minus what the stage reports
used). Head bodies run inline on the engine thread; worker bodies on a thread pool per bandwidth, under the
stage's own ``workers``.

The engine knows nothing about candidates, lineage or data. It reads three fields of an item for its decisions
(id, tag) and writes its bookkeeping (parent id, fan-out count, stamps, origin, lineage, outcome).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gepa.async_stages.events import EventLog
from gepa.async_stages.pipeline import Pipeline, PipelineError
from gepa.async_stages.resources import Resource, ResourceExhaustedError
from gepa.async_stages.stage import Stage, WorkerStage, WorkItem


@dataclass
class _Firing:
    stage: Stage[Any, Any]
    item: WorkItem[Any]
    demand: dict[str, float]  # what was taken from the resources for this firing
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
    bandwidth. ``gap_watch`` names the stage whose completions the gap histograms count against.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        resources: Mapping[str, Resource],
        *,
        executors: Mapping[str, int],
        run_dir: str | None = None,
        write_event_log: bool = True,
        poll_interval_seconds: float = 0.02,
        raise_on_exception: bool = True,
        on_item_end: Callable[[WorkItem[Any], str], None] | None = None,
        gap_watch: str | None = None,
        status_every: float | None = 2.0,
    ):
        self.pipeline = pipeline
        #: Seconds between writes of ``async_status.json`` under ``run_dir`` (a live snapshot of places, edges
        #: and resources for a dashboard); ``None`` turns it off.
        self.status_every = status_every
        self._status_written_at = float("-inf")
        self._lineage_alive: dict[int, int] = {}  # lineage id -> items of it alive
        self._origin_alive: dict[str, set[int]] = {}  # read stage -> lineages alive from it
        self._edge_counts: Counter[tuple[str, str, str | None]] = Counter()
        self._edge_recent: dict[tuple[str, str, str | None], deque[float]] = {}
        self._terminals: Counter[tuple[str, str]] = Counter()
        #: When set, every fire records how many completions of this stage happened since the item was created
        #: (its distribution per stage is ``stats["gaps"]``; "commit" in the shipped pipelines).
        self.gap_watch = gap_watch
        self.resources = dict(resources)
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
        self._in_flight_units: dict[str, float] = dict.fromkeys(self.resources, 0.0)

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
        self.pipeline.validate(shared=[n for n in self.resources if n not in self.pipeline.resources])
        for name, stage in self.pipeline.stages.items():
            for res_name, units in stage.nominal_demand.items():
                cap = getattr(self.resources[res_name], "cap", None)
                if cap is not None and units > cap:
                    raise PipelineError(f"stage {name!r} demands {units} of {res_name!r}, whose cap is {cap}")
        for resource in self.resources.values():
            resource.open()
        self.events.emit(
            "run_start", stages=list(self.pipeline.stages), resources=list(self.resources), topology=self.topology()
        )
        self._write_status(force=True)
        try:
            while True:
                self._drain_completions()
                if self._fatal is not None:
                    break
                if self._stop_requested and not self._stopping:
                    self._begin_stop(self._stop_reason or "requested")
                fired = self._fire_all()
                self.events.tick(
                    dict(self.inflight),
                    dict(self._in_flight_units),
                    {name: len(ids) for name, ids in self._origin_alive.items()},
                )
                self._write_status()
                busy = any(self.inflight.values())
                if not busy and not fired:
                    # Nothing runs and nothing can fire: the drain is over, or (items waiting) a wiring error.
                    if not self._stopping:
                        self._begin_stop("quiescent")
                    break
                if not fired and busy:
                    self._wait_for_completion()
        finally:
            for ex in self._executors.values():
                ex.shutdown(wait=True, cancel_futures=False)
            self._drain_completions(force=True)
            # Items still waiting in a consume place cannot run any more: end them.
            for name, stage in self.pipeline.stages.items():
                if stage.mode != "consume":
                    continue
                for leftover in list(self.places[name]):
                    self._end_item(leftover, "unfinished")
                self.places[name].clear()
            for resource in self.resources.values():
                resource.close()
            self.events.stop_reason = self._stop_reason
            self.stats = self.events.summary()
            self.stats["counters"] = dict(self.counters)
            self.events.emit("run_end", **{k: v for k, v in self.stats.items() if k not in ("stages", "resources")})
            self._write_status(force=True)
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

    # ------------------------------------------------------------------ firing

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
            if stage.max_in_flight is not None and len(self._origin_alive.get(stage.name, ())) >= stage.max_in_flight:
                return False
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
        item.usage = {}  # what this body reports used; a body that raises before annotating is charged nothing
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
        """Take ``demand`` from the resources. All or nothing. ``None`` = not now."""
        taken: dict[str, float] = {}
        for name, units in demand.items():
            if units <= 0:
                continue
            need = units
            resource = self.resources.get(name)
            if resource is None:
                raise RuntimeError(f"stage {stage.name!r} demands unknown resource {name!r}")
            try:
                got = resource.take(need)
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
                self.resources[name].put(units)  # never counted as taken or in flight: the demand was not met whole

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
            # The read item is synthetic; each item the body returned starts a lineage from this stage.
            for out in outputs:
                self._register_new(out, parent=None, origin=stage.name)
            if not outputs:
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
        # A fan-out (or a replacement): the input ends; children descend from it and stay in its lineage.
        for out in outputs:
            if out is item:
                continue
            self._register_new(out, parent=item)
            out.fanout = len(outputs)
        if continuing:
            self._route(stage, item)
        else:
            self._end_item(item, "fanned_out", log=False)
        for out in children:
            self._route(stage, out)

    def _settle(
        self, stage: Stage[Any, Any], item: WorkItem[Any], taken: Mapping[str, float], used: Mapping[str, float]
    ) -> None:
        """Put back demand minus used."""
        for name, units in taken.items():
            if units <= 0:
                continue
            u = float(used.get(name, 0.0))
            back = max(0.0, units - u)
            self.resources[name].put(back, used=min(u, units))
            self.events.resource(name).put_back += back
            self.events.resource(name).used += min(u, units)
            self._in_flight_units[name] -= units

    def _route(self, stage: Stage[Any, Any], item: WorkItem[Any]) -> None:
        tag, item.tag = item.tag, None
        dst = self.pipeline.successor(stage.name, tag)
        if dst is None:
            self._terminals[(stage.name, tag or "finished")] += 1
            self._end_item(item, tag or "finished")
            return
        key = (stage.name, dst.name, tag)
        self._edge_counts[key] += 1
        self._edge_recent.setdefault(key, deque()).append(time.monotonic())
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

    # ------------------------------------------------------------------ status

    def topology(self) -> dict[str, Any]:
        """The pipeline as drawn: stages in declaration order with their place mode, body kind, widths, nominal
        demand and in-flight cap; the labelled edges; the resources with their caps. Static for the run."""
        stages = []
        for name, stage in self.pipeline.stages.items():
            stages.append(
                {
                    "name": name,
                    "mode": stage.mode,
                    "kind": "worker" if isinstance(stage, WorkerStage) else "head",
                    "workers": stage.workers,
                    "max_workers": stage.max_workers,
                    "max_queue": stage.max_queue,
                    "bandwidth": stage.bandwidth,
                    "demand": dict(stage.nominal_demand),
                    "max_in_flight": stage.max_in_flight,
                }
            )
        edges = [{"src": src, "dst": dst, "label": label} for (src, label), dst in self.pipeline.edges.items()]
        resources = {name: {"cap": getattr(r, "cap", None)} for name, r in self.resources.items()}
        return {"stages": stages, "edges": edges, "resources": resources}

    def snapshot(self) -> dict[str, Any]:
        """The run right now: how full each place is, what runs, what crossed each edge and at what rate over the
        last minute, what ended where, and what each resource has left. Engine thread only."""
        now = time.monotonic()
        edges = {}
        for key, n in self._edge_counts.items():
            recent = self._edge_recent.get(key)
            if recent is not None:
                while recent and now - recent[0] > 60.0:
                    recent.popleft()
            src, dst, label = key
            edges[f"{src}->{dst}" + (f":{label}" if label else "")] = {
                "src": src,
                "dst": dst,
                "label": label,
                "count": n,
                "per_min": len(recent) if recent is not None else 0,
            }
        stages = {}
        for name, stage in self.pipeline.stages.items():
            stages[name] = {
                "waiting": len(self.places[name]),
                "in_flight": self.inflight[name],
                "workers": stage.max_workers,
                "completed": self.counters[name],
                "lineages": len(self._origin_alive.get(name, ())) if stage.mode == "read" else None,
            }
        resources = {}
        for name, r in self.resources.items():
            stats = self.events.resource(name)
            available = r.available
            resources[name] = {
                "available": available() if callable(available) else available,
                "cap": getattr(r, "cap", None),
                "in_flight": self._in_flight_units.get(name, 0.0),
                "taken": stats.taken,
                "used": stats.used,
                "refused": stats.refused,
            }
        return {
            "t": round(self.events.now(), 3),
            "topology": self.topology(),
            "stopping": self._stopping,
            "stop_reason": self._stop_reason,
            "stages": stages,
            "edges": edges,
            "terminals": {f"{stage}:{outcome}": n for (stage, outcome), n in self._terminals.items()},
            "outcomes": dict(self.events.outcomes),
            "resources": resources,
        }

    def _write_status(self, force: bool = False) -> None:
        if self.status_every is None or self.run_dir is None:
            return
        now = time.monotonic()
        if not force and now - self._status_written_at < self.status_every:
            return
        self._status_written_at = now
        path = os.path.join(self.run_dir, "async_status.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.snapshot(), fh, default=str)
            os.replace(tmp, path)
        except OSError as e:  # a full disk must not stop the run: give up on the snapshot, say so once
            self.status_every = None
            logging.getLogger(__name__).warning("async_status.json not written (%r); giving up on it for this run", e)

    # ------------------------------------------------------------------ items

    def _new_item(self, payload: Any, parent: WorkItem[Any] | None) -> WorkItem[Any]:
        item: WorkItem[Any] = WorkItem(payload=payload)
        self._register_new(item, parent)
        return item

    def _register_new(self, item: WorkItem[Any], parent: WorkItem[Any] | None, origin: str | None = None) -> None:
        item.id = self._next_id
        self._next_id += 1
        if parent is None:
            item.parent_id = None
            item.stamps = dict(self.counters)
            item.origin, item.lineage = origin, item.id
        else:
            item.parent_id = parent.id
            item.stamps = dict(parent.stamps)
            item.created_at = parent.created_at
            item.origin, item.lineage = parent.origin, parent.lineage
        if item.origin is not None:
            self._lineage_alive[item.lineage] = self._lineage_alive.get(item.lineage, 0) + 1
            self._origin_alive.setdefault(item.origin, set()).add(item.lineage)
        self.events.emit("item", item=item.id, parent=item.parent_id)

    def _end_item_at(self, stage: Stage[Any, Any], item: WorkItem[Any], outcome: str, **fields: Any) -> None:
        """End an item that is still in ``stage``'s place (a refused demand)."""
        if stage.mode == "consume":
            place = self.places[stage.name]
            for i, it in enumerate(place):
                if it is item:
                    del place[i]
                    break
        self._terminals[(stage.name, outcome)] += 1
        self._end_item(item, outcome, **fields)

    def _end_item(self, item: WorkItem[Any], outcome: str, log: bool = True, **fields: Any) -> None:
        if item.origin is not None and item.lineage in self._lineage_alive:
            left = self._lineage_alive[item.lineage] - 1
            if left > 0:
                self._lineage_alive[item.lineage] = left
            else:
                del self._lineage_alive[item.lineage]
                self._origin_alive.get(item.origin, set()).discard(item.lineage)
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
        for name, r in self.resources.items():
            lines.append(f"{name}: available {r.available}, in flight {self._in_flight_units.get(name, 0.0)}")
        return "\n".join(lines)
