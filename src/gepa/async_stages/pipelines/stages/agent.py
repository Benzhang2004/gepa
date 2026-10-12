# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stages whose bodies are coding-agent sessions, and the two that let a run reshape its own pipeline.

``AgentPropose`` materializes the parent candidate in a workspace copied from the task's source tree, writes the
feedback the evaluator gave, and runs one headless agent session (Claude Code by default) that edits the file;
the edited file is the child. Several proposers may run side by side under different *directions* (what the
session is asked to pursue); ``Route`` spreads proposals across them by weight. ``Spawner`` is the meta step: it
reads a one-page digest the harness builds (pool, places, per-direction ledger), asks a tool-less session for one
change to the pipeline (a new proposer with a direction, a resize, a retirement, a weight), and applies it on the
engine thread. The digest is built in ``demand`` and the change applied in ``after``, both on the engine thread;
only the agent call runs on a worker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gepa.async_stages.pipelines.stages.common import Deps, Proposal
from gepa.async_stages.resources import ResourceExhaustedError
from gepa.async_stages.stage import HeadStage, WorkerStage, WorkItem

# ---------------------------------------------------------------------------------------------- the agent CLI


@dataclass
class AgentResult:
    """What one headless session returned."""

    text: str
    structured: dict[str, Any] | None
    usd: float
    tokens: int
    seconds: float
    turns: int
    model: str
    error: str | None


#: A runner: ``(prompt, cwd, options) -> AgentResult``. Tests inject a fake; the default shells out to ``claude``.
AgentRunner = Callable[[str, Path, "AgentOptions"], AgentResult]


@dataclass
class AgentOptions:
    max_usd: float = 8.0
    timeout: float = 1800.0
    model: str | None = None
    #: Tools the session may use; ``None`` = no tools at all (the meta step).
    allowed_tools: list[str] | None = None
    schema: dict[str, Any] | None = None


DEFAULT_TOOLS = [
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Bash(python3:*)",
    "Bash(python:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(wc:*)",
    "Bash(diff:*)",
]


def run_claude(prompt: str, cwd: Path, opts: AgentOptions) -> AgentResult:
    """One fresh ``claude -p`` session in ``cwd``: no memory of any other call."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-budget-usd", str(opts.max_usd)]
    if opts.model:
        cmd += ["--model", opts.model]
    if opts.schema is not None:
        cmd += ["--json-schema", json.dumps(opts.schema)]
    if opts.allowed_tools is None:
        cmd += ["--tools", ""]
    else:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", *opts.allowed_tools]
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    t0 = time.time()
    try:
        # stdin closed: with a pipe or inherited descriptor on stdin, ``claude -p`` waits for it to end.
        p = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(cwd), timeout=opts.timeout, env=env, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        return AgentResult("", None, 0.0, 0, time.time() - t0, 0, "", f"agent timed out after {opts.timeout:.0f}s")
    secs = time.time() - t0
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return AgentResult("", None, 0.0, 0, secs, 0, "", f"no JSON from claude (rc={p.returncode}): {p.stderr[-800:]}")
    u = d.get("usage") or {}
    tokens = sum(
        int(u.get(k) or 0)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    result = d.get("result")
    error = None
    if d.get("is_error"):
        error = result if isinstance(result, str) else json.dumps(result)
    elif opts.schema is not None and not d.get("structured_output"):
        error = f"no structured output (stop={d.get('stop_reason')} usd={d.get('total_cost_usd')})"
    return AgentResult(
        text=result if isinstance(result, str) else json.dumps(result),
        structured=d.get("structured_output"),
        usd=float(d.get("total_cost_usd") or 0.0),
        tokens=tokens,
        seconds=secs,
        turns=int(d.get("num_turns") or 0),
        model=next(iter((d.get("modelUsage") or {}).keys()), ""),
        error=error,
    )


# ---------------------------------------------------------------------------------------------- resources


class AgentSessions:
    """Concurrent agent sessions. Rented: a session's unit comes back when it returns."""

    def __init__(self, cap: int = 3, *, name: str = "agent"):
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
        return 0.0

    def put(self, units: float, used: float = 0.0) -> None:
        self._out = max(0.0, self._out - units - used)

    def open(self) -> None:
        return

    def close(self) -> None:
        return


# ---------------------------------------------------------------------------------------------- the task tree


@dataclass
class AgentTask:
    """What a proposer needs to know about the task: where the source tree is, which file the candidate is, and
    the words for the session. ``feedback`` renders the evaluator's side info for the session."""

    source: str
    editable: str
    spec: str
    #: Paths under ``source`` not copied into a workspace (version control, run outputs).
    ignore: tuple[str, ...] = (".git", "runs", "__pycache__", ".venv")
    feedback: Callable[[float, dict[str, Any]], str] | None = None
    #: Higher is better; how a score reads for the session (``"score"`` by default).
    score_name: str = "score"
    #: Tools a proposer session may use for this task; ``None`` = :data:`DEFAULT_TOOLS`.
    tools: list[str] | None = None


def _default_feedback(score: float, info: Mapping[str, Any]) -> str:
    keep = {k: v for k, v in info.items() if not str(k).startswith("_")}
    return json.dumps(keep, default=str)[:4000]


# ---------------------------------------------------------------------------------------------- propose


class AgentPropose(WorkerStage[Proposal, Proposal]):
    """One agent session edits the parent candidate in a fresh workspace; the edited file is the child.

    ``direction`` is what the session is asked to pursue (the per-proposer specialisation); an empty direction
    means "improve it". The session sees the task spec, the parent's score and feedback, the pool's best score
    and the direction, nothing else: no other proposer's workspace, no earlier session.
    """

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        task: AgentTask,
        *,
        name: str = "propose",
        direction: str = "",
        runner: AgentRunner = run_claude,
        options: AgentOptions | None = None,
        usd_first_guess: float = 4.0,
        keep_workspaces: bool = False,
        **kw: Any,
    ):
        kw.setdefault("demand", {"agent": 1, "usd": usd_first_guess})
        super().__init__(name, bandwidth="agent", **kw)
        self.deps = deps
        self.task = task
        self.direction = direction
        self.runner = runner
        self.options = options or AgentOptions()
        if self.options.allowed_tools is None:
            self.options.allowed_tools = list(task.tools or DEFAULT_TOOLS)
        self.keep_workspaces = keep_workspaces
        self.calls = 0
        self.usd = 0.0

    # -- what the session sees --------------------------------------------------------------------------

    def _candidate_text(self, candidate: dict[str, str]) -> str:
        key = self.deps.str_candidate_key
        return candidate[key] if key is not None else next(iter(candidate.values()))

    def _child(self, parent: dict[str, str], text: str) -> dict[str, str]:
        key = self.deps.str_candidate_key or next(iter(parent))
        child = dict(parent)
        child[key] = text
        return child

    def _feedback(self, p: Proposal) -> str:
        if p.parent_eval is None or not p.parent_eval.trajectories:
            return "(the parent has not been evaluated yet)"
        render = self.task.feedback or _default_feedback
        lines = []
        for score, info in zip(p.parent_eval.scores, p.parent_eval.trajectories, strict=False):
            lines.append(render(float(score), dict(info) if isinstance(info, Mapping) else {}))
        return "\n".join(lines)

    def _best(self) -> str:
        state = self.deps.box.state
        if state is None or not state.program_full_scores_val_set:
            return "unknown"
        return f"{max(state.program_full_scores_val_set):.6g}"

    def prompt(self, p: Proposal) -> str:
        parent_score = sum(p.parent_eval.scores) if p.parent_eval is not None else float("nan")
        direction = (
            self.direction.strip() or "Improve the candidate in whatever way the feedback suggests is most promising."
        )
        return (
            f"{self.task.spec.strip()}\n\n"
            f"You are one of several independent sessions improving `{self.task.editable}` in this directory; you "
            f"start from a candidate that scores {parent_score:.6g} ({self.task.score_name}, higher is better). The best "
            f"score reached so far across all sessions is {self._best()}.\n\n"
            f"YOUR DIRECTION: {direction}\n\n"
            f"FEEDBACK ON THE STARTING CANDIDATE:\n{self._feedback(p)}\n\n"
            f"Edit `{self.task.editable}` in place. Keep it correct: a wrong answer scores nothing. Work in small "
            f"measured steps: make the first improvement early, measure it, keep the file in a working state after "
            f"every step, since the file as it stands when the session ends (or times out) is what is kept. Do not "
            f"create other files that the candidate needs; only `{self.task.editable}` is kept."
        )

    # -- the body -----------------------------------------------------------------------------------------

    def _workspace(self, item: WorkItem[Proposal]) -> Path:
        base = Path(self.deps.run_dir or ".") / "agents" / self.name
        ws = base / f"item{item.id}"
        if ws.exists():
            shutil.rmtree(ws)
        shutil.copytree(self.task.source, ws, ignore=shutil.ignore_patterns(*self.task.ignore))
        return ws

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        parent_text = self._candidate_text(p.parent)
        ws = self._workspace(item)
        target = ws / self.task.editable
        target.write_text(parent_text)
        prompt = self.prompt(p)
        (ws / "PROMPT.md").write_text(prompt)
        res = self.runner(prompt, ws, self.options)
        item.usage = {"usd": res.usd}
        child_text = target.read_text() if target.exists() else parent_text
        p.proposal_meta = {
            "stage": self.name,
            "direction": self.direction,
            "usd": res.usd,
            "tokens": res.tokens,
            "seconds": round(res.seconds, 1),
            "turns": res.turns,
            "model": res.model,
            "error": res.error,
        }
        if not self.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)
        if res.error and child_text == parent_text:
            item.tag = "agent_error"
            return [item]
        if child_text == parent_text:
            item.tag = "no_proposal"
            return [item]
        p.child = self._child(p.parent, child_text)
        return [item]

    def after(self, item: WorkItem[Proposal]) -> None:
        self.calls += 1
        self.usd += float(item.usage.get("usd", 0.0))
        if self.calls and self.usd > 0:
            self._demand["usd"] = self.usd / self.calls  # re-fit the per-call estimate from what sessions cost
        entry = self.deps.ledger.setdefault(self.name, _ledger_entry(self.direction))
        entry["sessions"] += 1
        entry["usd"] += float(item.usage.get("usd", 0.0))
        meta = item.payload.proposal_meta
        if item.tag is None:
            entry["proposed"] += 1
        elif item.tag == "agent_error":
            entry["errors"] += 1
        self.deps.logger.log(
            f"{self.name}: session {item.tag or 'proposed'} after {meta.get('seconds')}s, {meta.get('turns')} turns, "
            f"${float(meta.get('usd') or 0):.2f}" + (f"; error: {meta['error']}" if meta.get("error") else "")
        )


def _ledger_entry(direction: str) -> dict[str, Any]:
    return {
        "direction": direction,
        "sessions": 0,
        "proposed": 0,
        "accepted": 0,
        "committed": 0,
        "errors": 0,
        "usd": 0.0,
        "scores": [],
    }


# ---------------------------------------------------------------------------------------------- route


class Route(HeadStage[Proposal, Proposal]):
    """Spreads proposals across the live proposer stages by weight: tags each item with the proposer's name, which
    is the label of the edge that leads there. The spawner edits ``weights`` as it adds or retires proposers."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, weights: Mapping[str, float] | None = None, **kw: Any):
        super().__init__("route", **kw)
        self.deps = deps
        self.weights: dict[str, float] = dict(weights or {})

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        live = [(name, w) for name, w in self.weights.items() if w > 0]
        if not live:
            item.tag = "no_proposer"
            return [item]
        total = sum(w for _, w in live)
        r = self.deps.rng.random() * total
        for name, w in live:
            r -= w
            if r <= 0:
                item.tag = name
                return [item]
        item.tag = live[-1][0]
        return [item]


class LedgerGate(HeadStage[Proposal, Proposal]):
    """Books the gate's verdict and the commit's score against the proposer that made the proposal. Sits after the
    gate (``accepted`` / ``rejected`` pass through unchanged) and after commit."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, name: str, what: str, **kw: Any):
        super().__init__(name, **kw)
        self.deps = deps
        self.what = what  # "gate" or "commit"

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        stage = p.proposal_meta.get("stage")
        if self.what == "gate":
            item.tag = p.proposal_meta.get("verdict")  # re-tag: the gate's verdict picks the edge out of here
        if stage:
            entry = self.deps.ledger.setdefault(stage, _ledger_entry(str(p.proposal_meta.get("direction", ""))))
            if self.what == "gate":
                if item.tag == "accepted":
                    entry["accepted"] += 1
            else:
                entry["committed"] += 1
                if p.valset_evaluation is not None:
                    scores = list(p.valset_evaluation.scores_by_val_id.values())
                    if scores:
                        entry["scores"].append(round(sum(scores) / len(scores), 6))
                        entry["scores"] = entry["scores"][-10:]
        return [item]


# ---------------------------------------------------------------------------------------------- spawner


class Tee(HeadStage[Proposal, Proposal]):
    """After commit: the committed item goes on to the pool; every ``every`` commits a copy tagged ``meta`` goes to
    the spawner as well."""

    in_type = out_type = Proposal

    def __init__(self, deps: Deps, *, every: int = 2, **kw: Any):
        super().__init__("tee", **kw)
        self.deps = deps
        self.every = max(1, int(every))
        self.seen = 0

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        self.seen += 1
        if self.seen % self.every != 0:
            return [item]
        meta: WorkItem[Proposal] = WorkItem(Proposal(parent_idx=item.payload.parent_idx, parent=item.payload.parent))
        meta.tag = "meta"
        return [item, meta]


MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["none", "add_proposer", "set_workers", "set_weight", "retire"]},
        "stage": {"type": "string"},
        "direction": {"type": "string"},
        "workers": {"type": "integer"},
        "weight": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


class Spawner(WorkerStage[Proposal, Proposal]):
    """The meta step. Reads the digest (built in ``demand``, on the engine thread), asks a tool-less session for one
    change, applies it in ``after`` through the engine's mutation API. The change is drawn from a small catalogue:
    add a proposer under a new direction, resize a stage, reweight a proposer, retire one.

    Cheap by construction: a one-page prompt, a manifest as the only output, no tools, its own small budget cap.
    """

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        task: AgentTask,
        route: Route,
        *,
        new_proposer: Callable[[str, str], AgentPropose],
        downstream: str = "screen",
        runner: AgentRunner = run_claude,
        options: AgentOptions | None = None,
        max_proposers: int = 6,
        max_workers_per_stage: int = 4,
        usd_first_guess: float = 0.5,
        **kw: Any,
    ):
        kw.setdefault("demand", {"meta": 1, "usd": usd_first_guess})
        kw.setdefault("max_queue", 1)
        super().__init__("spawner", bandwidth="meta", **kw)
        self.deps = deps
        self.task = task
        self.route = route
        self.new_proposer = new_proposer
        self.downstream = downstream
        self.runner = runner
        self.options = options or AgentOptions(max_usd=1.0, timeout=600.0, allowed_tools=None, schema=MANIFEST_SCHEMA)
        self.options.schema = MANIFEST_SCHEMA
        self.options.allowed_tools = None
        self.max_proposers = max_proposers
        self.max_workers_per_stage = max_workers_per_stage
        self.decisions: list[dict[str, Any]] = []
        self._digest: str = ""
        self._manifest: dict[str, Any] | None = None
        self.calls = 0
        self.usd = 0.0

    # -- digest (engine thread) ---------------------------------------------------------------------------

    def digest(self) -> str:
        engine = self.deps.engine
        snap = engine.snapshot() if engine is not None else {"stages": {}, "edges": {}, "outcomes": {}, "resources": {}}
        state = self.deps.box.state
        pool: dict[str, Any] = {"candidates": 0}
        if state is not None:
            scores = list(state.program_full_scores_val_set)
            pool = {
                "candidates": len(scores),
                "best": max(scores) if scores else None,
                "seed": scores[0] if scores else None,
                "recent": [round(s, 6) for s in scores[-8:]],
            }
        stages = {
            name: {k: v for k, v in st.items() if k in ("waiting", "in_flight", "workers", "completed")}
            for name, st in snap["stages"].items()
        }
        usd = snap["resources"].get("usd", {})
        lines = [
            f"TASK: {self.task.spec.strip()[:600]}",
            f"POOL: {json.dumps(pool)}",
            f"STAGES (waiting / in flight / workers / completed): {json.dumps(stages)}",
            "FLOW (items per minute on each edge): " + json.dumps({k: v["per_min"] for k, v in snap["edges"].items()}),
            f"OUTCOMES: {json.dumps(snap['outcomes'])}",
            f"USD: used {usd.get('used', 0):.2f} of cap {usd.get('cap')}",
            "PROPOSERS (direction, sessions, proposed, accepted, committed, errors, usd, last committed scores):",
        ]
        for name, e in self.deps.ledger.items():
            lines.append(
                f"  {name}: {e['direction'] or '(none)'} | {e['sessions']} / {e['proposed']} / {e['accepted']} / "
                f"{e['committed']} / {e['errors']} / ${e['usd']:.2f} / {e['scores']}"
            )
        lines.append(f"ROUTE WEIGHTS: {json.dumps(self.route.weights)}")
        return "\n".join(lines)

    def prompt(self) -> str:
        return (
            "You are the meta step of a swarm of coding-agent sessions that improve one candidate program. You have "
            "no tools and no memory: everything you know is the digest below, built by the harness from "
            "measurements. Choose ONE change to the pipeline, or none, so that the next sessions are spent where "
            "they are most likely to improve the best score. Think about which directions have paid off, which are "
            "dead, whether proposals pile up waiting for evaluation (then evaluation needs workers, not proposals), "
            "and whether a direction nobody is trying is likely to hold the next breakthrough.\n\n"
            "Actions: add_proposer (give `direction`: a concrete operation family for the sessions to pursue, one "
            "sentence); set_workers (give `stage` and `workers`); set_weight (give `stage`, a proposer, and "
            "`weight` >= 0; 0 pauses it); retire (give `stage`, a proposer); none.\n\n"
            f"DIGEST:\n{self.digest()}\n"
        )

    def demand(self, item: WorkItem[Proposal]) -> dict[str, float]:
        self._digest = self.digest()
        return dict(self._demand)

    # -- the body (worker) --------------------------------------------------------------------------------

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        prompt = self.prompt()
        ws = Path(self.deps.run_dir or ".") / "agents" / "spawner"
        ws.mkdir(parents=True, exist_ok=True)
        res = self.runner(prompt, ws, self.options)
        item.usage = {"usd": res.usd}
        item.payload.proposal_meta = {"usd": res.usd, "seconds": round(res.seconds, 1), "error": res.error}
        self._manifest = res.structured if isinstance(res.structured, dict) else None
        if self._manifest is None:
            item.tag = "no_manifest"
        return [item]

    # -- apply (engine thread) ----------------------------------------------------------------------------

    def after(self, item: WorkItem[Proposal]) -> None:
        self.calls += 1
        self.usd += float(item.usage.get("usd", 0.0))
        if self.usd > 0:
            self._demand["usd"] = self.usd / self.calls
        manifest, self._manifest = self._manifest, None
        decision: dict[str, Any] = {"t": time.time(), "manifest": manifest, "applied": None}
        self.decisions.append(decision)
        if manifest is None:
            decision["applied"] = "no manifest"
            self._log(decision)
            return
        try:
            decision["applied"] = self.apply(manifest)
        except Exception as e:  # a bad manifest never stops the run
            decision["applied"] = f"refused: {e!r}"
        self._log(decision)
        if self.deps.run_dir:
            path = Path(self.deps.run_dir) / "spawner_decisions.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(decision, default=str) + "\n")

    def _log(self, decision: dict[str, Any]) -> None:
        m = decision["manifest"] or {}
        self.deps.logger.log(
            f"Spawner: {m.get('action', 'none')} {m.get('stage', '')} {m.get('direction', '')!r} -> "
            f"{decision['applied']} ({m.get('reason', '')})"
        )

    def apply(self, m: Mapping[str, Any]) -> str:
        engine = self.deps.engine
        if engine is None:
            return "no engine"
        action = m.get("action", "none")
        proposers = set(self.route.weights)
        if action == "none":
            return "none"
        if action == "add_proposer":
            if len([n for n, w in self.route.weights.items() if w > 0]) >= self.max_proposers:
                return f"refused: {self.max_proposers} proposers already"
            direction = str(m.get("direction", "")).strip()
            if not direction:
                return "refused: add_proposer needs a direction"
            name = f"propose_{len(proposers)}"
            while name in engine.pipeline.stages:
                name = name + "x"
            stage = self.new_proposer(name, direction)
            engine.add_stage(stage)
            engine.add_edge("route", name, name)
            engine.add_edge(name, self.downstream)
            self.route.weights[name] = 1.0
            self.deps.ledger.setdefault(name, _ledger_entry(direction))
            return f"added {name}"
        stage_name = str(m.get("stage", ""))
        if stage_name not in engine.pipeline.stages:
            return f"refused: no stage {stage_name!r}"
        if action == "set_workers":
            n = int(m.get("workers", 1))
            n = max(0, min(n, self.max_workers_per_stage))
            if stage_name in ("spawner", "tee", "route", "commit", "gate"):
                return f"refused: {stage_name} is not resizable"
            engine.set_workers(stage_name, n)
            return f"{stage_name} workers={n}"
        if action == "set_weight":
            if stage_name not in proposers:
                return f"refused: {stage_name} is not a proposer"
            w = max(0.0, float(m.get("weight", 1.0)))
            self.route.weights[stage_name] = w
            return f"{stage_name} weight={w}"
        if action == "retire":
            if stage_name not in proposers:
                return f"refused: {stage_name} is not a proposer"
            if (
                len([n for n, w in self.route.weights.items() if w > 0]) <= 1
                and self.route.weights.get(stage_name, 0) > 0
            ):
                return "refused: the last live proposer"
            self.route.weights[stage_name] = 0.0
            engine.retire_stage(stage_name)
            return f"retired {stage_name}"
        return f"refused: unknown action {action!r}"


class MetaSessions(AgentSessions):
    """The meta step's own lane, so it never waits behind the proposers' sessions."""

    def __init__(self, cap: int = 1, *, name: str = "meta"):
        super().__init__(cap, name=name)


__all__ = [
    "DEFAULT_TOOLS",
    "MANIFEST_SCHEMA",
    "AgentOptions",
    "AgentPropose",
    "AgentResult",
    "AgentRunner",
    "AgentSessions",
    "AgentTask",
    "LedgerGate",
    "MetaSessions",
    "Route",
    "Spawner",
    "Tee",
    "run_claude",
]
