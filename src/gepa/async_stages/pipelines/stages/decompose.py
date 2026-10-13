# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Splitting a coding-agent session into stages over shared places.

The trajectories of whole sessions show three serial blocks inside every one of them: orientation (re-reading the
task, 40-50% of session time), one long planning-and-rewrite turn, and findings that die with the session. The
pieces here move each block into a place that other stages read:

- :class:`Places`: the run's shared places on disk, ``brief/BRIEF.md`` and ``notes/*.md``.
- :class:`Brief`: one read-only session, run once after the seed is committed, writes the brief: the facts about
  the task and the machine that every later session would otherwise re-derive.
- :class:`Planner`: a tool-less session that reads the parent's feedback, the brief and the recent notes and emits
  K sub-tasks; the stage fans the proposal out into K children, each carrying its sub-task as its direction, so K
  implementer sessions run side by side instead of one session planning and then rewriting in a single turn.
- notes: every implementer session is asked to leave ``NOTES.md``; :class:`AgentPropose` files it into the notes
  place and the next prompts and the spawner's digest carry the recent ones.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from gepa.async_stages.pipelines.stages.agent import AgentOptions, AgentRunner, AgentTask, run_claude
from gepa.async_stages.pipelines.stages.common import Deps, Proposal
from gepa.async_stages.stage import WorkerStage, WorkItem

# ---------------------------------------------------------------------------------------------- places


class Places:
    """The shared places of a run: directories under ``<run_dir>/places``, items are write-once files."""

    def __init__(self, run_dir: str | None):
        self.root = Path(run_dir or ".") / "places"

    @property
    def brief_path(self) -> Path:
        return self.root / "brief" / "BRIEF.md"

    def brief(self) -> str | None:
        return self.brief_path.read_text() if self.brief_path.exists() else None

    def write_brief(self, text: str) -> None:
        self.brief_path.parent.mkdir(parents=True, exist_ok=True)
        self.brief_path.write_text(text)

    def add_note(self, name: str, text: str) -> Path:
        d = self.root / "notes"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{len(list(d.glob('*.md'))) + 1:04d}-{name}.md"
        path.write_text(text)
        return path

    def notes(self, k: int = 3, chars: int = 700) -> list[tuple[str, str]]:
        """The ``k`` most recent notes, newest first, each clipped to ``chars``."""
        d = self.root / "notes"
        if not d.exists():
            return []
        files = sorted(d.glob("*.md"))[-k:]
        return [(f.stem, f.read_text()[:chars]) for f in reversed(files)]

    def count_notes(self) -> int:
        d = self.root / "notes"
        return len(list(d.glob("*.md"))) if d.exists() else 0


def render_notes(notes: list[tuple[str, str]]) -> str:
    if not notes:
        return ""
    return "\n".join(f"--- {name}\n{text.strip()}" for name, text in notes)


# ---------------------------------------------------------------------------------------------- brief


BRIEF_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Write",
    "Bash(python3:*)",
    "Bash(python:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(wc:*)",
]


class Brief(WorkerStage[Proposal, Proposal]):
    """One session studies the task tree and writes ``BRIEF.md``: what every later session would otherwise spend
    its first quarter hour re-deriving. Runs once; the brief is read-only afterwards. Terminal: returns nothing."""

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        task: AgentTask,
        places: Places,
        *,
        runner: AgentRunner = run_claude,
        options: AgentOptions | None = None,
        usd_first_guess: float = 2.0,
        **kw: Any,
    ):
        kw.setdefault("demand", {"agent": 1, "usd": usd_first_guess})
        super().__init__("brief", bandwidth="agent", **kw)
        self.deps = deps
        self.task = task
        self.places = places
        self.runner = runner
        self.options = options or AgentOptions(max_usd=3.0, timeout=900.0, allowed_tools=list(BRIEF_TOOLS))
        if self.options.allowed_tools is None:
            self.options.allowed_tools = list(BRIEF_TOOLS)

    def prompt(self) -> str:
        return (
            f"{self.task.spec.strip()}\n\n"
            "You are the study step of a swarm of coding-agent sessions that will each improve "
            f"`{self.task.editable}` in a copy of this directory. They are fresh sessions with no memory; every one "
            "of them would otherwise spend its first quarter hour re-reading what you can read once. Write "
            "`BRIEF.md` in this directory, at most 250 lines, with exactly what an implementer needs before editing: "
            "the machine or problem model (the facts and numbers that decide what is fast, from the source, not "
            "from guesswork), the structure of the current candidate and where its time goes (measure it), how to "
            "measure a change quickly and how to check correctness, the rules on what may be changed, and the "
            "pitfalls you can see. Facts and measurements only; no plan and no opinions on what to try. "
            "Do not modify any other file."
        )

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        ws = Path(self.deps.run_dir or ".") / "agents" / "brief"
        if ws.exists():
            shutil.rmtree(ws)
        shutil.copytree(self.task.source, ws, ignore=shutil.ignore_patterns(*self.task.ignore))
        res = self.runner(self.prompt(), ws, self.options)
        item.usage = {"usd": res.usd}
        brief = ws / "BRIEF.md"
        text = brief.read_text() if brief.exists() else ""
        item.payload.proposal_meta = {
            "stage": "brief",
            "usd": res.usd,
            "seconds": round(res.seconds, 1),
            "error": res.error,
        }
        if text.strip():
            self.places.write_brief(text)
        shutil.rmtree(ws, ignore_errors=True)
        return []

    def after(self, item: WorkItem[Proposal]) -> None:
        meta = item.payload.proposal_meta
        self.deps.logger.log(
            f"brief: {'written' if self.places.brief() else 'missing'} after {meta.get('seconds')}s, "
            f"${float(meta.get('usd') or 0):.2f}" + (f"; error: {meta['error']}" if meta.get("error") else "")
        )


# ---------------------------------------------------------------------------------------------- planner


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"direction": {"type": "string"}, "why": {"type": "string"}},
                "required": ["direction"],
            },
        }
    },
    "required": ["subtasks"],
}


class Planner(WorkerStage[Proposal, Proposal]):
    """Reads the parent's feedback, the brief and the recent notes; emits up to ``k`` sub-tasks as a fan-out of
    the proposal, one child per sub-task, each carrying its direction. Tool-less, on the meta lane."""

    in_type = out_type = Proposal

    def __init__(
        self,
        deps: Deps,
        task: AgentTask,
        places: Places,
        *,
        k: int = 3,
        runner: AgentRunner = run_claude,
        options: AgentOptions | None = None,
        usd_first_guess: float = 0.5,
        **kw: Any,
    ):
        kw.setdefault("demand", {"meta": 1, "usd": usd_first_guess})
        super().__init__("plan", bandwidth="meta", **kw)
        self.deps = deps
        self.task = task
        self.places = places
        self.k = max(1, int(k))
        self.runner = runner
        self.options = options or AgentOptions(max_usd=1.0, timeout=600.0, allowed_tools=None, schema=PLAN_SCHEMA)
        self.options.allowed_tools = None
        self.options.schema = PLAN_SCHEMA
        self.calls = 0
        self.usd = 0.0

    def prompt(self, p: Proposal) -> str:
        from gepa.async_stages.pipelines.stages.agent import _default_feedback

        render = self.task.feedback or _default_feedback
        feedback = (
            "\n".join(
                render(float(s), dict(i) if isinstance(i, dict) else {})
                for s, i in zip(p.parent_eval.scores, p.parent_eval.trajectories or [], strict=False)
            )
            if p.parent_eval is not None
            else "(not evaluated)"
        )
        brief = self.places.brief() or "(no brief yet)"
        notes = render_notes(self.places.notes(5)) or "(none yet)"
        return (
            f"{self.task.spec.strip()}\n\n"
            "You are the planning step of a swarm of coding-agent sessions. You have no tools. From the brief, the "
            "feedback on the current candidate and the notes left by earlier sessions, name up to "
            f"{self.k} sub-tasks that independent sessions can pursue IN PARALLEL on the same starting candidate, each a "
            "concrete operation family with a one-sentence direction. Prefer sub-tasks whose gains add up and that "
            "the notes do not report as already tried and failed. Fewer, sharper sub-tasks beat vague ones.\n\n"
            f"BRIEF:\n{brief[:6000]}\n\nFEEDBACK ON THE CURRENT CANDIDATE:\n{feedback}\n\nRECENT NOTES:\n{notes}\n"
        )

    def run(self, item: WorkItem[Proposal]) -> list[WorkItem[Proposal]]:
        p = item.payload
        ws = Path(self.deps.run_dir or ".") / "agents" / "plan"
        ws.mkdir(parents=True, exist_ok=True)
        res = self.runner(self.prompt(p), ws, self.options)
        item.usage = {"usd": res.usd}
        subtasks = []
        if isinstance(res.structured, dict):
            for st in (res.structured.get("subtasks") or [])[: self.k]:
                d = str((st or {}).get("direction", "")).strip()
                if d:
                    subtasks.append(d)
        p.proposal_meta = {
            "stage": "plan",
            "usd": res.usd,
            "seconds": round(res.seconds, 1),
            "subtasks": subtasks,
            "error": res.error,
        }
        if not subtasks:
            item.tag = "no_plan" if res.error else None
            return [item]  # no plan: the proposal goes on undirected
        children: list[WorkItem[Proposal]] = []
        for d in subtasks:
            child = Proposal(
                parent_idx=p.parent_idx,
                parent=p.parent,
                minibatch_ids=list(p.minibatch_ids),
                minibatch=list(p.minibatch),
                parent_eval=p.parent_eval,
                components=p.components,
                proposal_meta={"direction": d, "planned": True},
            )
            children.append(WorkItem(child))
        return children

    def after(self, item: WorkItem[Proposal]) -> None:
        self.calls += 1
        self.usd += float(item.usage.get("usd", 0.0))
        if self.usd > 0:
            self._demand["usd"] = self.usd / self.calls
        meta = item.payload.proposal_meta
        subtasks = meta.get("subtasks") or []
        self.deps.logger.log(
            f"plan: {len(subtasks)} sub-tasks after {meta.get('seconds')}s, ${float(meta.get('usd') or 0):.2f}: "
            + " | ".join(s[:60] for s in subtasks)
            + (f"; error: {meta['error']}" if meta.get("error") else "")
        )


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "note"


__all__ = ["BRIEF_TOOLS", "PLAN_SCHEMA", "Brief", "Places", "Planner", "render_notes", "slug"]
