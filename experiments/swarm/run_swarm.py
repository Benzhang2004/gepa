"""Run a frontier task through the async backend's agent pipeline.

    TAKEHOME_SRC=/path/to/original_performance_takehome \
    uv run python -m experiments.swarm.run_swarm --task takehome --run-dir runs/swarm/t1 \
        --max-usd 40 --sessions 3 --directions "" "vectorize" --dynamic

Static arm: one proposer per ``--directions`` entry. ``--dynamic`` adds the spawner, which reshapes the pipeline
every ``--meta-every`` commits. The dashboard follows the run: ``python -m gepa.async_stages.dashboard <run-dir>``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

from gepa.oa.config import OptimizeAnythingConfig
from gepa.optimize_anything import optimize_anything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="takehome")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-usd", type=float, default=40.0)
    ap.add_argument("--max-evals", type=int, default=400)
    ap.add_argument("--sessions", type=int, default=3, help="concurrent agent sessions (the agent resource cap)")
    ap.add_argument("--eval-concurrency", type=int, default=4)
    ap.add_argument("--directions", nargs="*", default=[""])
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--meta-every", type=int, default=2)
    ap.add_argument("--in-flight", type=int, default=None, help="proposals alive at once (default: sessions)")
    ap.add_argument("--session-max-usd", type=float, default=8.0)
    ap.add_argument("--session-timeout", type=float, default=1800.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default="high", help="reasoning effort for proposer sessions")
    ap.add_argument("--meta-effort", default="medium", help="reasoning effort for the meta step")
    ap.add_argument("--keep-workspaces", action="store_true")
    ap.add_argument("--brief", action="store_true", help="one study session writes a brief every session reads")
    ap.add_argument("--plan", type=int, default=0, help="planner fan-out: K sub-tasks per proposal (0 = off)")
    ap.add_argument("--notes", type=int, default=3, help="recent session notes shown to each session")
    a = ap.parse_args()

    mod = importlib.import_module(f"experiments.swarm.tasks.{a.task}")
    task, seed, evaluator = mod.make()
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sessions = max(1, a.sessions)
    in_flight = a.in_flight or sessions
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    t0 = time.time()
    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective=task.spec,
        config=OptimizeAnythingConfig(
            engine="async",
            max_evals=a.max_evals,
            max_token_cost=a.max_usd,
            max_concurrency=a.eval_concurrency,
            run_dir=str(run_dir),
            output_dir=str(run_dir / "eval_server"),
            engine_config={
                "pipeline": {
                    "name": "agent",
                    "task": task,
                    "directions": a.directions,
                    "dynamic": a.dynamic,
                    "meta_every": a.meta_every,
                    "agent_options": {
                        "max_usd": a.session_max_usd,
                        "timeout": a.session_timeout,
                        "model": a.model,
                        "effort": a.effort,
                    },
                    "meta_options": {"max_usd": 1.0, "timeout": 600.0, "model": a.model, "effort": a.meta_effort},
                    "keep_workspaces": a.keep_workspaces,
                    "brief": a.brief,
                    "brief_options": {"max_usd": 3.0, "timeout": 900.0, "model": a.model, "effort": a.effort},
                    "plan": a.plan,
                    "plan_options": {"max_usd": 1.0, "timeout": 600.0, "model": a.model, "effort": a.meta_effort},
                    "notes": a.notes,
                },
                "resources": {"agent": {"cap": sessions}, "meta": {"cap": 1}},
                "stages": {
                    "sample": {"max_in_flight": in_flight},
                    "propose": {"workers": sessions},
                    "screen": {"workers": a.eval_concurrency},
                    "validate": {"workers": a.eval_concurrency},
                },
                "isolate_errors": True,
            },
        ),
    )
    stats = result.metadata.get("async", {})
    best = result.best_score
    summary = {
        "task": a.task,
        "best_score": best,
        "best_cycles": None if not best else round(mod.BASELINE / best) if hasattr(mod, "BASELINE") else None,
        "candidates": len(result.candidates) if hasattr(result, "candidates") else None,
        "total_evals": result.total_evals,
        "stop_reason": stats.get("stop_reason"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "resources": stats.get("resources"),
    }
    (run_dir / "best.py" if a.task == "takehome" else run_dir / "best.txt").write_text(str(result.best_candidate))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SWARM END " + json.dumps(summary, default=str), flush=True)
    plot_run(run_dir)


def plot_run(run_dir: Path) -> None:
    """The run's deliverable: best-so-far curves against wall time and dollars, drawn by the swarm repo's
    ``tools/pareto_plot.py`` (``SWARM_TOOLS`` names its directory) into ``<run_dir>/figs``."""
    tools = Path(os.environ.get("SWARM_TOOLS", "/data/benzhang04/swarm/tools"))
    script = tools / "pareto_plot.py"
    if not script.exists():
        print(f"no pareto plot: {script} not found (set SWARM_TOOLS)", flush=True)
        return
    import subprocess
    import sys

    r = subprocess.run([sys.executable, str(script), "--out", str(run_dir / "figs"), str(run_dir)], capture_output=True, text=True)
    print(("pareto figs under " + str(run_dir / "figs")) if r.returncode == 0 else ("pareto plot failed: " + r.stderr[-800:]), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
