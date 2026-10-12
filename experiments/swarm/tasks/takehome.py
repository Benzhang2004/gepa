"""Anthropic's VLIW SIMD performance take-home as an optimize_anything task.

The candidate is the text of ``perf_takehome.py``; the score is ``BASELINE / cycles`` (1.0 at the reference
kernel, about 108 at the hardest public threshold of 1,363 cycles), 0 for an incorrect or crashing kernel. The
evaluator is frozen: a clean temp dir with the pristine ``problem.py`` and ``tests/frozen_problem.py``, four random
forests, exact output check, run under ``python -I`` so nothing in the candidate's directory is importable. A static
check refuses imports and calls that would let a kernel read the reference or escape the simulator.

``TAKEHOME_SRC`` names the take-home repo (a clone of anthropics/original_performance_takehome).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from gepa.async_stages.pipelines.stages.agent import AgentTask

EDITABLE = "perf_takehome.py"
BASELINE = 147734
THRESHOLDS = [18532, 2164, 1790, 1579, 1548, 1487, 1363]
FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "importlib", "inspect", "ctypes", "socket", "shutil", "pathlib"}

EVAL_SCRIPT = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
from frozen_problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES
sys.path.insert(0, sys.argv[2])
from perf_takehome import KernelBuilder
cycles = []
for i in range(4):
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    m = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    m.enable_pause = False
    m.enable_debug = False
    m.run()
    for ref in reference_kernel2(mem):
        pass
    p = ref[6]
    if m.mem[p:p + len(inp.values)] != ref[p:p + len(inp.values)]:
        print(json.dumps({"status": "incorrect", "cycles": None, "diag": "output values differ from the reference on trial %d" % i}))
        sys.exit(0)
    cycles.append(m.cycle)
print(json.dumps({"status": "ok", "cycles": max(cycles), "all": cycles}))
"""

SPEC = """TASK: Anthropic performance take-home. Make the kernel built by `KernelBuilder.build_kernel` in
`perf_takehome.py` run in as few cycles as possible on the simulated VLIW SIMD machine defined in `problem.py`
(single core; N_CORES must stay 1). The score is BASELINE / cycles, so halving the cycles doubles the score.
Correctness is checked exactly against the reference on random forests; an incorrect kernel scores 0.
Read `problem.py` for the machine (engines, slots per bundle, VLEN, instruction set) and `Readme.md` for the rules.
Measure with `python3 tests/submission_tests.py` (prints CYCLES) before you finish. Only `perf_takehome.py` is
kept; `problem.py` and the tests are pristine copies and must not be changed."""


def source_dir() -> Path:
    src = os.environ.get("TAKEHOME_SRC")
    if not src:
        raise SystemExit("set TAKEHOME_SRC to the take-home repo (a clone of anthropics/original_performance_takehome)")
    return Path(src).resolve()


def static_check(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [a.name.split(".")[0] for a in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "").split(".")[0]]
            )
            for n in names:
                if n in FORBIDDEN_IMPORTS:
                    return f"forbidden import: {n}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "__import__"}:
            return f"forbidden call: {node.func.id}"
    return None


def run_frozen(code: str, src: Path, timeout: float = 900.0) -> dict[str, Any]:
    reason = static_check(code)
    if reason:
        return {"status": "forbidden", "cycles": None, "diag": reason}
    with tempfile.TemporaryDirectory(prefix="takehome-eval-") as td:
        tdp = Path(td)
        (tdp / "frozen").mkdir()
        shutil.copy(src / "tests" / "frozen_problem.py", tdp / "frozen" / "frozen_problem.py")
        (tdp / "cand").mkdir()
        shutil.copy(src / "problem.py", tdp / "cand" / "problem.py")
        (tdp / "cand" / EDITABLE).write_text(code)
        (tdp / "eval.py").write_text(EVAL_SCRIPT)
        try:
            p = subprocess.run(
                [sys.executable, "-I", str(tdp / "eval.py"), str(tdp / "frozen"), str(tdp / "cand")],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tdp),
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "cycles": None, "diag": f"evaluation timed out after {timeout:.0f}s"}
        out = p.stdout.strip().splitlines()
        if p.returncode != 0 or not out:
            return {"status": "error", "cycles": None, "diag": "evaluator crashed:\n" + (p.stderr or "")[-1500:]}
        try:
            return json.loads(out[-1])
        except json.JSONDecodeError:
            return {"status": "error", "cycles": None, "diag": "unparseable evaluator output: " + out[-1][:300]}


def feedback(score: float, info: dict[str, Any]) -> str:
    cycles = info.get("cycles")
    if info.get("status") == "ok" and cycles:
        passed = [t for t in THRESHOLDS if cycles < t]
        nxt = min((t for t in THRESHOLDS if cycles >= t), default=None)
        line = f"status ok, {cycles} cycles (score {score:.3f}), {len(passed)} of {len(THRESHOLDS)} public thresholds passed"
        return line + (f"; next threshold is under {nxt} cycles" if nxt else "; every public threshold passed")
    return f"status {info.get('status')}, score 0: {info.get('diag', '')}"


def make(src: Path | None = None) -> tuple[AgentTask, str, Any]:
    """The task tree, the seed candidate and the evaluator."""
    src = src or source_dir()
    seed = (src / EDITABLE).read_text()

    def evaluator(candidate: str, example: Any = None, **kw: Any) -> tuple[float, dict[str, Any]]:
        r = run_frozen(candidate, src)
        cycles = r.get("cycles")
        score = BASELINE / cycles if r.get("status") == "ok" and cycles else 0.0
        return score, {"status": r.get("status"), "cycles": cycles, "diag": r.get("diag", ""), "all": r.get("all")}

    task = AgentTask(source=str(src), editable=EDITABLE, spec=SPEC, feedback=feedback, score_name="BASELINE/cycles")
    return task, seed, evaluator
