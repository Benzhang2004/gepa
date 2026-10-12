"""FrontierCS algorithmic problem 0, polyomino packing (Team-of-N's headline task), as an optimize_anything task.

The candidate is the text of ``solution.cpp``. The score is the mean over the 70 public tests of the checker's
ratio (cells / W*H, in (0, 1]); a test that fails to run, times out (2 s) or is rejected by the checker scores 0.
Compilation is static, through ``docker run gcc:13`` (havoc has no compiler on the host); the binaries then run
on the host. The checker is FrontierCS's own ``chk.cc`` (testlib), compiled the same way once and cached.

``POLYOMINO_SRC`` names the ``Frontier-CS/algorithmic`` directory. The task tree handed to proposer sessions
holds the statement, the public tests, a seed ``solution.cpp`` (every piece in a row), the checker binary and
two helpers the session may run: ``./build.sh`` (compile) and ``python3 judge.py`` (score on the public tests).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gepa.async_stages.pipelines.stages.agent import AgentTask

PROBLEM = "0"
TIME_LIMIT = 2.0
GCC_IMAGE = "gcc:13"
COMPILE = "g++ -std=gnu++17 -O2 -static"

SEED = r"""// Seed: every piece in a row, no rotation. Correct, and a poor packing.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n; if (scanf("%d", &n) != 1) return 0;
    long long W = 0, H = 0;
    vector<array<long long, 2>> place(n);
    for (int i = 0; i < n; ++i) {
        int k; scanf("%d", &k);
        long long mnx = LLONG_MAX, mny = LLONG_MAX, mxx = LLONG_MIN, mxy = LLONG_MIN;
        for (int j = 0; j < k; ++j) { long long x, y; scanf("%lld %lld", &x, &y);
            mnx = min(mnx, x); mny = min(mny, y); mxx = max(mxx, x); mxy = max(mxy, y); }
        place[i] = {W - mnx, -mny};
        W += mxx - mnx + 1; H = max(H, mxy - mny + 1);
    }
    printf("%lld %lld\n", W, H);
    for (int i = 0; i < n; ++i) printf("%lld %lld 0 0\n", place[i][0], place[i][1]);
    return 0;
}
"""

BUILD_SH = f"""#!/usr/bin/env bash
# Compile solution.cpp to ./solution (static, through docker, since this host has no compiler).
set -e
cd "$(dirname "$0")"
docker run --rm -v "$PWD":/w -w /w {GCC_IMAGE} {COMPILE} solution.cpp -o solution
echo "built ./solution"
"""

JUDGE_PY = r'''#!/usr/bin/env python3
"""Score ./solution on the public tests: mean checker ratio over all tests (0 for a failed test)."""
import subprocess, sys, re, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
TL = 2.0
def one(i):
    inp, ans = HERE / "testdata" / f"{i}.in", HERE / "testdata" / f"{i}.ans"
    out = HERE / "out" / f"{i}.out"; out.parent.mkdir(exist_ok=True)
    try:
        with open(inp) as fi, open(out, "w") as fo:
            subprocess.run([str(HERE / "solution")], stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, timeout=TL + 0.5, check=False)
    except subprocess.TimeoutExpired:
        return i, 0.0, "time limit"
    c = subprocess.run([str(HERE / "chk"), str(inp), str(out), str(ans)], capture_output=True, text=True)
    m = re.search(r"Ratio: ([0-9.]+)", c.stderr + c.stdout)
    if m: return i, float(m.group(1)), "ok"
    return i, 0.0, (c.stderr or c.stdout).strip()[:120]
tests = sorted(int(p.stem) for p in (HERE / "testdata").glob("*.in"))
if len(sys.argv) > 1: tests = [int(a) for a in sys.argv[1:]]
with ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, tests))
rows.sort(); total = 0.0
for i, r, msg in rows:
    total += r
    if msg != "ok" or "-v" in os.environ.get("JUDGE_FLAGS", ""): print(f"test {i}: {r:.4f} {msg}")
print(f"MEAN RATIO {total / len(rows):.6f} over {len(rows)} tests")
'''

SPEC = """TASK: FrontierCS problem 0, "Pack the Polyominoes (Reflections Allowed)". Write `solution.cpp` (C++17) that
reads n polyominoes and places all of them, with rotations, reflections and translations, into one axis-aligned
rectangle with no overlaps, minimizing its area. Read `statement.txt` for the exact input and output format. Each
test has a 2 s time limit (n up to 10^4 pieces, up to 10 cells each). The score is the mean over the 70 public
tests of cells / (W*H); an invalid or timed-out output scores 0 on that test. Build with `./build.sh` and score
with `python3 judge.py` (or `python3 judge.py 1 2 3` for a few tests). Only `solution.cpp` is kept."""


def source_dir() -> Path:
    src = os.environ.get("POLYOMINO_SRC")
    if not src:
        raise SystemExit("set POLYOMINO_SRC to the Frontier-CS/algorithmic directory")
    return Path(src).resolve()


def docker_compile(workdir: Path, cpp: str, out: str, extra: str = "") -> tuple[bool, str]:
    cmd = [
        "docker", "run", "--rm", "-v", f"{workdir}:/w", "-w", "/w", GCC_IMAGE, "bash", "-c",
        f"{COMPILE} {extra} {cpp} -o {out}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return p.returncode == 0, (p.stderr or p.stdout)[-2000:]


def prepare_tree(src: Path, tree: Path) -> Path:
    """The task tree proposer sessions get: statement, public tests, seed, checker binary, helpers."""
    problem = src / "problems" / PROBLEM
    if (tree / "chk").exists() and (tree / "solution.cpp").exists():
        return tree
    tree.mkdir(parents=True, exist_ok=True)
    shutil.copy(problem / "statement.txt", tree / "statement.txt")
    if (tree / "testdata").exists():
        shutil.rmtree(tree / "testdata")
    shutil.copytree(problem / "testdata", tree / "testdata")
    (tree / "solution.cpp").write_text(SEED)
    (tree / "build.sh").write_text(BUILD_SH)
    (tree / "build.sh").chmod(0o755)
    (tree / "judge.py").write_text(JUDGE_PY)
    if not (tree / "chk").exists():
        shutil.copy(problem / "chk.cc", tree / "chk.cc")
        shutil.copy(src / "judge" / "include" / "testlib.h", tree / "testlib.h")
        ok, log = docker_compile(tree, "chk.cc", "chk", extra="-I/w")
        if not ok:
            raise SystemExit("checker did not compile:\n" + log)
        (tree / "chk.cc").unlink()
        (tree / "testlib.h").unlink()
    return tree


def run_tests(binary: Path, tree: Path, workers: int = 8) -> list[tuple[int, float, str]]:
    tests = sorted(int(p.stem) for p in (tree / "testdata").glob("*.in"))
    chk = tree / "chk"

    def one(i: int) -> tuple[int, float, str]:
        inp, ans = tree / "testdata" / f"{i}.in", tree / "testdata" / f"{i}.ans"
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fo:
            out = Path(fo.name)
        try:
            with open(inp) as fi, open(out, "w") as fo2:
                subprocess.run([str(binary)], stdin=fi, stdout=fo2, stderr=subprocess.DEVNULL, timeout=TIME_LIMIT + 0.5, check=False)
        except subprocess.TimeoutExpired:
            out.unlink(missing_ok=True)
            return i, 0.0, "time limit"
        c = subprocess.run([str(chk), str(inp), str(out), str(ans)], capture_output=True, text=True)
        out.unlink(missing_ok=True)
        m = re.search(r"Ratio: ([0-9.]+)", c.stderr + c.stdout)
        if m:
            return i, float(m.group(1)), "ok"
        return i, 0.0, (c.stderr or c.stdout).strip()[:120]

    with ThreadPoolExecutor(workers) as ex:
        return sorted(ex.map(one, tests))


def feedback(score: float, info: dict[str, Any]) -> str:
    if not info.get("compiled", True):
        return f"compile error, score 0:\n{info.get('diag', '')}"
    fails = info.get("failed") or []
    line = f"mean ratio {score:.4f} over {info.get('tests')} tests; {len(fails)} tests scored 0"
    if fails:
        line += ": " + "; ".join(f"test {i}: {msg}" for i, msg in fails[:5])
    worst = info.get("worst") or []
    if worst:
        line += "; lowest ratios: " + ", ".join(f"test {i} {r:.3f}" for i, r in worst)
    return line


def make(src: Path | None = None, tree: Path | None = None) -> tuple[AgentTask, str, Any]:
    src = src or source_dir()
    tree = prepare_tree(src, tree or Path(os.environ.get("POLYOMINO_TREE", str(src.parent / "polyomino_tree"))))
    seed = (tree / "solution.cpp").read_text()

    def evaluator(candidate: str, example: Any = None, **kw: Any) -> tuple[float, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="polyomino-eval-", dir=str(tree.parent)) as td:
            tdp = Path(td)
            (tdp / "solution.cpp").write_text(candidate)
            ok, log = docker_compile(tdp, "solution.cpp", "solution")
            if not ok:
                return 0.0, {"compiled": False, "diag": log, "tests": 0}
            rows = run_tests(tdp / "solution", tree)
        total = sum(r for _, r, _ in rows)
        failed = [(i, msg) for i, r, msg in rows if msg != "ok"]
        worst = sorted(((i, r) for i, r, msg in rows if msg == "ok"), key=lambda t: t[1])[:3]
        score = total / len(rows) if rows else 0.0
        return score, {"compiled": True, "tests": len(rows), "failed": failed, "worst": worst, "ratios": [r for _, r, _ in rows]}

    task = AgentTask(
        source=str(tree),
        editable="solution.cpp",
        spec=SPEC,
        ignore=(".git", "out", "solution"),
        feedback=feedback,
        score_name="mean ratio",
        tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash(./build.sh:*)", "Bash(python3:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)"],
    )
    return task, seed, evaluator
