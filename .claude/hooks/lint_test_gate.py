#!/usr/bin/env python3
"""Stop hook: don't let a turn end with failing lint or a red suite.

The scoping differs by phase, because the right check differs:

refactor mode (default)
  - ruff on CHANGED files only. The legacy tree has ~74 pre-existing ruff errors;
    gating the whole package would block every turn forever. New and rewritten
    code still has to be clean.
  - the legacy pytest suite (`tests/`, ~40s, currently green) gates the turn.

rewrite mode
  - ruff on the WHOLE new package. It has no inherited debt, so changed-files-only
    would let rot accumulate in files you stopped touching.
  - only the NEW test suite gates. Running the legacy suite here is either vacuous
    or actively wrong — it tests code being deliberately replaced.
  - degrades to a no-op while the new package/tests don't exist yet.

Escape hatch, because a gate that can't be bypassed gets deleted:
    export SKIP_QUALITY_GATE=1     # skip entirely (e.g. mid-refactor, knowingly red)

Exit 0 = let the turn end, exit 2 = block (stderr is fed back to Claude).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mode import (  # noqa: E402
    NEW_PACKAGE_DIRS,
    NEW_TEST_DIR,
    REWRITE,
    TRUTHY,
    current_mode,
    repo_root,
)

# These must sum to comfortably LESS than the Stop hook's `timeout` in
# .claude/settings.json (480s). If the outer harness timeout fires first it kills this
# process mid-run, so the graceful "a hung check must not block the turn" path in run()
# never executes and the outcome is left to the harness instead of to us.
RUFF_TIMEOUT_S = 60
PYTEST_TIMEOUT_S = 300


def changed_python_files(root: Path) -> list[str]:
    """Modified, added and untracked .py files that still exist on disk."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:]
        # Renames/copies are reported as "old -> new"; only the new path matters.
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        entry = entry.strip().strip('"')
        if entry.endswith(".py") and (root / entry).is_file():
            paths.append(entry)
    return sorted(set(paths))


def run(cmd: list[str], root: Path, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hung check must not deadlock the session; warn, don't block.
        return 0, f"(timed out after {timeout}s — skipped)"
    except OSError as exc:
        return 0, f"(could not run {cmd[0]}: {exc})"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def tail(text: str, limit: int = 40) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(["...", *lines[-limit:]])


def plan_refactor(root: Path) -> tuple[list[str], str | None]:
    """(ruff targets, pytest target) for the hybrid phase."""
    changed = changed_python_files(root)
    if not changed:
        return [], None
    # Hook scripts and agent config don't affect the package under test.
    touches_package = any(not p.startswith(".claude/") for p in changed)
    return changed, ("tests/" if touches_package else None)


def plan_rewrite(root: Path) -> tuple[list[str], str | None]:
    """(ruff targets, pytest target) for the clean-slate phase."""
    ruff_targets = [d for d in NEW_PACKAGE_DIRS if (root / d).is_dir()]
    pytest_target = NEW_TEST_DIR if (root / NEW_TEST_DIR).is_dir() else None
    return ruff_targets, pytest_target


def main() -> int:
    if os.environ.get("SKIP_QUALITY_GATE", "").strip().lower() in TRUTHY:
        return 0

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    # Set when this Stop already fired once and Claude was told to continue.
    # Without this guard the hook can trap the session in a loop.
    if payload.get("stop_hook_active"):
        return 0

    root = repo_root()
    mode = current_mode(root)

    ruff_targets, pytest_target = (
        plan_rewrite(root) if mode == REWRITE else plan_refactor(root)
    )
    if not ruff_targets and not pytest_target:
        return 0

    failures: list[str] = []

    if ruff_targets and shutil.which("ruff"):
        code, out = run(["ruff", "check", *ruff_targets], root, RUFF_TIMEOUT_S)
        if code != 0 and out:
            scope = "changed files" if mode != REWRITE else "new package"
            failures.append(f"ruff check failed on {scope}:\n{tail(out)}")

    if pytest_target:
        code, out = run(
            [sys.executable, "-m", "pytest", pytest_target, "-q", "-x",
             "--no-header", "-p", "no:cacheprovider"],
            root, PYTEST_TIMEOUT_S,
        )
        if code != 0 and out:
            failures.append(f"pytest failed ({pytest_target}):\n{tail(out)}")

    if not failures:
        return 0

    sys.stderr.write(
        f"Quality gate failed ({mode} mode) — fix these before ending the turn:\n\n"
        + "\n\n".join(failures)
        + "\n\nIf the breakage is intentional (mid-refactor), tell the user rather than "
          "silently ignoring it; SKIP_QUALITY_GATE=1 exists for that case.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
