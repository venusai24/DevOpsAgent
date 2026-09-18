#!/usr/bin/env python3
"""PreToolUse hook: enforce the write boundary for the current phase.

refactor mode (default)
    Denylist. The modules the rewrite replaces (`devops_agent/orchestrator/**`,
    `core/recovery/**`, `core/orchestrator/**`) become read-only so the old design
    can be referenced but not incrementally drifted back into. The tool/data
    substrate stays editable — it is meant to be adapted and kept.
    INERT unless RCA_REWRITE_LOCK=1, so it blocks nothing before work begins.

rewrite mode
    Allowlist. Only the new package and its test suite are writable; everything
    else is reference material. Active by default — the mode file is the switch,
    no env var needed. An allowlist is used because a denylist has to enumerate
    every legacy path and will eventually miss one; this fails safe instead.

Exit 0 = allow, exit 2 = block (stderr is fed back to Claude).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mode import (  # noqa: E402
    REWRITE,
    TRUTHY,
    current_mode,
    is_writable,
    new_package_exists,
    repo_root,
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Never block on a malformed payload.

    raw_path = (payload.get("tool_input") or {}).get("file_path")
    if not raw_path:
        return 0

    root = repo_root()
    mode = current_mode(root)

    if mode == REWRITE:
        # Guard against locking down the repo before there is anywhere to write.
        # Without this, flipping the mode file early would block every edit.
        if not new_package_exists(root):
            return 0
    elif os.environ.get("RCA_REWRITE_LOCK", "").strip().lower() not in TRUTHY:
        return 0  # refactor mode, lock not explicitly enabled

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or str(root)
    try:
        rel = Path(raw_path).resolve().relative_to(Path(project_dir).resolve())
    except (ValueError, OSError):
        return 0  # Outside the project, or unresolvable — not ours to police.

    rel_posix = rel.as_posix()
    allowed, reason = is_writable(rel_posix, mode)
    if allowed:
        return 0

    sys.stderr.write(
        f"BLOCKED ({mode} mode): {rel_posix} is not writable — {reason}.\n\n"
        "Reads are never blocked, so use it freely as reference. Options:\n"
        "  - Write the equivalent logic in the new package instead.\n"
        "  - If this edit is genuinely required, ask the user to change .claude/mode "
        "or unset RCA_REWRITE_LOCK, rather than working around this hook.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
