#!/usr/bin/env python3
"""Single source of truth for which phase this repo is in.

Two phases, because the tooling has to behave differently in each:

  refactor  (default) — the hybrid path. Legacy and new code coexist; legacy is
                        editable; lint is scoped to changed files because the
                        legacy tree carries ~74 pre-existing ruff errors; the
                        legacy pytest suite is the gate.
  rewrite            — clean-slate build of a new package. Only the new package
                        is writable (allowlist, not denylist — a denylist has to
                        enumerate every legacy path and will miss one); lint
                        covers the whole new package since it has no inherited
                        debt; only the new test suite gates.

Mode resolution order:
  1. RCA_MODE env var (one-off override, e.g. RCA_MODE=refactor for a single command)
  2. .claude/mode file contents  ← the durable setting; commit it
  3. default "refactor"

A FILE is used rather than an env var on purpose. An env var's failure mode is
"forgot to export in a new shell, lock silently inert, drift goes unnoticed."
A file's failure mode is "stale setting over-blocks", which is loud and safe.
Prefer the mechanism whose failure is loud.
"""

from __future__ import annotations

import os
from pathlib import Path

REFACTOR = "refactor"
REWRITE = "rewrite"
VALID_MODES = (REFACTOR, REWRITE)

# ── Update these two when the rewrite package is actually named ───────────────
# Paths (repo-relative, prefix-matched) that hold the clean-slate rewrite.
NEW_PACKAGE_DIRS: tuple[str, ...] = ("devops_agent_v2/",)
# Test suite that gates the turn in rewrite mode.
NEW_TEST_DIR: str = "tests_v2/"

# Always writable, in either mode: the tooling itself and repo-level config.
# Entries ending in "/" are directory prefixes; everything else must match exactly.
# Without that distinction a bare "CLAUDE.md" would prefix-match "CLAUDE.md.bak",
# and ".gitignore" would match ".gitignore_old".
ALWAYS_WRITABLE: tuple[str, ...] = (
    ".claude/",
    "CLAUDE.md",
    "pyproject.toml",
    "Makefile",
    ".gitignore",
)

# Read-only in refactor mode: the modules the rewrite replaces outright. The
# tool/data substrate (devops_agent/tools/, devops_agent/core/db/) is absent on
# purpose — it is meant to be adapted and kept.
LEGACY_DENYLIST: tuple[str, ...] = (
    "devops_agent/orchestrator/",
    "devops_agent/core/recovery/",
    "devops_agent/core/orchestrator/",
)

TRUTHY = {"1", "true", "yes", "on"}


def repo_root() -> Path:
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir).resolve()
    return Path(__file__).resolve().parents[2]


def current_mode(root: Path | None = None) -> str:
    """Resolve the active phase. Unknown values fall back to refactor, loudly ignored."""
    env_mode = os.environ.get("RCA_MODE", "").strip().lower()
    if env_mode in VALID_MODES:
        return env_mode

    root = root or repo_root()
    mode_file = root / ".claude" / "mode"
    try:
        value = mode_file.read_text(encoding="utf-8").strip().lower()
    except (OSError, ValueError):
        # ValueError covers UnicodeDecodeError (a non-UTF-8 byte in the file), which
        # is NOT an OSError. Letting it escape would crash the hook with exit 1 —
        # neither 0 (allow) nor 2 (block) — i.e. it would fail OPEN, the exact
        # opposite of this module's fail-loud intent.
        return REFACTOR
    # Tolerate a commented/annotated file — first non-comment token wins.
    for line in value.splitlines():
        token = line.split("#", 1)[0].strip()
        if token in VALID_MODES:
            return token
    return REFACTOR


def new_package_exists(root: Path | None = None) -> bool:
    """True once at least one configured rewrite package directory exists on disk."""
    root = root or repo_root()
    return any((root / d).is_dir() for d in NEW_PACKAGE_DIRS)


def is_writable(rel_posix: str, mode: str) -> tuple[bool, str]:
    """Decide whether a repo-relative path may be written in the given mode.

    Returns (allowed, reason). `reason` is only meaningful when not allowed.
    """
    for p in ALWAYS_WRITABLE:
        # Directory entries (trailing "/") match by prefix; bare filenames must be exact.
        if rel_posix == p or (p.endswith("/") and rel_posix.startswith(p)):
            return True, ""

    if mode == REWRITE:
        # Allowlist: only the new package and its tests. Fails safe — anything
        # unenumerated is reference material, including legacy paths nobody
        # remembered to list.
        allowed_prefixes = (*NEW_PACKAGE_DIRS, NEW_TEST_DIR)
        if any(rel_posix.startswith(p) for p in allowed_prefixes):
            return True, ""
        return False, (
            f"rewrite mode allows writes only under {', '.join(allowed_prefixes)} "
            f"(plus {', '.join(ALWAYS_WRITABLE)})"
        )

    # refactor mode: denylist the modules the rewrite replaces.
    for prefix in LEGACY_DENYLIST:
        if rel_posix.startswith(prefix):
            return False, f"{prefix} is part of the legacy tree the rewrite replaces"
    return True, ""
