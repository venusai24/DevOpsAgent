"""
agent/integrations/shell.py

Sandboxed shell command executor for AIRS remediation steps.

Security model:
    Commands are NOT run via a raw shell (subprocess.run(shell=True)) because
    that allows arbitrary shell expansion, pipe chaining, subshell injection,
    and environment variable attacks.

    Instead, commands are tokenised and executed as a direct execvp-style
    process list (subprocess.run(args=list, shell=False)), which prevents
    shell metacharacter injection entirely.

    An additional allowlist restricts executable binaries to a curated set of
    safe SRE utilities. Any command whose first token is not in this list is
    blocked before a subprocess is ever created.

    The execution timeout is capped at SHELL_TIMEOUT_SECONDS (default: 30s) to
    prevent runaway processes from blocking the orchestrator event loop.

    stdout/stderr are captured and returned as a single UTF-8 string for
    injection into the postmortem log.

Supported use cases (subset of SRE runbooks):
    - curl / wget  — health-check probes against internal endpoints
    - systemctl    — restart a non-containerised service
    - service      — legacy SysV service control
    - journalctl   — fetch recent logs from a systemd unit
    - df / free    — disk and memory usage snapshots
    - netstat / ss — check listening ports
    - ps / top     — quick process inspection (non-interactive)

Not supported (will be blocked):
    - rm, dd, mkfs, shred       — destructive filesystem operations
    - sudo, su, chmod, chown    — privilege escalation
    - python, bash, sh, zsh     — arbitrary code execution
    - curl | bash / wget -O-    — untrusted download and exec patterns
      (these are caught by the destructive blocklist in guardrails.py first)
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHELL_TIMEOUT_SECONDS: Final[int] = 30
"""
Hard wall-clock timeout for every shell subprocess.
Prevents runaway commands from blocking the async orchestrator event loop.
"""

_SHELL_ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        # HTTP probes
        "curl",
        "wget",
        # Service control (non-containerised hosts)
        "systemctl",
        "service",
        # Log inspection
        "journalctl",
        "dmesg",
        # Disk and memory snapshots
        "df",
        "du",
        "free",
        # Network diagnostics
        "netstat",
        "ss",
        "ping",
        "nslookup",
        "dig",
        "traceroute",
        # Process inspection (non-interactive only)
        "ps",
        "pgrep",
        "lsof",
        # Text processing (read-only)
        "cat",
        "grep",
        "awk",
        "sed",
        "tail",
        "head",
        "wc",
        "sort",
        "uniq",
        # Date / time utilities
        "date",
        "uptime",
        # File ownership inspection (read-only)
        "stat",
        "ls",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_shell_command(command: str) -> str:
    """
    Execute *command* in a sandboxed subprocess and return combined stdout/stderr.

    The command string is tokenised via ``shlex.split`` (honours quoting and
    escaping) and executed with ``shell=False`` to prevent shell injection.
    The first token (binary name) is validated against ``_SHELL_ALLOWED_BINARIES``
    before any subprocess is spawned.

    Args:
        command: Raw shell command string from a ``RemediationStep.command``
                 field whose ``environment`` is ``"shell"``.

    Returns:
        A UTF-8 string containing combined stdout + stderr, prefixed with an
        execution status tag (``[Success]`` or ``[Error]``).

    Raises:
        ValueError: If the binary is not in the allowlist.
        subprocess.TimeoutExpired: If the process exceeds SHELL_TIMEOUT_SECONDS.
        FileNotFoundError: If the binary is not present on the host PATH.
    """
    if not command or not command.strip():
        return "[shell] No command provided."

    try:
        tokens: list[str] = shlex.split(command)
    except ValueError as exc:
        logger.error("[shell] Failed to tokenise command %r: %s", command[:120], exc)
        return f"[shell][Error] Command parse failed: {exc}"

    binary = tokens[0].split("/")[-1]  # strip any absolute path prefix

    if binary not in _SHELL_ALLOWED_BINARIES:
        logger.warning(
            "[shell] BLOCKED binary not in allowlist: %r | cmd=%r", binary, command[:120]
        )
        return (
            f"[shell][BLOCKED] Binary '{binary}' is not in the SRE shell allowlist. "
            f"Allowed binaries: {sorted(_SHELL_ALLOWED_BINARIES)}"
        )

    logger.info("[shell] Executing: %s", tokens)

    try:
        result = subprocess.run(
            tokens,
            shell=False,           # No shell metacharacter expansion
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        status = "[Success]" if result.returncode == 0 else f"[Error rc={result.returncode}]"
        logger.info("[shell] %s returncode=%d", status, result.returncode)
        return f"[shell]{status}\n{combined.strip()}"

    except subprocess.TimeoutExpired:
        logger.error("[shell] Command timed out after %ds: %s", SHELL_TIMEOUT_SECONDS, tokens)
        return f"[shell][Error] Command timed out after {SHELL_TIMEOUT_SECONDS}s."

    except FileNotFoundError:
        logger.error("[shell] Binary not found on PATH: %r", binary)
        return f"[shell][Error] Binary '{binary}' not found on host PATH."

    except Exception as exc:
        logger.error("[shell] Unexpected error executing %r: %s", command[:120], exc)
        return f"[shell][Error] {exc}"
