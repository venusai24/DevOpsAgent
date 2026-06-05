"""
agent/security/guardrails.py

Static analysis guardrails for AIRS remediation commands.

Defense-in-depth strategy (layered, applied in order):

  Layer 1 — Destructive blocklist:
    A regex blocklist of unconditionally forbidden patterns (rm -r, DROP TABLE,
    etc.) that should never appear in any AIRS-generated command, regardless of
    KB grounding.  This is the original guardrail, retained and extended.

  Layer 2 — kubectl RBAC verb allowlist (inversion of blocklist):
    For any command that contains 'kubectl', the verb is extracted and compared
    against a whitelist of permitted kubectl verbs. Any kubectl command using a
    verb NOT in the allowlist is blocked, even if it passed Layer 1.
    Allowlist inversion is safer than a blocklist because it is closed by default:
    unknown or future verbs are blocked until explicitly permitted.

  Layer 3 — Namespace protection:
    Commands that target protected system namespaces (kube-system, kube-public,
    kube-node-lease) are blocked unless they match a safe read-only verb
    (get, describe, logs, top).

  Layer 4 — KB command grounding check (optional):
    When a KB entry is available, is_command_grounded() checks whether the
    generated command appears in the KB entry's remediation steps.  Commands
    that deviate from the KB ground truth are flagged.  Callers in execute_node
    may choose to block or escalate based on this result.

  Layer 5 — Shell binary allowlist (shell environment only):
    For steps whose environment is 'shell', the first token (binary name) is
    checked against a curated allowlist of safe SRE utilities.  Binaries not
    in the list are blocked before any subprocess is spawned.
    Delegated to: is_safe_shell_command()

  Layer 6 — psql DDL/DCL blocklist (psql environment only):
    For steps whose environment is 'psql', the SQL statement type is validated
    against an allowlist of operational DML keywords.  DDL (DROP, CREATE,
    ALTER TABLE) and DCL (GRANT, REVOKE) are always blocked.
    Delegated to: is_safe_psql_command()

Usage:
    from agent.security.guardrails import (
        is_safe_command,
        is_safe_shell_command,
        is_safe_psql_command,
        is_command_grounded,
    )

    # Generic entry point (routes by environment):
    if not is_safe_command(command, environment=step.environment):
        # block

    # Or call environment-specific helpers directly:
    if not is_safe_shell_command(command):
        # block shell
    if not is_safe_psql_command(command):
        # block psql
    if not is_command_grounded(command, kb_entry):
        # flag / escalate
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.state import KBEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1 — Destructive blocklist
# ---------------------------------------------------------------------------

_DESTRUCTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-r", re.IGNORECASE),
    re.compile(r"\bkill\s+-9\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+namespace\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r">\s*/dev/null", re.IGNORECASE),
    re.compile(r"\bformat\b.*\bdisk\b", re.IGNORECASE),
    re.compile(r"\bwipefs\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Layer 2 — kubectl RBAC verb allowlist
# ---------------------------------------------------------------------------

_KUBECTL_ALLOWED_VERBS: frozenset[str] = frozenset(
    {
        # Read-only
        "get",
        "describe",
        "logs",
        "top",
        "explain",
        "api-resources",
        "api-versions",
        "version",
        "cluster-info",
        "port-forward",
        # Safe mutations (restartable / reversible)
        "rollout",       # rollout restart / undo / status / history
        "scale",
        "patch",
        "annotate",
        "label",
        "set",           # set resources / set env / set image
        "apply",         # kubectl apply -f
        # Conditionally safe (checked further in Layer 3)
        "exec",
        "cordon",
        "uncordon",
        "drain",
        # Delete is allowed only for specific resource types (checked below)
        "delete",
    }
)

# When 'delete' is used, only these resource types are permitted
_KUBECTL_DELETE_ALLOWED_RESOURCES: frozenset[str] = frozenset(
    {
        "pod",
        "pods",
        "job",
        "jobs",
        "pvc",
        "persistentvolumeclaim",
        "persistentvolumeclaims",
        "configmap",
        "configmaps",
        "cm",
        "hpa",
        "horizontalpodautoscaler",
        "horizontalpodautoscalers",
    }
)

# ---------------------------------------------------------------------------
# Layer 3 — Protected namespaces
# ---------------------------------------------------------------------------

_PROTECTED_NAMESPACES: frozenset[str] = frozenset(
    {"kube-system", "kube-public", "kube-node-lease"}
)

# Verbs allowed even in protected namespaces (read-only + safe scaling)
_NAMESPACE_SAFE_VERBS: frozenset[str] = frozenset(
    {"get", "describe", "logs", "top", "rollout", "scale"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_safe_command(command: str, environment: str = "kubectl", kb_entry=None) -> bool:
    """
    Evaluate whether *command* passes all guardrail layers.

    Routes to the environment-specific guardrail after the universal Layer 1
    destructive blocklist is applied:

      environment='kubectl' → Layers 1, 2, 3 (kubectl RBAC + namespace)
      environment='shell'   → Layers 1, 5 (binary allowlist)
      environment='psql'    → Layers 1, 6 (SQL statement type allowlist)

    Layer 4 (KB grounding) is a separate function: ``is_command_grounded``.

    Args:
        command:     The raw shell / kubectl / psql command string.
        environment: The execution target from ``RemediationStep.environment``.
                     One of 'kubectl', 'shell', 'psql'. Defaults to 'kubectl'
                     for backwards compatibility.
        kb_entry:    Optional KBEntry; reserved for future Layer 4 inline check.

    Returns:
        True if the command passes all layers, False if any layer blocks it.
    """
    if not command or not command.strip():
        return True  # Empty command is trivially safe

    # Layer 1: Destructive pattern blocklist — applies to ALL environments
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            logger.warning(
                "[guardrail:L1] BLOCKED destructive pattern match: %s | cmd=%r",
                pattern.pattern,
                command[:120],
            )
            return False

    # Route to environment-specific layers
    if environment == "shell":
        return is_safe_shell_command(command)

    if environment == "psql":
        return is_safe_psql_command(command)

    # Default: kubectl layers 2–3
    if "kubectl" in command:
        verb = _extract_kubectl_verb(command)

        # Layer 2: RBAC verb allowlist
        if verb and verb not in _KUBECTL_ALLOWED_VERBS:
            logger.warning(
                "[guardrail:L2] BLOCKED forbidden kubectl verb %r | cmd=%r",
                verb,
                command[:120],
            )
            return False

        # Layer 2b: delete resource-type restriction
        if verb == "delete":
            resource = _extract_kubectl_resource_type(command)
            if resource and resource not in _KUBECTL_DELETE_ALLOWED_RESOURCES:
                logger.warning(
                    "[guardrail:L2b] BLOCKED kubectl delete on disallowed resource %r | cmd=%r",
                    resource,
                    command[:120],
                )
                return False

        # Layer 3: Protected namespace check
        namespace = _extract_kubectl_namespace(command)
        if namespace and namespace in _PROTECTED_NAMESPACES:
            if verb and verb not in _NAMESPACE_SAFE_VERBS:
                logger.warning(
                    "[guardrail:L3] BLOCKED mutating command in protected namespace %r | cmd=%r",
                    namespace,
                    command[:120],
                )
                return False

    return True


def is_safe_shell_command(command: str) -> bool:
    """
    Layer 5 — Shell binary allowlist.

    Validates that the first token of *command* (the binary name) is in the
    curated allowlist defined in ``agent/integrations/shell.py``.  This check
    is performed here so that guardrail decisions are centralised and testable
    without importing the subprocess-executing integration module.

    Args:
        command: Raw shell command string.

    Returns:
        True if the binary is in the allowlist, False otherwise.
    """
    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError:
        logger.warning("[guardrail:L5] Failed to tokenise shell command: %r", command[:120])
        return False

    if not tokens:
        return True

    binary = tokens[0].split("/")[-1]  # strip absolute path prefix

    # Import allowlist from the integration module to keep it DRY
    from agent.integrations.shell import _SHELL_ALLOWED_BINARIES
    if binary not in _SHELL_ALLOWED_BINARIES:
        logger.warning(
            "[guardrail:L5] BLOCKED shell binary not in allowlist: %r | cmd=%r",
            binary,
            command[:120],
        )
        return False

    return True


def is_safe_psql_command(command: str) -> bool:
    """
    Layer 6 — psql SQL statement type allowlist.

    Extracts the SQL from a raw psql command string (stripping any
    ``kubectl exec ... psql -c`` wrapper) and verifies that the leading
    SQL keyword is in the operational DML allowlist.  DDL and DCL keywords
    are always blocked.

    Args:
        command: Raw psql or ``psql -c '...'`` command string.

    Returns:
        True if the statement type is permitted, False otherwise.
    """
    # Import helpers from the integration module to keep SQL parsing DRY
    from agent.integrations.psql import _extract_sql, _PSQL_ALLOWED_STATEMENT_TYPES, _PSQL_BLOCKED_KEYWORDS

    sql = _extract_sql(command)

    for pattern in _PSQL_BLOCKED_KEYWORDS:
        if pattern.search(sql):
            logger.warning(
                "[guardrail:L6] BLOCKED psql DDL/DCL keyword %r | sql=%r",
                pattern.pattern,
                sql[:120],
            )
            return False

    first_keyword = sql.strip().split()[0].lower() if sql.strip() else ""
    if first_keyword not in _PSQL_ALLOWED_STATEMENT_TYPES:
        logger.warning(
            "[guardrail:L6] BLOCKED disallowed psql statement type %r | sql=%r",
            first_keyword,
            sql[:120],
        )
        return False

    return True


def is_command_grounded(command: str, kb_entry) -> bool:
    """
    Layer 4 — KB grounding check.

    Checks whether *command* appears in the KB entry's remediation steps after
    normalisation (volatile tokens like pod names, IPs, and hex values are
    stripped before comparison).

    Returns True if the command is grounded (matches a KB step), or if no KB
    commands are available for comparison (giving the LLM the benefit of the
    doubt when the KB entry has no commands).

    Args:
        command:   The generated command to validate.
        kb_entry:  The KBEntry retrieved from the KB store.

    Returns:
        True if grounded or no KB commands available, False if the command
        deviates from all KB steps.
    """
    kb_commands = [
        s.command for s in kb_entry.remediation_steps if s.command
    ]
    if not kb_commands:
        return True  # No KB commands to compare against

    normalised_cmd = _normalise_cmd(command)
    for kb_cmd in kb_commands:
        normalised_kb = _normalise_cmd(kb_cmd)
        # Pass 1: normalised comparison handles both {service} placeholder substitution
        # and runtime token removal (pod names, IPs, integers)
        if normalised_kb in normalised_cmd or normalised_cmd in normalised_kb:
            return True

    logger.warning(
        "[guardrail:L4] Command NOT grounded in KB entry %s | cmd=%r",
        kb_entry.entry_id,
        command[:120],
    )
    return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_kubectl_verb(command: str) -> str | None:
    """
    Extract the first positional argument after 'kubectl' (the verb).

    Handles common patterns:
      kubectl get pod ...
      kubectl rollout restart ...
      kubectl -n prod get ...  (flags before verb)
    """
    # Strip kubectl and any flags (tokens starting with -)
    parts = command.split()
    try:
        kubectl_idx = next(i for i, p in enumerate(parts) if "kubectl" in p)
    except StopIteration:
        return None

    for token in parts[kubectl_idx + 1 :]:
        if not token.startswith("-"):
            return token.lower()
    return None


def _extract_kubectl_resource_type(command: str) -> str | None:
    """
    Extract the resource type from a 'kubectl delete <resource>' command.
    Returns the first non-flag token after 'delete'.
    """
    parts = command.split()
    for i, token in enumerate(parts):
        if token.lower() == "delete" and i + 1 < len(parts):
            for candidate in parts[i + 1 :]:
                if not candidate.startswith("-"):
                    # Strip any trailing slash or qualifier
                    return candidate.split("/")[0].lower()
    return None


def _extract_kubectl_namespace(command: str) -> str | None:
    """
    Extract the target namespace from a kubectl command.
    Supports both:
      -n <namespace>
      --namespace <namespace>
      --namespace=<namespace>
    """
    parts = command.split()
    for i, token in enumerate(parts):
        if token in ("-n", "--namespace") and i + 1 < len(parts):
            return parts[i + 1].lower()
        if token.startswith("--namespace="):
            return token.split("=", 1)[1].lower()
    return None


def _normalise_cmd(cmd: str) -> str:
    """
    Normalise a command for KB grounding comparison.
    Removes volatile runtime tokens: pod name suffixes, IPs, hex values,
    integers, and the {service} template placeholder (replaced by the runtime
    service name before comparison).
    """
    # Replace {service} placeholder with a generic token so the template
    # and the runtime-substituted command compare as equal.
    normalised = re.sub(r"\{service\}", "<service>", cmd)
    # Also replace the actual service name pattern (word-char sequences between
    # deployment/ and the next space or end) with the same token, since the
    # generated command will have the real service name while the KB command
    # still has {service} which we've already replaced above.
    normalised = re.sub(r"(?<=deployment/)[\w-]+", "<service>", normalised)
    # Remove Kubernetes pod name suffixes: -7d9f8b-xkzp2
    normalised = re.sub(r"-[a-z0-9]{5,10}-[a-z0-9]{5}\b", "", normalised)
    # Remove IP addresses
    normalised = re.sub(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<ip>", normalised)
    # Remove hex values
    normalised = re.sub(r"0x[0-9a-fA-F]+", "<hex>", normalised)
    # Remove standalone integers (ports, counts, memory sizes)
    normalised = re.sub(r"\b\d{3,}\b", "<n>", normalised)
    return re.sub(r"\s+", " ", normalised).strip().lower()
