"""
agent/integrations/psql.py

PostgreSQL command executor for AIRS remediation steps.

Security model:
    Commands are NOT executed by shelling out to the ``psql`` CLI binary.
    Doing so would require credentials in environment variables or .pgpass
    files, expose them in ``/proc/<pid>/cmdline``, and allow shell injection
    if the command string is ever interpolated.

    Instead, this module uses the ``psycopg2`` Python SDK to open a
    parameterised connection and execute the SQL string directly. The
    connection parameters are read exclusively from the ``DATABASE_URL``
    environment variable (already used by the AIRS checkpointer).

    SQL Restrictions (applied before execution):
      - Only DML-level operational statements are permitted: SELECT, UPDATE
        (connection pool limits, GUC settings), and procedural calls.
      - DDL (CREATE, DROP, ALTER TABLE) and DCL (GRANT, REVOKE) are blocked.
      - The destructive pattern blocklist in guardrails.py intercepts
        DROP TABLE / TRUNCATE before this module is ever called.

    All queries run inside an explicit transaction that is automatically
    rolled back on error, preventing partial mutations.

Supported use cases (subset of SRE runbooks):
    - SELECT pg_terminate_backend(pid) ...   — kill leaked connections
    - SELECT pg_cancel_backend(pid)  ...     — cancel long-running queries
    - SELECT * FROM pg_stat_activity         — inspect active sessions
    - UPDATE pg_settings SET ...             — tune GUC parameters at runtime
    - CALL / SELECT <stored_procedure>()     — invoke a pre-approved SP

Simulated mode:
    If DATABASE_URL is not set or psycopg2 is not installed, the module
    returns a clearly labelled simulation string instead of raising, matching
    the behaviour of k8s.py and shell.py in demo/test environments.
"""

from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Psycopg2 optional import (matches k8s.py defensive import pattern)
# ---------------------------------------------------------------------------

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PSQL_TIMEOUT_SECONDS: Final[int] = 30
"""
Statement-level timeout injected into every connection via
``options='-c statement_timeout=<ms>'``. Prevents runaway queries from
blocking the orchestrator event loop.
"""

# SQL statements whose first keyword must be in this set to proceed.
_PSQL_ALLOWED_STATEMENT_TYPES: frozenset[str] = frozenset(
    {
        "select",
        "update",   # GUC / settings tuning
        "call",     # Stored procedures
        "do",       # PL/pgSQL anonymous blocks (DB-side, not host-side)
        "explain",  # Query plan inspection (read-only)
        "show",     # Show a single GUC setting
        "set",      # Session-level GUC override
    }
)

# DDL/DCL keywords whose presence in the statement signals a blocked operation.
_PSQL_BLOCKED_KEYWORDS: list[re.Pattern] = [
    re.compile(r"\bdrop\b", re.IGNORECASE),
    re.compile(r"\bcreate\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r"\balter\s+table\b", re.IGNORECASE),
    re.compile(r"\bgrant\b", re.IGNORECASE),
    re.compile(r"\brevoke\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_psql_command(command: str) -> str:
    """
    Execute a PostgreSQL SQL statement using the psycopg2 SDK and return
    a human-readable result string for injection into the postmortem log.

    The ``command`` argument is the raw SQL string from a ``RemediationStep``
    whose ``environment`` is ``"psql"``.  It may be prefixed with
    ``psql -c "..."`` (kubectl exec style) — the function strips that wrapper
    and extracts the inner SQL before execution.

    Args:
        command: Raw SQL or ``psql -c "..."`` command string.

    Returns:
        UTF-8 result string tagged with ``[psql][Success]`` or ``[psql][Error]``.
    """
    if not command or not command.strip():
        return "[psql] No command provided."

    sql = _extract_sql(command)

    # ------------------------------------------------------------------
    # Static analysis: blocked keywords
    # ------------------------------------------------------------------
    for pattern in _PSQL_BLOCKED_KEYWORDS:
        if pattern.search(sql):
            logger.warning(
                "[psql] BLOCKED DDL/DCL keyword %r | sql=%r",
                pattern.pattern,
                sql[:120],
            )
            return (
                f"[psql][BLOCKED] Statement contains a forbidden DDL/DCL keyword "
                f"({pattern.pattern!r}). Only operational DML statements are permitted."
            )

    # ------------------------------------------------------------------
    # Statement type allowlist
    # ------------------------------------------------------------------
    first_keyword = sql.strip().split()[0].lower() if sql.strip() else ""
    if first_keyword not in _PSQL_ALLOWED_STATEMENT_TYPES:
        logger.warning(
            "[psql] BLOCKED disallowed statement type %r | sql=%r",
            first_keyword,
            sql[:120],
        )
        return (
            f"[psql][BLOCKED] Statement type '{first_keyword}' is not in the "
            f"AIRS allowed list: {sorted(_PSQL_ALLOWED_STATEMENT_TYPES)}"
        )

    # ------------------------------------------------------------------
    # Simulated mode: no psycopg2 or no DATABASE_URL
    # ------------------------------------------------------------------
    if psycopg2 is None:
        logger.info("[psql] psycopg2 not installed — returning simulated result.")
        return f"[psql][Simulated] psycopg2 not installed. Would execute:\n{sql}"

    from config import settings  # lazy import to avoid circular dependency

    if not settings.DATABASE_URL:
        logger.info("[psql] DATABASE_URL not set — returning simulated result.")
        return f"[psql][Simulated] DATABASE_URL not configured. Would execute:\n{sql}"

    # ------------------------------------------------------------------
    # Real execution via psycopg2
    # ------------------------------------------------------------------
    timeout_ms = PSQL_TIMEOUT_SECONDS * 1000
    logger.info("[psql] Executing SQL (timeout=%dms): %s", timeout_ms, sql[:200])

    try:
        conn = psycopg2.connect(
            settings.DATABASE_URL,
            options=f"-c statement_timeout={timeout_ms}",
            connect_timeout=10,
        )
        conn.autocommit = False

        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)

                # Fetch results for SELECT / EXPLAIN / SHOW
                if cur.description:
                    rows = cur.fetchall()
                    if not rows:
                        return "[psql][Success] Query returned 0 rows."
                    # Format as a simple text table
                    headers = [desc.name for desc in cur.description]
                    lines = [" | ".join(headers)]
                    lines.append("-" * len(lines[0]))
                    for row in rows[:50]:  # cap at 50 rows to prevent context overflow
                        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
                    truncated = "" if len(rows) <= 50 else f"\n... ({len(rows) - 50} more rows truncated)"
                    return f"[psql][Success]\n" + "\n".join(lines) + truncated
                else:
                    rowcount = cur.rowcount
                    return f"[psql][Success] Statement affected {rowcount} row(s)."

    except psycopg2.OperationalError as exc:
        logger.error("[psql] Connection error: %s", exc)
        return f"[psql][Error] Database connection failed: {exc}"

    except psycopg2.Error as exc:
        logger.error("[psql] Query error: %s", exc)
        return f"[psql][Error] Query failed: {exc}"

    except Exception as exc:
        logger.error("[psql] Unexpected error: %s", exc)
        return f"[psql][Error] {exc}"

    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_sql(command: str) -> str:
    """
    Extract the inner SQL string from a command that may be wrapped in a
    ``kubectl exec ... -- psql -c "..."`` envelope.

    Handles patterns such as:
      - Pure SQL:  ``SELECT pg_terminate_backend(pid) FROM pg_stat_activity``
      - psql -c:   ``psql -c "SELECT ..."``
      - kubectl exec psql:
          ``kubectl exec -n prod deployment/svc -- psql -U postgres -c "SELECT ..."``

    If no ``-c`` flag is found, the entire command is treated as raw SQL.
    """
    # Match -c 'sql' or -c "sql"
    match = re.search(r"""-c\s+(?P<q>['"])(?P<sql>.+?)(?P=q)""", command, re.DOTALL)
    if match:
        return match.group("sql").strip()

    # Match -c sql  (unquoted, rest of string)
    match = re.search(r"""-c\s+(.+)$""", command, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No -c flag: treat entire string as SQL
    return command.strip()
