import re
import logging

logger = logging.getLogger(__name__)

# A simple deterministic list of regexes that block highly destructive operations.
DESTRUCTIVE_PATTERNS = [
    # Filesystem / SQL destructive operations
    re.compile(r"\brm\s+-r", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r">\s*/dev/null", re.IGNORECASE),
    # Infrastructure-level destruction
    re.compile(r"\bterraform\s+destroy\b", re.IGNORECASE),
    # Kubernetes namespace / mass-delete operations
    re.compile(r"\bkubectl\s+delete\s+(namespace|ns)\b", re.IGNORECASE),
    re.compile(r"\bkubectl\b.*\bdelete\b.*--all\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+namespace\b", re.IGNORECASE),
]

def is_safe_command(command: str) -> bool:
    """
    Evaluates if a shell or kubectl command passes basic static analysis guardrails.
    Returns False if any destructive patterns are found.
    """
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            logger.warning(f"Guardrail triggered! Destructive command blocked: {command}")
            return False
    return True
