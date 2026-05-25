"""
agent/kb/seed.py

CLI script that loads data/kb_seed.yaml and inserts all entries into the
KB SQLite store.

Usage:
    python -m agent.kb.seed                     # seed from default YAML
    python -m agent.kb.seed --file data/my.yaml # seed from a custom YAML file
    python -m agent.kb.seed --count             # print the current entry count and exit

The seeding operation is idempotent: entries with existing entry_ids are
skipped (INSERT OR IGNORE), so running the script multiple times is safe.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml  # PyYAML>=6.0.0

# Ensure project root is on sys.path when run as a module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.kb.store import kb_count, kb_insert
from agent.state import KBEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("kb.seed")

_DEFAULT_SEED_FILE = _PROJECT_ROOT / "data" / "kb_seed.yaml"


async def _seed(yaml_path: Path) -> None:
    """Load *yaml_path* and insert all entries into the KB store."""
    if not yaml_path.exists():
        logger.error("Seed file not found: %s", yaml_path)
        sys.exit(1)

    logger.info("Loading seed file: %s", yaml_path)
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    raw_entries: list[dict] = data.get("entries", [])
    if not raw_entries:
        logger.warning("No entries found in seed file.")
        return

    logger.info("Seeding %d entries into the KB store...", len(raw_entries))
    inserted = 0
    skipped = 0
    errors = 0

    for raw in raw_entries:
        try:
            entry = KBEntry.model_validate(raw)
            before = kb_count()
            await kb_insert(entry)
            after = kb_count()
            if after > before:
                inserted += 1
                logger.info("  [INSERTED] %s — %s", entry.entry_id, entry.incident_taxonomy)
            else:
                skipped += 1
                logger.debug("  [SKIPPED]  %s — already exists", entry.entry_id)
        except Exception as exc:
            errors += 1
            logger.error("  [ERROR]    Failed to process entry: %s — %s", raw.get("entry_id", "?"), exc)

    logger.info(
        "Seeding complete: %d inserted, %d skipped, %d errors. "
        "Total KB entries: %d",
        inserted, skipped, errors, kb_count(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the AIRS KB store from a YAML file.")
    parser.add_argument(
        "--file",
        type=Path,
        default=_DEFAULT_SEED_FILE,
        help=f"Path to the YAML seed file (default: {_DEFAULT_SEED_FILE})",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the current number of KB entries and exit.",
    )
    args = parser.parse_args()

    if args.count:
        print(f"KB store currently contains {kb_count()} entries.")
        return

    asyncio.run(_seed(args.file))


if __name__ == "__main__":
    main()
