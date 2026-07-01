"""Database migration definitions."""

from .migration_base import BaseMigration
from .v001_investigation_registry import Migration as M001
from .v002_checkpoint_tables import Migration as M002
from .v003_baseline_store import Migration as M003
from .v004_circuit_breaker_state import Migration as M004

__all__ = ["BaseMigration", "M001", "M002", "M003", "M004"]

# Ordered list used by MigrationManager when no explicit module list is given.
ALL_MIGRATIONS = [M001, M002, M003, M004]
