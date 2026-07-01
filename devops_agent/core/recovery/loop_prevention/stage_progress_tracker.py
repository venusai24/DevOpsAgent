"""Detects structural looping (duplicate calls, zero state change)."""

import json
from typing import Any


class StageProgressTracker:
    def __init__(self):
        self.call_fingerprints = set()
        self.duplicates = 0
        self.total_calls = 0

    def record_tool_call(self, tool_name: str, params: dict[str, Any]) -> None:
        self.total_calls += 1
        # Replay detection
        fingerprint = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
        if fingerprint in self.call_fingerprints:
            self.duplicates += 1
        else:
            self.call_fingerprints.add(fingerprint)

    def is_looping(self) -> bool:
        # Exit if 3 identical replays in one stage (increased from 2 to allow retries)
        return self.duplicates >= 3
