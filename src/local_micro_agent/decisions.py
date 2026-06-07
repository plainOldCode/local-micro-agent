from __future__ import annotations

from .state import CodeChange


class ReadDecision:
    def __init__(self, files: list[str], reason: str = ""):
        self.files = files
        self.reason = reason


class CodeCandidate:
    def __init__(
        self,
        candidate_id: str,
        changes: list[CodeChange],
        reason: str = "",
        strategy_axis: str = "",
    ):
        self.candidate_id = candidate_id
        self.changes = changes
        self.reason = reason
        self.strategy_axis = strategy_axis


class CodeDecision:
    def __init__(
        self,
        changes: list[CodeChange],
        candidates: list[CodeCandidate] | None = None,
    ):
        self.changes = changes
        self.candidates = candidates or [CodeCandidate("1", changes, "single candidate")]


class TestDecision:
    def __init__(self, status: str, reason: str = "", next_focus: str = ""):
        self.status = status
        self.reason = reason
        self.next_focus = next_focus
