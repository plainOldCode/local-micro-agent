"""Domain mixins composed into MicroAgent.

Each module owns one controller concern extracted from the former
orchestrator god-class. The mixins hold no state of their own: MicroAgent
defines __init__ and the shared attributes (config/state/models/mcp), and
every mixin method operates through self.
"""
from .candidates import CandidateRecordsMixin
from .context import PromptContextMixin
from .model_runtime import ModelRuntimeMixin
from .search_memory import AdaptiveSearchMixin
from .tactics import BrainstormTacticsMixin
from .telemetry import TelemetryMixin
from .todos import TodoLifecycleMixin

__all__ = [
    "AdaptiveSearchMixin",
    "BrainstormTacticsMixin",
    "CandidateRecordsMixin",
    "ModelRuntimeMixin",
    "PromptContextMixin",
    "TelemetryMixin",
    "TodoLifecycleMixin",
]
