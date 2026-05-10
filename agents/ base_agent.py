"""
agents/base_agent.py

Abstract base class that every agent in the system must implement.
All agents (retriever, memory, web, tool, reasoning, critic, router, planner)
inherit from BaseAgent and implement the run() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"      # got something, but incomplete
    SKIPPED  = "skipped"     # agent decided it wasn't the right one to handle this


@dataclass
class AgentInput:
    """
    Unified input envelope passed to every agent's run() method.

    Attributes:
        query:      The natural-language query or instruction.
        context:    Optional dict of extra data the caller wants to pass in
                    (e.g. retrieved chunks, prior conversation turns, user id).
        session_id: Used by memory_agent to scope reads/writes per session.
        metadata:   Arbitrary key-value pairs for logging / tracing.
    """
    query: str
    context: dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOutput:
    """
    Unified output envelope returned by every agent's run() method.

    Attributes:
        status:     One of AgentStatus values.
        result:     The main payload — could be a string answer, a list of
                    chunks, a task graph, a score, etc.
        agent_name: Automatically set by BaseAgent.run() — don't set manually.
        error:      Populated only when status == FAILURE.
        metadata:   Arbitrary key-value pairs (tokens used, latency, sources…).
        latency_ms: Wall-clock time of the run() call, set automatically.
    """
    status: AgentStatus
    result: Any
    agent_name: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


# Base class

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclass contract
    -----------------
    1. Set self.name  — short identifier, e.g. "retriever"
    2. Set self.description — one sentence used by router/planner to decide
                              whether to delegate to this agent
    3. Implement _run(input: AgentInput) -> AgentOutput

    Do NOT override run() — it adds timing, logging, and error wrapping
    around your _run() implementation automatically.

    Example subclass
    ----------------
    class RetrieverAgent(BaseAgent):
        name = "retriever"
        description = "Searches the vector store for relevant document chunks."

        def _run(self, input: AgentInput) -> AgentOutput:
            chunks = self.retrieval_service.search(input.query)
            return AgentOutput(
                status=AgentStatus.SUCCESS,
                result=chunks,
            )
    """

    # Subclasses MUST set these two class-level attributes
    name: str = ""
    description: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce that every concrete subclass declares name and description
        if not getattr(cls, "__abstractmethods__", None):
            if not cls.name:
                raise TypeError(f"{cls.__name__} must define a non-empty class attribute: name")
            if not cls.description:
                raise TypeError(f"{cls.__name__} must define a non-empty class attribute: description")

    # Public entry point — do not override

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Wraps _run() with timing, structured logging, and error handling.
        Always call this from outside the agent; never call _run() directly.
        """
        start = time.perf_counter()
        logger.info("[%s] starting | session=%s | query=%r",
                    self.name, input.session_id, input.query[:120])
        try:
            output = self._run(input)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception("[%s] unhandled exception after %.1f ms", self.name, elapsed)
            output = AgentOutput(
                status=AgentStatus.FAILURE,
                result=None,
                error=str(exc),
                latency_ms=elapsed,
            )
        else:
            output.latency_ms = (time.perf_counter() - start) * 1000

        output.agent_name = self.name
        logger.info("[%s] done | status=%s | latency=%.1f ms",
                    self.name, output.status, output.latency_ms)
        return output

    # Required implementation hook

    @abstractmethod
    def _run(self, input: AgentInput) -> AgentOutput:
        """
        Core agent logic. Implement this in every subclass.

        - Access input.query for the user's request
        - Access input.context for data passed by the caller
        - Access input.session_id for memory scoping
        - Return an AgentOutput with status and result populated
        - Raise exceptions freely — run() will catch and wrap them
        """

    def can_handle(self, input: AgentInput) -> bool:
        """
        Optional pre-flight check used by router_agent to decide if this
        agent is suited for the given input.

        Default: always returns True (agent is always willing to try).
        Override to add intent-matching logic, capability guards, etc.

        Example:
            def can_handle(self, input):
                return "search" in input.query.lower()
        """
        return True

    def success(self, result: Any, **metadata) -> AgentOutput:
        """Shorthand for building a SUCCESS output."""
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            result=result,
            metadata=metadata,
        )

    def failure(self, error: str, **metadata) -> AgentOutput:
        """Shorthand for building a FAILURE output."""
        return AgentOutput(
            status=AgentStatus.FAILURE,
            result=None,
            error=error,
            metadata=metadata,
        )

    def partial(self, result: Any, error: str = "", **metadata) -> AgentOutput:
        """Shorthand for a PARTIAL output — got something but not everything."""
        return AgentOutput(
            status=AgentStatus.PARTIAL,
            result=result,
            error=error,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"