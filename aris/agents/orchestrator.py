"""
Supervisor node. Reads current ResearchState and decides which agent runs next.

Routing logic (in order):
  1. No concepts yet              → concept_extractor
  2. Concepts but no bridges      → bridge_discoverer
  3. Concepts but no contradictions checked → contradiction_analyst
  4. Bridges but no hypotheses    → hypothesis_formulator
  5. No gaps identified yet       → gap_analyst
  6. All steps complete or max iterations reached → END
"""
from __future__ import annotations

from datetime import datetime, timezone
from langgraph.graph import END

from aris.agents.state import ResearchState, StreamEvent

MAX_ITERATIONS = 10


def orchestrator_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0) + 1
    completed = set(state.get("completed_steps", []))

    if iteration > MAX_ITERATIONS:
        return _emit(state, END, iteration, "Max iterations reached — stopping.")

    if "concept_extractor" not in completed:
        return _emit(state, "concept_extractor", iteration, "Extracting concepts from corpus.")

    if "bridge_discoverer" not in completed:
        return _emit(state, "bridge_discoverer", iteration, "Discovering cross-domain bridges.")

    if "contradiction_analyst" not in completed:
        return _emit(state, "contradiction_analyst", iteration, "Analysing contradictions.")

    if "hypothesis_formulator" not in completed:
        return _emit(state, "hypothesis_formulator", iteration, "Formulating hypotheses.")

    if "gap_analyst" not in completed:
        return _emit(state, "gap_analyst", iteration, "Identifying research gaps.")

    return _emit(state, END, iteration, "Research analysis complete.")


def route_next(state: ResearchState) -> str:
    return state.get("_next_node", END)


def _emit(state: ResearchState, next_node: str, iteration: int, message: str) -> dict:
    event: StreamEvent = {
        "agent_node": "orchestrator",
        "event_type": "step_complete" if next_node == END else "routing",
        "content": {"next": next_node, "message": message, "iteration": iteration},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "iteration": iteration,
        "_next_node": next_node,
        "stream_events": [event],
    }
