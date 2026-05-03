"""
LangGraph research network.

Compile with:
    graph = build_research_graph()
    result = await graph.ainvoke(initial_state, config={
        "configurable": {
            "thread_id": "...",
            "session": session,
            "embedder": embedder,
            "llm_provider": llm_provider,
        }
    })
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from aris.agents.bridge_discoverer import bridge_discoverer_node
from aris.agents.concept_extractor import concept_extractor_node
from aris.agents.contradiction_analyst import contradiction_analyst_node
from aris.agents.gap_analyst import gap_analyst_node
from aris.agents.hypothesis_formulator import hypothesis_formulator_node
from aris.agents.orchestrator import orchestrator_node, route_next
from aris.agents.state import ResearchState


def build_research_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("concept_extractor", concept_extractor_node)
    graph.add_node("bridge_discoverer", bridge_discoverer_node)
    graph.add_node("contradiction_analyst", contradiction_analyst_node)
    graph.add_node("hypothesis_formulator", hypothesis_formulator_node)
    graph.add_node("gap_analyst", gap_analyst_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_next,
        {
            "concept_extractor": "concept_extractor",
            "bridge_discoverer": "bridge_discoverer",
            "contradiction_analyst": "contradiction_analyst",
            "hypothesis_formulator": "hypothesis_formulator",
            "gap_analyst": "gap_analyst",
            END: END,
        },
    )

    for node in [
        "concept_extractor",
        "bridge_discoverer",
        "contradiction_analyst",
        "hypothesis_formulator",
        "gap_analyst",
    ]:
        graph.add_edge(node, "orchestrator")

    return graph.compile(checkpointer=checkpointer)
