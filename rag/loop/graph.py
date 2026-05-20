"""LangGraph wiring for Loop RAG (PDCA closed-loop)."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.core.config import get_settings
from rag.loop.agents import act_node, check_node, make_do_node, plan_node, should_continue
from rag.loop.state import LoopState


def build_graph(pg_store=None, neo_store=None):
    g = StateGraph(LoopState)
    g.add_node("plan", plan_node)
    g.add_node("do", make_do_node(pg_store=pg_store, neo_store=neo_store))
    g.add_node("check", check_node)
    g.add_node("act", act_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "do")
    g.add_edge("do", "check")
    g.add_conditional_edges("check", should_continue, {"continue": "act", "end": END})
    g.add_edge("act", "plan")

    s = get_settings()
    return g.compile().with_config(recursion_limit=s.loop_max_iterations * 4 + 4)
