"""Agentic RAG: ReAct-style loop with vector_search + kg_lookup tools."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag.agentic.state import AgenticState
from rag.agentic.tools import TOOLS
from rag.core.llm import get_chat_model

SYSTEM_PROMPT = (
    "You are a research assistant. Use the provided tools (vector_search, kg_lookup) "
    "to gather context before answering. When you have enough information, write the "
    "final answer with citations like [source#chunk]."
)


def _agent_node(state: AgenticState):
    llm = get_chat_model(temperature=0.0).bind_tools(TOOLS)
    msgs = state.get("messages") or [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    resp = llm.invoke(msgs)
    update = {"messages": [resp], "history": [{"node": "agent", "tool_calls": getattr(resp, "tool_calls", [])}]}
    if not getattr(resp, "tool_calls", None):
        update["answer"] = getattr(resp, "content", "")
    return update


def build_graph():
    g = StateGraph(AgenticState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
