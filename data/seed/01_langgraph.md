# LangGraph

LangGraph is a library by LangChain Inc. for building stateful, multi-actor
applications with LLMs. It models workflows as graphs where nodes are functions
and edges define control flow.

Core concepts:
- StateGraph: a typed graph of nodes that read/write a shared state object.
- Nodes: pure Python functions that take state and return a partial state update.
- Edges: static (always go to node X) or conditional (a function decides).
- Compile: turn the graph into a runnable application.

LangGraph supports cycles, which makes it suitable for agent loops, multi-step
reasoning, and PDCA-style closed-loop systems.
