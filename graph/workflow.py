"""LangGraph workflow — wires parse → act → summarize nodes into a compiled graph."""

from langgraph.graph import END, StateGraph

from models.schemas import GraphState
from services.browser_service import playwright_fill_form
from services.llm_service import parse_node


async def act_node(state: GraphState) -> GraphState:
    """LangGraph node: invoke Playwright to fill the form with parsed data."""

    if not state.get("parsed"):
        return {
            "input": state["input"],
            "parsed": state.get("parsed"),
            "result_text": "No parsed data.",
        }
    data = state["parsed"]
    result = await playwright_fill_form(data)
    return {
        "input": state["input"],
        "parsed": state["parsed"],
        "result_text": result["result_text"],
        "screenshot_path": result["screenshot_path"],
    }


async def summarize_node(state: GraphState) -> GraphState:
    """LangGraph node: produce a human-readable summary of the execution."""

    summary = "Playwright executed the form submission."
    if state.get("result_text"):
        summary += f"\n\nPage result:\n{state['result_text']}"
    return {
        "input": state["input"],
        "parsed": state.get("parsed"),
        "result_text": summary,
        "screenshot_path": state.get("screenshot_path"),
    }


def build_graph():
    """Construct and compile the LangGraph state graph."""

    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("act", act_node)
    graph.add_node("summarize", summarize_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "act")
    graph.add_edge("act", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
