"""Chainlit chat application — thin entry point that delegates to the LangGraph workflow."""

import chainlit as cl

from graph.workflow import build_graph
from models.schemas import GraphState


@cl.on_message
async def on_message(message: cl.Message):
    try:
        app = build_graph()
        final_state: GraphState = await app.ainvoke({"input": message.content})
        elements = []
        if final_state.get("screenshot_path"):
            elements.append(cl.Image(path=final_state["screenshot_path"], name="result"))
        await cl.Message(
            content=final_state.get("result_text") or "No result.",
            elements=elements,
        ).send()
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()


if __name__ == "__main__":
    # Run with: chainlit run app.py -w
    import os

    os.system("chainlit run app.py -w")
