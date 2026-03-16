import json
from pathlib import Path
from typing import Any, Dict, TypedDict

import chainlit as cl
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, ValidationError


class FormData(BaseModel):
    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")
    message: str = Field(..., description="User message")


class GraphState(TypedDict, total=False):
    input: str
    parsed: Dict[str, Any]
    result_text: str
    screenshot_path: str


async def parse_node(state: GraphState) -> GraphState:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.2,
    )
    parser = JsonOutputParser(pydantic_object=FormData)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract structured fields for a form. "
                "Return JSON only and follow the schema strictly."
                "\n{format_instructions}",
            ),
            ("human", "User request: {input}"),
        ]
    )
    chain = prompt | llm | parser
    try:
        parsed = chain.invoke({"input": state["input"], "format_instructions": parser.get_format_instructions()})
    except (ValidationError, json.JSONDecodeError):
        # Fallback: get raw text and try to parse JSON manually
        raw = (prompt | llm).invoke({"input": state["input"], "format_instructions": parser.get_format_instructions()})
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = json.loads(content)
    return {"input": state["input"], "parsed": parsed}


async def playwright_fill_form(data: Dict[str, Any]) -> Dict[str, str]:
    root = Path(__file__).parent
    html_path = root / "sample_site" / "index.html"
    url = html_path.as_uri()

    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)
    screenshot_path = output_dir / "last_run.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.fill("input[name='name']", data["name"])
        await page.fill("input[name='email']", data["email"])
        await page.fill("textarea[name='message']", data["message"])
        await page.click("button[type='submit']")
        result_text = await page.locator("#result").inner_text()
        await page.screenshot(path=str(screenshot_path), full_page=True)
        await browser.close()

    return {
        "result_text": result_text,
        "screenshot_path": str(screenshot_path),
    }


async def act_node(state: GraphState) -> GraphState:
    if not state.get("parsed"):
        return {"input": state["input"], "parsed": state.get("parsed"), "result_text": "No parsed data."}
    data = state["parsed"]
    result = await playwright_fill_form(data)
    return {
        "input": state["input"],
        "parsed": state["parsed"],
        "result_text": result["result_text"],
        "screenshot_path": result["screenshot_path"],
    }


async def summarize_node(state: GraphState) -> GraphState:
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
    graph = StateGraph(GraphState)
    graph.add_node("parse", parse_node)
    graph.add_node("act", act_node)
    graph.add_node("summarize", summarize_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "act")
    graph.add_edge("act", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


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
