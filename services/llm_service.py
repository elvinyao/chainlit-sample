"""LLM service — uses LangChain + Gemini to parse user input into structured form data."""

import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from models.schemas import FormData, GraphState


async def parse_node(state: GraphState) -> GraphState:
    """LangGraph node: parse free-text input into FormData JSON."""

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
        parsed = chain.invoke(
            {"input": state["input"], "format_instructions": parser.get_format_instructions()}
        )
    except (ValidationError, json.JSONDecodeError):
        raw = (prompt | llm).invoke(
            {"input": state["input"], "format_instructions": parser.get_format_instructions()}
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = json.loads(content)

    return {"input": state["input"], "parsed": parsed}
