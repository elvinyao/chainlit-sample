"""
LLM 服务模块 (services/llm_service.py)
======================================

【作用】
    封装与大语言模型（LLM）的交互逻辑。
    本模块使用 LangChain 框架调用 Google Gemini 模型，
    将用户的自然语言输入解析为结构化的表单数据（JSON）。

【核心概念】
    - LangChain：一个用于构建 LLM 应用的框架，提供了 Prompt 模板、
      输出解析器、模型封装等工具，让你可以像搭积木一样组合它们。
    - ChatPromptTemplate：提示词模板，定义发送给 LLM 的消息格式。
    - JsonOutputParser：输出解析器，将 LLM 返回的文本解析为 JSON 对象。
    - ChatGoogleGenerativeAI：LangChain 对 Google Gemini 模型的封装。

【为什么单独放在 services/ 目录】
    - LLM 调用是一个独立的"服务"，与 UI（Chainlit）和浏览器操作（Playwright）无关。
    - 将来如果要换 LLM（比如从 Gemini 换成 OpenAI），只需要改这一个文件。
    - 便于单独测试 LLM 的输入输出。

【用法】
    from services.llm_service import parse_node

    # parse_node 是一个 LangGraph 节点函数，接收 GraphState 返回 GraphState
    result = await parse_node({"input": "姓名Alice，邮箱alice@test.com，消息Hello"})
    # result["parsed"] => {"name": "Alice", "email": "alice@test.com", "message": "Hello"}
"""

import json

# --- LangChain 核心组件 ---
# JsonOutputParser: 将 LLM 的文本输出解析为 Python 字典
#   它会根据 Pydantic 模型自动生成格式说明（format_instructions），
#   告诉 LLM 应该返回什么样的 JSON 结构。
from langchain_core.output_parsers import JsonOutputParser

# ChatPromptTemplate: 聊天提示词模板
#   使用 ("role", "content") 元组列表来定义多轮对话消息。
#   - "system"：系统指令，告诉 LLM 它的角色和任务
#   - "human"：用户消息
#   模板中的 {变量名} 会在调用时被替换为实际值。
from langchain_core.prompts import ChatPromptTemplate

# ChatGoogleGenerativeAI: Google Gemini 模型的 LangChain 封装
#   - model: 指定使用哪个 Gemini 模型版本
#   - temperature: 控制输出的随机性，0=确定性最高，1=最随机
#     这里设为 0.2 因为我们需要稳定的 JSON 输出，不需要太多创造性。
#   - API Key 会自动从环境变量 GOOGLE_API_KEY 或 GEMINI_API_KEY 读取。
from langchain_google_genai import ChatGoogleGenerativeAI

# ValidationError: Pydantic 验证失败时抛出的异常
from pydantic import ValidationError

# 导入我们自定义的数据模型
from models.schemas import FormData, GraphState


async def parse_node(state: GraphState) -> GraphState:
    """LangGraph 节点：将用户自然语言输入解析为结构化表单数据。

    【工作流程】
        1. 创建 Gemini LLM 实例
        2. 创建 JSON 输出解析器（基于 FormData 模型）
        3. 构建提示词模板，包含系统指令和格式说明
        4. 用 "|" 运算符将 prompt → llm → parser 串联成一个处理链（chain）
        5. 调用链，传入用户输入，得到解析后的 JSON
        6. 如果解析失败，使用降级策略手动解析

    【关于 LangChain 的 "|" 运算符（LCEL）】
        chain = prompt | llm | parser
        这叫 LangChain Expression Language (LCEL)。
        "|" 表示数据管道：prompt 的输出 → 传给 llm → llm 的输出 → 传给 parser
        相当于：parser(llm(prompt(input)))
        这种写法让数据处理流程非常清晰。

    参数:
        state: 包含 "input" 键的 GraphState 字典

    返回:
        更新后的 GraphState，增加了 "parsed" 键
    """

    # ----- 第1步：创建 LLM 实例 -----
    # model: 选用 Gemini 3.1 Flash Lite，速度快、成本低，适合简单的 JSON 提取任务
    # temperature=0.2: 低温度 → 输出更确定，适合结构化数据提取
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.2,
    )

    # ----- 第2步：创建输出解析器 -----
    # pydantic_object=FormData 告诉解析器期望的 JSON 结构
    # 解析器会自动生成格式说明，例如：
    #   "返回一个 JSON 对象，包含字段 name(string), email(string), message(string)"
    parser = JsonOutputParser(pydantic_object=FormData)

    # ----- 第3步：构建提示词模板 -----
    # system 消息：告诉 LLM 它的任务是提取表单字段，必须返回纯 JSON
    # {format_instructions} 会被替换为解析器自动生成的格式说明
    # {input} 会被替换为用户的实际输入
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

    # ----- 第4步：用 LCEL 管道串联成处理链 -----
    # 数据流向：提示词模板 → LLM 生成回答 → 解析器提取 JSON
    chain = prompt | llm | parser

    # ----- 第5步：调用链并处理错误 -----
    try:
        # invoke() 执行整个链，传入模板变量
        parsed = chain.invoke(
            {"input": state["input"], "format_instructions": parser.get_format_instructions()}
        )
    except (ValidationError, json.JSONDecodeError):
        # ----- 降级策略 -----
        # 如果解析器无法解析 LLM 的输出（比如 LLM 在 JSON 外面加了多余文字），
        # 就跳过解析器，直接拿 LLM 的原始输出，手动用 json.loads 解析。
        raw = (prompt | llm).invoke(
            {"input": state["input"], "format_instructions": parser.get_format_instructions()}
        )
        # LLM 返回的是一个 Message 对象，.content 才是实际文本
        content = raw.content if hasattr(raw, "content") else str(raw)
        parsed = json.loads(content)

    # 返回更新后的状态，包含原始输入和解析结果
    return {"input": state["input"], "parsed": parsed}
