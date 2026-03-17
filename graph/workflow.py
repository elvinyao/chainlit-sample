"""
LangGraph 工作流模块 (graph/workflow.py)
=======================================

【作用】
    定义和编排整个 AI Agent 的工作流程。
    使用 LangGraph 将三个节点串联成一个有向图：

        parse（解析）→ act（执行）→ summarize（汇总）

    每个节点是一个异步函数，接收 GraphState 并返回更新后的 GraphState。

【什么是 LangGraph？】
    LangGraph 是 LangChain 生态系统中的一个库，专门用于构建
    有状态的、多步骤的 LLM 应用工作流。

    核心概念：
    - StateGraph: 状态图，定义了工作流的结构
    - Node（节点）: 工作流中的每一步操作，是一个函数
    - Edge（边）: 节点之间的连接，定义执行顺序
    - State（状态）: 在节点之间传递的共享数据

    为什么用 LangGraph 而不是简单地顺序调用函数？
    - 工作流可视化：可以导出为图形查看
    - 易于扩展：添加条件分支、循环、并行执行等
    - 状态管理：自动处理状态的传递和合并
    - 容错：可以配置重试、回退等策略

【为什么单独放在 graph/ 目录】
    - 工作流编排是"胶水"层，连接 services/ 中的各个服务。
    - 将来如果要修改工作流（如添加新步骤、加条件判断），只改这里。
    - 与 UI（Chainlit）和具体服务实现解耦。

【用法】
    from graph.workflow import build_graph

    app = build_graph()
    result = await app.ainvoke({"input": "用户的自然语言输入"})
    # result 是完整的 GraphState，包含 result_text、screenshot_path 等
"""

# --- LangGraph 核心组件 ---
# END: 一个特殊常量，表示工作流的终点。
#   当最后一个节点通过 add_edge 连接到 END 时，工作流结束。
# StateGraph: 状态图构建器。
#   用法：graph = StateGraph(StateType) → 添加节点和边 → graph.compile()
from langgraph.graph import END, StateGraph

# 导入我们自定义的状态类型和服务
from models.schemas import GraphState
from services.browser_service import playwright_fill_form
from services.llm_service import parse_node


# =========================================================================
# 工作流节点定义
# =========================================================================
#
# 每个节点函数的签名都是：
#   async def node_name(state: GraphState) -> GraphState
#
# 输入：当前的工作流状态
# 输出：更新后的状态（LangGraph 会自动合并到全局状态中）
#


async def act_node(state: GraphState) -> GraphState:
    """工作流节点 #2：执行浏览器操作。

    【作用】
        接收 parse_node 解析出的表单数据，
        调用 Playwright 服务填写并提交本地 HTML 表单。

    【为什么要检查 parsed 是否存在？】
        如果 parse_node 解析失败（LLM 返回了无法解析的内容），
        parsed 可能为空。此时直接返回错误信息，避免 Playwright 报错。
    """

    # 防御性检查：如果 LLM 没有成功解析出数据，提前返回错误信息
    if not state.get("parsed"):
        return {
            "input": state["input"],
            "parsed": state.get("parsed"),
            "result_text": "No parsed data.",
        }

    data = state["parsed"]
    # 调用浏览器服务，执行表单填写和截图
    result = await playwright_fill_form(data)

    return {
        "input": state["input"],
        "parsed": state["parsed"],
        "result_text": result["result_text"],
        "screenshot_path": result["screenshot_path"],
    }


async def summarize_node(state: GraphState) -> GraphState:
    """工作流节点 #3：生成执行摘要。

    【作用】
        将 Playwright 得到的结果整理成用户友好的文本摘要。
        这个文本最终会显示在 Chainlit 的聊天界面中。

    【为什么需要这个节点？】
        - 原始的 result_text 可能不够友好（只是网页上的文本）
        - 这里可以加入更多上下文信息
        - 将来可以用 LLM 生成更自然的摘要
    """

    summary = "Playwright executed the form submission."
    if state.get("result_text"):
        summary += f"\n\nPage result:\n{state['result_text']}"

    return {
        "input": state["input"],
        "parsed": state.get("parsed"),
        "result_text": summary,
        "screenshot_path": state.get("screenshot_path"),
    }


# =========================================================================
# 工作流图构建
# =========================================================================


def build_graph():
    """构建并编译 LangGraph 工作流。

    【工作流结构】

        ┌─────────┐     ┌─────────┐     ┌───────────┐     ┌─────┐
        │  parse   │ ──→ │   act   │ ──→ │ summarize │ ──→ │ END │
        └─────────┘     └─────────┘     └───────────┘     └─────┘
         解析用户输入     Playwright      生成摘要          工作流结束
         提取表单数据     填写并提交表单   整理结果文本

    【构建步骤详解】
        1. StateGraph(GraphState) — 创建一个新的状态图，指定状态类型
        2. add_node("名称", 函数) — 注册一个节点
        3. set_entry_point("名称") — 设置工作流的入口节点（第一个执行的节点）
        4. add_edge("A", "B") — 添加一条边，表示 A 执行完后执行 B
        5. compile() — 编译图，返回一个可执行的应用对象

    【compile() 返回什么？】
        返回一个 CompiledGraph 对象，可以用以下方式调用：
        - app.invoke(state)  — 同步执行整个工作流
        - app.ainvoke(state) — 异步执行整个工作流
        - app.stream(state)  — 流式执行，逐步返回每个节点的输出

    返回:
        CompiledGraph: 编译后的可执行工作流
    """

    # 第1步：创建状态图，「GraphState」定义了节点之间传递的数据结构
    graph = StateGraph(GraphState)

    # 第2步：注册三个节点
    # 注意：parse_node 来自 services/llm_service.py（LLM 解析）
    graph.add_node("parse", parse_node)
    # act_node 在本文件中定义（Playwright 执行）
    graph.add_node("act", act_node)
    # summarize_node 在本文件中定义（结果汇总）
    graph.add_node("summarize", summarize_node)

    # 第3步：设置入口点 —— 工作流从 "parse" 节点开始
    graph.set_entry_point("parse")

    # 第4步：添加边 —— 定义节点的执行顺序
    graph.add_edge("parse", "act")          # parse 完成后执行 act
    graph.add_edge("act", "summarize")      # act 完成后执行 summarize
    graph.add_edge("summarize", END)        # summarize 完成后工作流结束

    # 第5步：编译并返回可执行的工作流
    return graph.compile()
