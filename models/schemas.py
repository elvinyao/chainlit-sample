"""
数据模型定义模块 (models/schemas.py)
====================================

【作用】
    定义整个应用中共享的数据结构，包括：
    1. FormData  — 表单数据的结构化定义（使用 Pydantic）
    2. GraphState — LangGraph 工作流中各节点之间传递的状态（使用 TypedDict）

【为什么单独放在这里】
    - 数据模型是最底层的依赖，不依赖任何业务逻辑。
    - services/ 和 graph/ 都会用到这些定义，放在独立模块里可以避免循环导入。
    - 修改数据结构时只需要改这一个文件。

【用法】
    from models.schemas import FormData, GraphState
"""

from typing import Any, Dict, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic 模型 —— 用于 LLM 输出的结构化校验
# ---------------------------------------------------------------------------
#
# 什么是 Pydantic？
#   Pydantic 是 Python 最流行的数据验证库。你定义一个类，声明每个字段的类型，
#   Pydantic 就会在实例化时自动做类型检查和转换。
#   例如：FormData(name="Alice", email="a@b.com", message="Hi")
#   如果 email 不是字符串，就会抛出 ValidationError。
#
# 为什么在这里使用 Pydantic？
#   我们让 LLM（大语言模型）返回 JSON，然后用 Pydantic 来验证 JSON 是否
#   符合我们期望的字段和类型。这样即使 LLM 返回了格式错误的内容，
#   也能在运行时及时捕获错误。
#
# Field(...) 中的 "..." 表示该字段是必填的（没有默认值），
# description 会被传给 LLM 作为提示，告诉它每个字段应该填什么内容。
#

class FormData(BaseModel):
    """表单数据模型 —— 描述了用户要提交的表单字段。

    LLM 会根据用户的自然语言输入，提取出以下三个字段并以 JSON 格式返回。
    Pydantic 确保返回的数据类型正确。
    """

    name: str = Field(..., description="User name")
    """用户姓名，字符串类型，必填"""

    email: str = Field(..., description="User email")
    """用户邮箱，字符串类型，必填"""

    message: str = Field(..., description="User message")
    """用户消息内容，字符串类型，必填"""


# ---------------------------------------------------------------------------
# TypedDict 状态 —— LangGraph 工作流中各节点共享的数据容器
# ---------------------------------------------------------------------------
#
# 什么是 TypedDict？
#   TypedDict 是 Python 标准库 typing 模块提供的类型，
#   它让你可以给普通字典（dict）定义键的名称和值的类型。
#   本质上还是一个 dict，但 IDE 和类型检查器能帮你检查拼写和类型。
#
# 什么是 LangGraph？
#   LangGraph 是 LangChain 生态中的一个库，用于构建"有向图"工作流。
#   每个节点（node）是一个函数，接收 state 并返回更新后的 state。
#   节点之间通过边（edge）连接，数据通过 state 在节点间传递。
#
# total=False 的含义：
#   表示所有字段都是可选的。因为在工作流的不同阶段，
#   不是所有字段都有值——比如刚开始只有 input，
#   parsed 要到 parse_node 执行完才会有。
#

class GraphState(TypedDict, total=False):
    """LangGraph 工作流状态 —— 在 parse → act → summarize 三个节点之间传递。

    工作流中的数据流向：
        1. parse 节点：接收 input，生成 parsed（解析后的表单数据）
        2. act 节点：  使用 parsed 调用 Playwright，生成 result_text 和 screenshot_path
        3. summarize 节点：汇总 result_text，生成最终的用户可读摘要
    """

    input: str
    """用户在聊天框中输入的原始文本，例如 "请提交表单，姓名Alice，邮箱alice@test.com，消息Hello" """

    parsed: Dict[str, Any]
    """LLM 从 input 中提取出的结构化数据，格式为 {"name": "...", "email": "...", "message": "..."}"""

    result_text: str
    """Playwright 执行表单提交后，从网页上抓取到的结果文本"""

    screenshot_path: str
    """Playwright 截图的文件路径，例如 "outputs/last_run.png" """
