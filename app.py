"""
Chainlit 聊天应用入口 (app.py)
==============================

【作用】
    这是整个应用的入口文件，负责：
    1. 启动 Chainlit 聊天界面
    2. 接收用户的聊天消息
    3. 调用 LangGraph 工作流处理消息
    4. 将结果（文本 + 截图）返回给用户

【什么是 Chainlit？】
    Chainlit 是一个开源的 Python 框架，用于快速构建 LLM 聊天应用的 Web UI。
    类似于 Streamlit，但专为聊天场景设计。

    核心特点：
    - 提供开箱即用的聊天界面（类似 ChatGPT 的对话框）
    - 支持消息中嵌入图片、文件等富媒体元素
    - 使用装饰器（@cl.on_message）来定义消息处理逻辑
    - 自动处理 WebSocket 通信、会话管理等底层细节

    启动方式：chainlit run app.py -w
    其中 -w 表示 watch 模式（文件修改后自动重启）

【为什么 app.py 这么简短？】
    这是有意为之的设计。app.py 只负责"胶水"工作：
    - 接收消息 → 调用工作流 → 返回结果
    所有业务逻辑都在 models/、services/、graph/ 中。
    这样做的好处：
    - 更换 UI 框架时（如换成 Gradio），只需改这一个文件
    - 业务逻辑的测试不受 UI 框架影响

【用法】
    # 开发模式（文件修改自动重启）：
    uv run chainlit run app.py -w

    # 或者直接运行：
    uv run python app.py
"""

import chainlit as cl

from graph.workflow import build_graph
from models.schemas import GraphState


# ---------------------------------------------------------------------------
# Chainlit 消息处理器
# ---------------------------------------------------------------------------
#
# @cl.on_message 装饰器：
#   注册一个函数来处理用户发送的每一条聊天消息。
#   每当用户在聊天框中输入文本并发送时，Chainlit 会自动调用这个函数。
#
# cl.Message 对象：
#   - message.content: 用户输入的文本内容（字符串）
#   - message.author: 消息发送者（默认为 "User"）
#
# cl.Image 对象：
#   用于在消息中嵌入图片。
#   - path: 图片文件的路径
#   - name: 图片的显示名称
#


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户发送的聊天消息。

    【完整流程】
        用户输入 → 调用 LangGraph 工作流 → 返回文本结果 + 截图

    【工作流内部步骤】（详见 graph/workflow.py）
        1. parse:     LLM 将自然语言解析为表单数据 (name, email, message)
        2. act:       Playwright 用解析出的数据填写并提交表单
        3. summarize: 生成执行摘要文本

    【错误处理】
        如果工作流中任何步骤失败（如 LLM 调用失败、Playwright 启动失败），
        会捕获异常并将错误信息显示在聊天界面中。
    """

    try:
        # 构建 LangGraph 工作流（每次消息都重新构建，确保无状态残留）
        app = build_graph()

        # ainvoke: 异步执行整个工作流
        # 输入：{"input": "用户的消息文本"}
        # 输出：完整的 GraphState，包含所有节点的处理结果
        final_state: GraphState = await app.ainvoke({"input": message.content})

        # 构建消息中的富媒体元素（这里是截图）
        elements = []
        if final_state.get("screenshot_path"):
            # cl.Image: 将截图嵌入到聊天消息中
            elements.append(cl.Image(path=final_state["screenshot_path"], name="result"))

        # 发送回复消息给用户
        # content: 文本内容（summarize_node 生成的摘要）
        # elements: 附带的图片等富媒体元素
        await cl.Message(
            content=final_state.get("result_text") or "No result.",
            elements=elements,
        ).send()

    except Exception as e:
        # 任何未捕获的异常都会以错误消息的形式显示给用户
        await cl.Message(content=f"Error: {e}").send()


# ---------------------------------------------------------------------------
# 直接运行入口
# ---------------------------------------------------------------------------
# 当直接执行 python app.py 时（而不是通过 chainlit run），
# 用 os.system 启动 chainlit 命令行。
# 通常推荐直接使用 chainlit run app.py -w 来启动。
if __name__ == "__main__":
    import os

    os.system("chainlit run app.py -w")
