"""
浏览器自动化服务模块 (services/browser_service.py)
===================================================

【作用】
    使用 Playwright 库自动化操作浏览器：
    打开本地 HTML 表单 → 填入数据 → 提交 → 抓取结果文本 → 截图保存。
    这模拟了一个"真人在浏览器中填写表单"的过程。

【什么是 Playwright？】
    Playwright 是微软开发的浏览器自动化库（类似 Selenium，但更现代）。
    它可以用代码控制 Chromium、Firefox、WebKit 三种浏览器，
    常用于：
    - 网页自动化测试
    - 网页爬虫
    - 自动化填表、截图等 RPA（机器人流程自动化）任务

    本项目中用它来自动填写一个本地 HTML 表单并截图，
    展示 AI Agent 不仅能"理解"指令，还能"执行"操作。

【为什么使用 async（异步）版本？】
    Playwright 提供同步和异步两种 API：
    - 同步：from playwright.sync_api import sync_playwright
    - 异步：from playwright.async_api import async_playwright
    这里用异步版本是因为 Chainlit 和 LangGraph 都是异步框架，
    使用异步可以在等待浏览器操作时不阻塞其他任务。

【为什么单独放在 services/ 目录】
    - Playwright 逻辑是独立的"浏览器服务"，与 LLM 和 UI 无关。
    - 将来如果要换成其他浏览器自动化工具（如 Selenium），只需改这一个文件。
    - 便于单独测试和复用。

【用法】
    from services.browser_service import playwright_fill_form

    result = await playwright_fill_form({
        "name": "Alice",
        "email": "alice@example.com",
        "message": "Hello!"
    })
    # result => {"result_text": "Submitted!\\nName: Alice\\n...", "screenshot_path": "/path/to/last_run.png"}
"""

from pathlib import Path
from typing import Any, Dict

# async_playwright: Playwright 的异步上下文管理器
# 用法：async with async_playwright() as p: ...
# 进入 with 块时自动启动 Playwright 服务，退出时自动清理资源。
from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# 项目根目录路径
# ---------------------------------------------------------------------------
# __file__ 是当前文件的路径: .../services/browser_service.py
# .resolve() 转为绝对路径
# .parent 取上一级目录: .../services/
# .parent 再取上一级: .../chainlit-sample/ （项目根目录）
#
# 为什么这样做？
#   因为我们需要定位 sample_site/index.html 和 outputs/ 目录，
#   它们都在项目根目录下。使用相对于 __file__ 的路径可以确保
#   无论从哪个工作目录运行脚本，路径都是正确的。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def playwright_fill_form(data: Dict[str, Any]) -> Dict[str, str]:
    """用 Playwright 自动填写本地 HTML 表单并截图。

    【执行步骤】
        1. 定位本地 HTML 文件 (sample_site/index.html)
        2. 启动无头（headless）Chromium 浏览器
        3. 打开表单页面
        4. 依次填入 name、email、message 字段
        5. 点击 Submit 按钮
        6. 等待结果显示，抓取结果文本
        7. 对整个页面截图
        8. 关闭浏览器，返回结果

    参数:
        data: 包含 "name"、"email"、"message" 键的字典

    返回:
        字典，包含：
        - "result_text": 表单提交后页面上显示的结果文本
        - "screenshot_path": 截图文件的绝对路径
    """

    # ----- 定位 HTML 文件 -----
    # sample_site/index.html 是我们的示例表单页面
    html_path = _PROJECT_ROOT / "sample_site" / "index.html"
    # as_uri() 将文件路径转为 file:// 协议的 URL
    # 例如：file:///Users/.../sample_site/index.html
    # 浏览器需要 URL 格式才能打开本地文件
    url = html_path.as_uri()

    # ----- 准备输出目录 -----
    output_dir = _PROJECT_ROOT / "outputs"
    # mkdir(exist_ok=True): 如果目录已存在则不报错，不存在则创建
    output_dir.mkdir(exist_ok=True)
    screenshot_path = output_dir / "last_run.png"

    # ----- 启动 Playwright 并操作浏览器 -----
    # async with 确保即使出错也能正确清理 Playwright 资源
    async with async_playwright() as p:

        # launch(headless=True):
        #   启动 Chromium 浏览器，headless=True 表示无头模式（不显示窗口）。
        #   在服务器环境中必须用 headless 模式，因为没有显示器。
        #   调试时可以改为 headless=False 来观察浏览器的实际操作。
        browser = await p.chromium.launch(headless=True)

        # new_page(): 在浏览器中打开一个新标签页
        page = await browser.new_page()

        # goto(url): 导航到指定 URL（相当于在地址栏输入网址并回车）
        await page.goto(url)

        # ----- 填写表单 -----
        # fill(selector, value): 找到匹配 CSS 选择器的输入框，清空并填入文本
        #   "input[name='name']"  → 找到 <input name="name" ...> 元素
        #   "input[name='email']" → 找到 <input name="email" ...> 元素
        #   "textarea[name='message']" → 找到 <textarea name="message" ...> 元素
        await page.fill("input[name='name']", data["name"])
        await page.fill("input[name='email']", data["email"])
        await page.fill("textarea[name='message']", data["message"])

        # ----- 提交表单 -----
        # click(selector): 点击匹配 CSS 选择器的元素
        #   "button[type='submit']" → 找到 <button type="submit"> 并点击
        await page.click("button[type='submit']")

        # ----- 获取结果 -----
        # locator("#result"): 定位 id="result" 的元素
        # inner_text(): 获取该元素内的纯文本内容（不含 HTML 标签）
        result_text = await page.locator("#result").inner_text()

        # ----- 截图 -----
        # screenshot(): 对页面截图
        #   path: 截图保存路径
        #   full_page=True: 截取整个页面（包括需要滚动才能看到的部分）
        await page.screenshot(path=str(screenshot_path), full_page=True)

        # ----- 关闭浏览器 -----
        # 释放浏览器进程占用的资源
        await browser.close()

    return {
        "result_text": result_text,
        "screenshot_path": str(screenshot_path),
    }
