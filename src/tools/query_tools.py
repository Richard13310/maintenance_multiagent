"""
查询工具定义 - 学习示例
工具是 Agent 可以调用的函数，用于执行具体任务。
使用 @tool 装饰器定义工具，LangGraph 会自动识别并让 Agent 调用。

学习要点：
- @tool：LangChain 工具装饰器
- 工具描述（docstring）：会被 Agent 用来理解工具用途
- 工具参数：定义工具需要的参数
- 工具返回：返回工具执行结果
"""
from langchain_core.tools import tool
@tool
def query_tool(dummy: str = '') -> str:
    """
    简单的查询工具示例
    这是一个示例工具，实际项目中可以替换为真实的查询工具。
    参数：- dummy: 占位参数，LangGraph 工具调用需要至少一个参数
    返回：- str: 查询结果
    示例：
        用户："查询设备状态"
        → Agent 调用此工具ikkkkkk
        → 返回 "这是一个示例查询结果"
    """
    # 实际项目中，这里应该是真实的查询逻辑
    # 例如：调用 API、查询数据库等
    return "这是一个示例查询结果"

# ========== 如何添加新工具 ==========
"""
示例：添加一个查询场站信息的工具
"""


@tool
def get_station_info(station_name: str) -> str:
    """
    查询场站信息
    参数：- station_name: 场站名称
    返回： - str: 场站信息（JSON 格式）
    """
    # 调用真实 API
    # response = api.get_station(station_name)
    # return json.dumps(response)
    return f"场站 {station_name} 的信息"

# 添加到工具列表
QUERY_TOOLS = [query_tool, get_station_info]

# 在 query_agent.py 中更新：
# query_assistant = create_react_agent(
#     llm_no_think.bind_tools(QUERY_TOOLS),
#     tools=QUERY_TOOLS,
#     prompt=query_prompt,
# )

# 代码说明：
# 1. 功能定位：定义LangChain工具，为Agent提供可调用的业务操作接口；
# 2. 核心逻辑：
#    - 用@tool装饰器将普通函数转为Agent可识别的工具；
#    - 通过docstring为Agent提供工具描述、参数说明与使用示例；
#    - 包含新工具的扩展示例，演示如何新增业务工具并注册；
# 3. 技术特点：
#    - 适配LangGraph的工具调用规则，需至少定义一个参数（占位符也可）；
#    - 支持工具列表批量管理，便于Agent绑定多个工具；
# 4. 应用场景：作为Agent与业务系统的桥梁，实现设备状态查询、场站信息查询等实际业务操作。