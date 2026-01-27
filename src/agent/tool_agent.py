"""
工具链 Agent - 学习示例
工具链 Agent = Tool Chain Agent
核心流程：问题 → Agent分析 → 选择工具 → 调用工具 → 处理结果 → 返回答案

与 RAG Agent 的区别：
- 工具链 Agent：调用外部工具（API、数据库等）执行操作
- RAG Agent：从知识库中检索信息，基于文档内容回答

学习要点：
- create_react_agent：创建 React Agent（自动工具调用）
- bind_tools：将工具绑定到 LLM
- InjectedState：从 LangGraph 状态中注入消息
- trim_messages：清理上下文消息
"""
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Annotated
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain.tools import tool
from llm_db_config.chatmodel import llm_no_think
from src.tools.query_tools import query_tool
from src.prompts.agent_prompts import query_prompt

# ========== 步骤1：创建 React Agent ==========
# React Agent 会自动：
# 1. 理解用户问题
# 2. 决定是否需要调用工具
# 3. 调用工具并处理结果
# 4. 生成最终回复
tool_assistant = create_react_agent(
    llm_no_think.bind_tools([query_tool]),  # 绑定工具到LLM
    tools=[query_tool],  # 可用工具列表
    prompt=query_prompt,  # Agent 的 Prompt
)


# ========== 步骤2：定义工具函数 ==========
# 这个工具会被 LangGraph 调用，用于处理工具调用请求
@tool
def tool_agent_tool(state: Annotated[dict, InjectedState], dummy: str = '') -> str:
    """
    工具链 Agent 工具：调用外部工具执行操作
    这是一个工具函数，会被 LangGraph 的 ToolNode 调用。
    它内部使用 React Agent 来处理用户问题并调用工具。

    参数说明：
    - state: LangGraph 自动注入的对话状态，包含 messages
    - dummy: 占位参数，LangGraph 工具调用需要至少一个参数

    返回：
    - str: 工具执行结果文本

    示例：
        用户："查询设备状态"
        → Agent 分析问题
        → 调用 simple_query_tool
        → 返回查询结果
    """
    # 从状态中获取消息历史
    messages = state.get("messages", [])

    # 清理上下文消息（避免 token 超限）
    cleaned_messages = trim_messages(
        messages[-9:],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=16384,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True
    )

    # 调用 React Agent 处理消息
    result = tool_assistant.invoke({"messages": cleaned_messages})

    # 提取最终回复内容
    ret_content = result["messages"][-1].content

    # 清理思考标签（如果有）
    if "" in ret_content:
        ret_content = ret_content.split("")[-1]
    if "" in ret_content:
        ret_content = ret_content.split("")[0]

    return ret_content


# ========== 使用示例 ==========
"""
在 graph_simple.py 中的使用：
from src.agent.tool_agent import tool_agent_tool

def tool_agent_node(state: State, config):
    result = tool_agent_tool.invoke(state, config)
    return {"messages": state.get("messages", []) + [AIMessage(content=result)]}

工作流程：
1. 用户输入："查询设备状态"
2. graph_simple.py 路由到 tool_agent_node
3. tool_agent_node 调用 tool_agent_tool
4. tool_agent_tool 内部：
   - 清理消息上下文
   - 调用 tool_assistant（React Agent）
   - tool_assistant 分析问题，调用 simple_query_tool
   - 返回结果
5. 结果返回给用户
"""

# 代码说明：
# 1. 功能定位：实现基于React Agent的工具链调用，是业务操作执行的核心模块，替代原query_agent.py的功能；
# 2. 核心逻辑：通过create_react_agent构建自动工具调用的Agent，再封装为LangChain工具供LangGraph调用；
# 3. 技术特点：
#    - 集成上下文裁剪逻辑，适配LLM的token输入限制；
#    - 清理Agent回复中的标签，仅返回用户可见结果；
#    - 遵循LangGraph的InjectedState注入规则，适配工作流状态传递；
# 4. 应用场景：处理设备运维的业务操作类请求（如查询设备状态、执行运维指令），是Agent与外部业务系统交互的核心入口。