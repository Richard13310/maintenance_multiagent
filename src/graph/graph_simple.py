"""
LangGraph 图构建 - 简化版
核心流程：意图分类 → 路由 → RAG知识问答/工具链Agent/闲聊
两种 Agent 类型：
1. RAG Agent (src/rag/)：基于向量检索的知识问答
2. 工具链 Agent (src/agent/)：基于工具调用的操作执行
"""
from typing import Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from src.intent_demo.intent_schemas import State
from src.intent_demo.intent_map import INTENT_STR_KEY
from src.intent_demo.intent_cls import intent_cls_factory
from src.intent_demo.planner import planner_node
from llm_db_config.chatmodel import llm_no_think
from llm_db_config.checkpointer import checkpointer
from src.agent.tool_agent import tool_agent_tool  # 工具链Agent
from src.rag.rag_agent import create_simple_rag_node  # RAG Agent
from src.chit_chat.chit_chat import create_chit_chat_node
from src.tools.query_tools import QUERY_TOOLS

def tool_react_agent_node(state: State, config):
    """
    工具链 Agent 节点 - 学习示例
    用于处理需要调用外部工具的操作（如查询API、执行操作等）
    工作流程：
    1. 接收用户问题
    2. 调用 tool_agent_tool 处理问题
    3. tool_agent_tool 内部使用 React Agent 调用工具
    4. 返回执行结果
    示例：
        用户输入："查询设备状态"
        → 路由到此节点
        → 调用 tool_agent_tool
        → Agent 调用 simple_query_tool
        → 返回查询结果
    """
    result = tool_agent_tool.invoke(state, config)
    return {"messages": [AIMessage(content=result)]}

def rag_agent_node(state: State, config):
    """
    RAG Agent 节点 - 学习示例
    用于处理基于知识库检索的知识问答
    工作流程：
    1. 接收用户问题
    2. 向量检索相关文档
    3. 将文档作为上下文，让 LLM 生成答案
    4. 返回答案
    示例：
        用户输入："什么是设备？"
        → 路由到此节点
        → 向量检索相关知识文档
        → 基于文档内容生成答案
        → 返回答案
    """
    # RAG Agent 节点会在 build_graph 中创建
    pass

def tool_Structured_Agent_node(builder):
    # 注册Planner相关节点
    builder.add_node("business", planner_node)
    builder.add_node("tools", ToolNode(tools=QUERY_TOOLS))
    return builder

def should_continue(state: State) -> str:
    """判断节点：根据AI最新消息，决定「继续调用工具（tools）」还是「终止（END）」"""
    ai_message = state.messages[-1]
    # 若AI消息包含工具调用指令→继续执行工具节点；否则→终止循环，返回结果
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return "tools"
    return END

def build_graph(llm, intent_str_key: Dict[str, str] = None):
    """构建LangGraph工作流图"""
    builder = StateGraph(State)
    agent_sign = 1
    # 注册意图分类节点
    intent_cls_node = intent_cls_factory(llm, intent_str_key or INTENT_STR_KEY)
    builder.add_node("intent_cls", intent_cls_node)

    # 两种 agent 设计模式：
    if agent_sign == 1: # plan+tool_calls（先规划后执行），灵活度高，但复杂
        tool_Structured_Agent_node(builder)
    elif agent_sign == 2: # chain.bind_tools（规划执行一体） 简易的ReAct agent
        builder.add_node("business", tool_react_agent_node)
    elif agent_sign == 3: # React_agent流程
        builder.add_node("tools", ToolNode(tools=QUERY_TOOLS))
    # 注册RAG Agent 节点（知识问答）
    builder.add_node("rag_agent", create_simple_rag_node(llm))

    # 注册闲聊节点
    chit_chat_node = create_chit_chat_node(llm)
    builder.add_node("chit_chat", chit_chat_node)

    # 路由函数：意图分类后的二次路由
    def route_after_intent(state: State):
        key = (state.get("intent_key") or "").strip()
        if key == "chit_chat" or key == "":
            return "chit_chat"
        elif key == "question":
            return "question"
        return "business" # 所有业务相关的都是 business 意图key

    # 设置图的入口点
    builder.set_entry_point("intent_cls")
    # 添加意图分类后的条件路由（intent_cls_node → 其他节点）
    builder.add_conditional_edges(
        "intent_cls_node",
        route_after_intent,
        {
            "business": "business",
            "chit_chat": "chit_chat",
            "question": "rag_agent",
        }
    )


    # 添加流程连续边
    if agent_sign == 1: # plan+tool_calls（先规划后执行），灵活度高，但复杂
        builder.add_edge("business", "tools")
        builder.add_edge("tools", END)
    elif agent_sign == 2: # tool_agent流程，chain.bind_tools（规划执行一体） 简易的ReAct agent
        builder.add_edge("business", END)
    elif agent_sign == 3: # React_agent流程
        # React_agent思考节点后，走判断逻辑，决定去工具节点还是终止END
        builder.add_conditional_edges("business", should_continue)
        # 上面没有END，则继续下一轮ReAct循环
        builder.add_edge("tools", "intent_cls")
    builder.add_edge("rag_agent", END)    # RAG Agent 完成
    builder.add_edge("tool_agent", END)  # 工具链 Agent 完成
    builder.add_edge("chit_chat", END)   # 闲聊完成

    # 编译图并返回
    return builder.compile(checkpointer=checkpointer)

graph = build_graph(llm=llm_no_think)

# 代码说明：
# 1. 功能定位：整合RAG Agent、工具链Agent与闲聊系统的核心工作流，实现基于意图的多分支处理；
# 2. 核心升级：
#    - 新增RAG Agent节点，处理知识问答类问题；
#    - 重构工具链Agent节点，基于tool_agent.py实现业务操作；
#    - 优化路由逻辑，通过关键词匹配区分知识问答与操作执行类问题；
# 3. 路由规则：
#    - 含"是什么/为什么"等关键词 → RAG Agent；
#    - 含"查询/执行"等关键词 → 工具链Agent；
#    - 无业务关键词 → 闲聊节点；
# 4. 应用场景：作为设备运维智能体的总调度中心，实现不同类型用户请求的精细化处理，是多Agent协作的核心载体。