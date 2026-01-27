from langchain_core.messages import HumanMessage, AIMessage
from src.intent_demo.intent_schemas import State, Plan
from src.intent_demo.intent_map import INTENT_KEY_AGENT
from src.utils.model_hook import get_last_user_input

def planner_node(state: State, config):
    # 获取意图标识并去除首尾空格
    key = (state.get("intent_key") or "").strip()
    # 若意图标识为空，返回无法识别意图的结果
    if not key:return {"messages": [AIMessage(content="无法识别用户意图")]}
    # 根据意图标识匹配对应的Agent
    agent_tool = INTENT_KEY_AGENT.get(key)
    # 若匹配不到Agent，返回不支持该意图的结果
    if not agent_tool:return {"messages": [AIMessage(content=f"暂不支持该意图: {key}")]}
    # 从对话消息中提取最新的用户输入
    user_text = get_last_user_input(state.get("messages", []))
    # 构造执行计划：包含工具、参数、执行后总结的配置
    plan: Plan = {
        "type": "plan",
        "steps": [{
            "agent_tool": agent_tool,
            "params": {"userText": user_text},
            "summary_after": True,
        }],
    }
    # 处理执行计划并生成工具调用消息
    tool_calls = [{
        "name": step["agent_tool"],
        "args": step.get("params", {}),
        "id": f"call_{idx}"
    } for idx, step in enumerate(plan.get("steps", []), start=1)]
    tool_str =",".join([i.get("name","") for i in tool_calls])
    return {"plan": plan,"messages": [AIMessage(content=f"调用工具：{tool_str}", tool_calls=tool_calls)]}
# 只返回部分 key 是完全允许的 ——graph（如 LangChain StateGraph）会自动做「状态合并」：用你返回的新 key 覆盖旧状态，未返回的 key 保留原有值
# 代码说明：
# 1. 功能定位：这是LLM Agent框架中的“计划器”模块，负责根据用户意图生成工具执行计划；
# 2. 核心逻辑：
#    - 提取用户意图标识，匹配对应的处理Agent；
#    - 从对话历史中获取最新用户输入；
#    - 构造包含工具、参数、执行后总结的执行计划；
#    - 处理意图识别失败、Agent匹配失败的异常场景；
# 3. 应用场景：在LangChain的多Agent/工具链流程中，作为意图到执行的中间层，实现用户需求到工具调用的自动化映射，是Agent决策流程的关键组件。