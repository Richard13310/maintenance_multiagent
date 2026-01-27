from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Optional, List

def get_last_user_input(messages: List[BaseMessage]) -> Optional[str]:
    """从消息列表中提取最后一条用户输入"""
    for msg in reversed(BaseMessage(messages)):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content or ""
            elif isinstance(content, list):
                # 如果是列表，提取第一个字符串元素
                for item in content:
                    if isinstance(item, str):
                        return item

def trim_msg(state):
    """清理上下文消息，保留最近的对话"""
    trimmed_messages = trim_messages(
        state["messages"][-15:],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=16384,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    state['messages'] = trimmed_messages
    return state

# 代码说明：
# 1. 核心作用：该文件是Agent的**上下文消息处理工具**，通过LangChain的`trim_messages`实现对话消息的智能裁剪，避免上下文过长导致的LLM输入超限问题；
# 2. 函数解析：
#    - trim_msg：接收Agent的运行时状态（State），对消息列表做两层裁剪：先截取最后15条消息，再基于token数量（最大16384）做精细化裁剪；
#    - 裁剪规则：以“人类消息”为起始和结束节点，保留系统消息，使用`count_tokens_approximately`做近似token计数，平衡裁剪精度与性能；
# 3. 技术特点：
#    - 双层裁剪策略：先按消息条数粗裁，再按token数精裁，兼顾效率与LLM的上下文窗口限制；
#    - 基于LangChain的原生工具：复用`trim_messages`的成熟逻辑，减少自定义代码的维护成本；
# 4. 应用场景：在闲聊节点、意图分类节点等需要处理历史对话的模块中调用，用于压缩上下文长度，适配大语言模型的token输入限制，提升Agent的响应稳定性。