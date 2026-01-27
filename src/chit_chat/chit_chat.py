"""
闲聊功能模块 - 简化版
核心功能：意图识别 + LLM调用
"""
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from llm_db_config.chatmodel import llm_no_think
from src.prompts.agent_prompts import chit_chat_prompt
from src.utils import trim_msg, get_last_user_input

def create_chit_chat_node(llm):
    """
    创建闲聊节点（简化版）
    Args:
        llm: 语言模型实例
    Returns:
        节点函数 (state, config) -> partial_state
    """
    def chit_chat_node(state, config):
        """闲聊节点：处理非业务对话"""
        # 1. 提取用户输入
        user_input = get_last_user_input(state.get("messages", []))
        if not user_input:
            return {"messages": [AIMessage(content="抱歉，我无法理解您的问题。")]}
        # 2. 清理上下文（保留最近10条消息）
        trimmed_state = trim_msg(state)
        cleaned_messages = trimmed_state['messages'][-10:]
        # 4. 构建提示词并调用LLM
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", chit_chat_prompt),
            ("placeholder", "{messages}")# 可以注入 多轮对话历史（比如包含多个人类消息、助手消息的列表），无需手动拼接每一轮的角色
        ])
        chain = prompt_template | llm
        try:
            result = chain.invoke({"messages": cleaned_messages}, config=config)
            result_content = result.content if hasattr(result, 'content') else str(result)
            # 字数限制：超过100字截断
            if len(result_content) > 100:
                result_content = result_content[:100] + "..."
            return {"messages": [AIMessage(content=result_content)]}
        except Exception as e:
            return {"messages": [AIMessage(content="抱歉，我无法回答这个问题。")]}

    return chit_chat_node

def test_chat_node():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", chit_chat_prompt),
        ("user", "{query}")  # 可以注入 多轮对话历史（比如包含多个人类消息、助手消息的列表），无需手动拼接每一轮的角色
    ])
    llm = llm_no_think
    chain = prompt_template | llm
    try:
        user_input = "你好"
        msg = HumanMessage(content=user_input)
        config={"configurable":{"checkpoint_id":"wenjie"}}
        result = chain.invoke({"query": user_input}, config=config)
        result_content = result.content if hasattr(result, 'content') else str()
        print(result_content)
    except Exception as e:
        print(str(e))

def wj():
    from openai import OpenAI
    # 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
    api_key = "b1192130-bd42-4f58-9fcf-224bd7c0ee8b"
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )
    response = client.responses.create(
        model="doubao-seed-1-8-251228",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"
                    },
                    {
                        "type": "input_text",
                        "text": "你看见了什么？"
                    },
                ],
            }
        ]
    )
    print(response)

if __name__ == "__main__":
    # wj()
    test_chat_node()
# 代码说明：
# 1. 功能定位：该文件是LLM Agent的**闲聊模块**，专门处理用户的非业务类对话，与业务工具调用模块形成互补；
# 2. 核心函数解析：
#    - extract_user_input：从对话消息列表中反向提取最新的用户输入，兼容字符串/列表类型的消息内容；
#    - is_business_intent：通过关键词匹配检测用户输入是否包含业务意图（设备/运维相关），避免闲聊模块处理业务问题；
#    - create_chit_chat_node：工厂函数，生成可嵌入LangGraph的闲聊节点，核心逻辑包含用户输入提取、业务意图检测、上下文清理、LLM调用与结果处理；
#    - chit_chat_node：实际的闲聊节点函数，实现“输入校验→意图检测→上下文裁剪→LLM响应→异常处理”的完整流程；
# 3. 技术特点：
#    - 上下文裁剪：仅保留最近10条消息，减少LLM输入token消耗，提升响应效率；
#    - 结果截断：对LLM回复做100字长度限制，适配端侧展示场景；
#    - 异常兜底：通过try-except捕获LLM调用异常，返回友好提示；
# 4. 应用场景：作为LangGraph工作流的分支节点，承接用户的闲聊类请求（如日常对话、非业务咨询），提升Agent的交互体验，是智能客服类场景的重要组成部分。