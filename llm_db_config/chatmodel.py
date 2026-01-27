from langchain_openai import ChatOpenAI
from core.config import get_settings

# 获取配置实例
settings = get_settings()

# 1. 流式模型（启用thinking，默认）
# 用途：各种Agent（configure_agent, log_agent, device_agent等）
# 特点：streaming=True，支持流式输出，显示思考过程
llm_stream = ChatOpenAI(
    base_url=settings.llm.api_base,
    api_key=settings.llm.api_key,
    model=settings.llm.model,
    temperature=0.1,
    streaming=True
)

# 2. 无思考模式模型（禁用thinking）
# 用途：快速响应的问答/闲聊场景，减少不必要的思考过程
llm_no_think = ChatOpenAI(
    base_url=settings.llm.api_base,
    api_key=settings.llm.api_key,
    model=settings.llm.model,
    temperature=0.1,
    streaming=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

# 代码说明：
# 1. 功能定位：创建LangChain封装的OpenAI兼容大模型实例，为Agent提供不同模式的LLM能力；
# 2. 核心配置：
#    - llm_stream：启用流式输出与思考模式，适配需要展示推理过程的Agent；
#    - llm_no_think：禁用思考模式，提升响应速度，适配简单问答/闲聊场景；
# 3. 技术特点：
#    - 从配置中心读取LLM的API地址、密钥、模型名，实现配置与代码解耦；
#    - 设置temperature=0.1，降低输出随机性，保证业务回答的稳定性；
# 4. 应用场景：为整个Agent系统提供大模型推理能力，是LLM落地业务的核心入口。