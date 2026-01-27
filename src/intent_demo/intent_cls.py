from typing import Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from src.intent_demo.intent_schemas import IntentSchema, State
from src.intent_demo.intent_map import INTENT_STR_KEY
from src.utils.model_hook import get_last_user_input


def build_intent_chain(llm, intent_str_key: Dict[str, str]):
    # 系统提示词：定义意图分类器的角色与输出格式
    system_text = ( # 查询统计、设备管理、健康自检、提单系统
        "你是一个严格的意图分类器。只返回JSON，且必须符合给定的Pydantic。"
        "\n##意图分类期规则："
"\n1. **业务相关**：如果问题与设备业务相关（涉及场站、设备、运维等），则在以下映射中选择最贴近的意图（名称->key）："
        + "\n".join([f" {k} -> {v}" for k, v in intent_str_key.items()])  # "\n uptime分析列表 -> uptimeList"
        + "\n 如果完全没有匹配但与设备业务相关，可以选择最接近的一个"  # "文本1"+"文本2"→"文本1文本2"
"\n2. **业务无关**：如果问题与设备业务无关，按以下规则分类："
"\n - **提问类（question）**：所有询问信息的问题，包括但不限于："
"\n   * '什么是XXX？'、'XXX是什么？'、'XXX的XXX是什么？'"
"\n   * '如何XXX？'、'怎么XXX？'、'怎样XXX？'"
"\n   * 'XXX的电话是什么？'、'XXX的地址是什么？'、'XXX的联系方式是什么？'"
"\n   * 任何以问号结尾的询问类问题"
"\n - **闲聊类（chit_chat）**：非询问性的对话，包括："
"\n   * 问候：'你好'、'早上好'、'再见'"
"\n   * 闲聊：'今天天气怎么样'、'讲个笑话'、'聊聊天'"
"\n   * 非问题性的陈述或感叹"
"\n⚠️ 关键区分：只要是以问号结尾的询问类问题，即使与设备业务无关，也应该归类为 'question'（提问），而不是 'chit_chat'（闲聊）。"
"\n⚠️ 分类关键原则："
"\n1. 识别核心动作词：'统计'、'查询'、'分析'、'配置'等"
"\n2. 识别目标对象：'设备信息'、'场站信息'、'设备昵称'等"
"\n3. 忽略上下文前缀：'场站 XXX'、'设备 XXX'等分类词不影响意图"
"\n示例（重要）："
"\n- '场站 深圳场站 设备信息统计' → 核心='设备信息统计' → chargeStatistics"
"\n- '查询深圳场站信息' → 核心='查询场站信息' → stationInfo"
"\n- 'Autel Europe UK Ltd的电话是什么？' → 业务无关，提问类（询问联系方式） → question"
"\n- '今天天气怎么样' → 业务无关，闲聊类（闲聊话题） → chit_chat"
"\n严格按照此JSON模式输出：\n{format_instructions}"
    )
    # 初始化Pydantic解析器，用于将LLM输出转换为IntentSchema对象 [泛型指定](用户传进的类对象)
    parser = PydanticOutputParser[IntentSchema](pydantic_object=IntentSchema)
    # 构造提示词模板（包含系统角色与用户查询）
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("user", "{query}")
    ]).partial(format_instructions=parser.get_format_instructions())
    # 构建“提示词→LLM→解析器”的处理链，逻辑依赖下的唯一合理顺序
    return prompt | llm | parser

def intent_cls_factory(llm, intent_str_key: Dict[str, str] = None):
    # 构建意图分类链（默认使用DEFAULT_INTENT_MAP）
    chain = build_intent_chain(llm, intent_str_key or INTENT_STR_KEY)

    def node(state: State):
        # 从状态中获取对话消息列表
        messages = state.get("messages", [])
        user_text = ""
        # 反向遍历消息，提取最新的用户输入
        user_text = get_last_user_input(messages)
        # 若无用户输入，返回空消息
        if not user_text: return {"messages": []}
        # 调用分类链，获取意图识别结果
        result: IntentSchema = chain.invoke({"query": user_text})
        # 构造AI消息，记录意图分类结果
        ai_msg = AIMessage(content=result.model_dump_json(), name="intent_cls")
        # 返回更新后的状态（包含新消息、意图标识、意图名称、置信度）
        return {
            "messages": messages + [ai_msg], # [ai_mas]就够了，State里是messages: Annotated，graph会自动合进去
            "intent_name": result.intent_name,
            "intent_key": result.intent_key,
            "confidence": result.confidence,
        }
    return node

# 代码说明：
# 1. 功能定位：该文件是LLM Agent的“意图分类模块”，负责将用户输入转换为标准化的意图信息；
# 2. 核心逻辑：
#    - build_intent_chain：构建“提示词+LLM+解析器”的处理链，定义意图分类的规则与输出格式；
#    - intent_classifier_node_factory：生成意图分类节点，从对话中提取用户输入，调用分类链得到意图结果，并更新状态；
# 3. 技术特点：
#    - 使用PydanticOutputParser确保LLM输出符合IntentSchema结构；
#    - 支持自定义意图映射表，适配不同业务场景；
# 4. 应用场景：作为LangChain Agent工作流的前置节点，实现用户意图的自动识别，为后续工具调度提供依据，是Agent理解用户需求的核心组件。
