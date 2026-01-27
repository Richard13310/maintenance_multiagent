知识问答和闲聊系统框架  
本项目是基于 LangGraph 构建的设备运维场景智能体系统，集成知识问答与闲聊双核心能力，为 LangGraph 多智能体开发的学习版实现。  
一、系统整体架构  
用户输入经意图分类后，分流至知识问答或闲聊系统处理：  
用户输入 → 意图分类（intent_cls）  
分支处理：  
业务意图 → 知识问答系统（query_agent）：基于 React Agent 调用工具完成业务查询  
闲聊意图 → 闲聊系统（chit_chat）：经安全过滤后调用 LLM 生成闲聊回复  
二、核心模块说明  
1. 知识问答系统（src/agent/query_agent.py）  
功能：处理设备运维相关业务查询（如设备状态、uptime 分析、场站信息查询）  
实现方式：  
通过create_react_agent构建 React Agent  
支持工具调用、多轮对话与上下文管理  
核心特性：  
自动工具调用与结果处理  
上下文自动清理与截断  
API 错误检测与报告  
数据格式化（时间、百分比、表格）  
2. 闲聊系统（src/chit_chat/chit_chat.py）  
功能：处理非业务类日常对话  
实现方式：多层安全防护、意图识别、缓存机制、分层 LLM 调用、上下文管理  
关键组件：  
PreSafetyFilter：前置安全过滤器（检查用户输入）  
PostSafetyFilter：后置安全过滤器（检查 LLM 输出）  
IntentRecognizer：意图识别器（区分业务 / 闲聊）  
ResponseCache：回复缓存（90% 相似度匹配）  
LLMController：LLM 控制器（分层模型调用）  
核心特性：  
特性分类	具体实现  
多层安全防护	敏感词 / 话题分类过滤（前置 + 后置）  
智能意图识别	业务关键词匹配、问候语规则匹配  
性能优化	回复缓存、上下文缓存、分层模型调用  
内容质量控制	100 字字数限制、重复检测、换行处理  
3. 图结构（src/graph/graph_simple.py）  
核心路由逻辑：  
intent_cls → 初步意图分类  
intent_classifier → 进一步判断业务 / 闲聊意图  
query_agent → 知识问答业务处理  
chit_chat → 闲聊对话处理  
三、使用方法  
1. 运行交互式聊天  
bash
运行  
cd learning_maintenance  
python chat.py  
2. 功能测试  
知识问答测试：输入业务相关问题  
plaintext  
User: 查询设备状态    
User: uptime分析    
User: 查询场站信息    
闲聊系统测试：输入日常对话  
plaintext  
User: 你好  
User: 今天天气怎么样？  
User: 谢谢  
四、项目文件结构  
plaintext  
learning_maintenance/  
├── src/  
│   ├── agent/  
│   │   └── query_agent.py  # 知识问答Agent核心  
│   ├── chit_chat/  
│   │   ├── __init__.py  
│   │   └── chit_chat.py    # 闲聊系统实现  
│   ├── graph/  
│   │   └── graph_simple.py # LangGraph图结构定义  
│   ├── prompts/  
│   │   └── agent_prompts.py # Prompt模板（问答/闲聊）  
│   ├── tools/  
│   │   └── query_tools.py  # 业务查询工具定义  
│   ├── utils/  
│   │   └── model_hook.py   # 上下文管理工具  
│   ├── chat.py             # 项目入口文件  
└── README_KNOWLEDGE_CHAT.md # 项目说明文档  
五、扩展指南  
1. 添加新查询工具  
在src/tools/query_tools.py中定义工具：  
python  
运行  
@tool  
def my_new_tool(param: str) -> str:  
    """新工具的描述"""  
    return "工具执行结果"  
在src/agent/query_agent.py中导入并注册：  

from src.tools.query_tools import my_new_tool  
  
query_assistant = create_react_agent(  
    llm_no_think.bind_tools([simple_query_tool, my_new_tool]),  
    tools=[simple_query_tool, my_new_tool],  
    prompt=query_prompt,  
)  
2. 修改闲聊系统行为  
安全策略：编辑SafetyPolicyLibrary相关逻辑  
意图识别：修改IntentRecognizer的匹配规则  
缓存策略：调整ResponseCache的相似度阈值与缓存有效期  
LLM 调用：修改LLMController的分层模型选择逻辑  
3. 调整 Prompt 模板  
编辑src/prompts/agent_prompts.py：  
query_prompt：修改知识问答的提示词，调整 AI 的回答风格与规则  
chit_chat_prompt：修改闲聊的提示词，定义闲聊的角色与回复规范  
六、学习与注意事项  
1. 学习建议  
理解图结构：阅读src/graph/graph_simple.py，掌握 LangGraph 的节点与路由设计  
理解知识问答：阅读src/agent/query_agent.py，学习 React Agent 的工具调用模式  
理解闲聊系统：阅读src/chit_chat/chit_chat.py，分析多层安全防护与缓存机制  
理解工具定义：阅读src/tools/query_tools.py，掌握 LangChain 工具的开发规范  
2. 注意事项  
环境配置：确保llm_db_config和core模块的配置文件正确  
LLM 配置：在config/local.yaml中填写有效的 LLM API 密钥与地址  
状态管理：通过checkpointer实现对话状态的持久化与断点续跑  
错误处理：系统内置基础错误兜底，建议新增日志模块完善问题排查  
3. 与主项目的区别（学习版）  
特性	学习版状态	说明  
核心问答 / 闲聊系统	保留	维持业务核心能力  
消息过滤逻辑	简化	降低学习门槛  
安全防护体系	保留	完整保留多层过滤能力  
缓存 / 性能优化机制	保留	维持核心性能优化逻辑  
SSE/Kafka 推送系统	移除	简化系统依赖  
工具发现机制	移除	改为手动注册工具  
意图分类逻辑	简化	减少复杂的意图匹配规则  
七、下一步优化方向  
扩展工具库：新增设备故障排查、数据统计等业务工具  
优化 Prompt：通过提示词工程提升问答的准确性与闲聊的自然度  
强化安全策略：补充更多敏感词与风险话题，完善安全过滤规则  
优化缓存策略：基于用户对话频率动态调整缓存有效期  
日志与监控：添加日志模块，记录系统运行状态与用户交互数据  
多模态支持：扩展系统能力，支持图片、语音等多模态输入输出  
