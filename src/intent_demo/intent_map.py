from typing import Dict
# 意图映射表：将用户意图的自然语言描述映射为统一的意图标识
INTENT_STR_KEY: Dict[str, str] = {
    "uptime分析列表": "uptimeList"
}
# 意图-工具映射表：将意图标识映射为对应的处理工具
INTENT_KEY_AGENT: Dict[str, str] = {
    "uptimeList": "query_tool"
}

# 代码说明：
# 1. 核心作用：该文件是意图与工具的映射配置中心，实现“用户自然语言意图→统一意图标识→处理工具”的两层映射；
# 2. 映射逻辑：
#    - DEFAULT_INTENT_MAP：将用户输入的自然语言意图（如“uptime分析列表”）转换为标准化的意图标识（如“uptimeList”）；
#    - INTENT_TO_AGENT：将意图标识映射为具体的处理工具（如“query_tool”）；
# 3. 应用场景：配合planner模块使用，实现用户意图到工具调用的自动化匹配，是LLM Agent中意图路由的关键配置，提升意图识别与工具调度的可维护性。
