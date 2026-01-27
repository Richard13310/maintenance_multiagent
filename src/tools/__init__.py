from src.tools.query_tools import QUERY_TOOLS

# 意图与工具的映射配置：定义不同意图对应的工具信息（API密钥、请求URL、中文名称）
# INTENT_CONFIG_MAP = {
#     "configure_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "配置类工具库"},
#     "health_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "健康类工具库"},
#     "information_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "用户相关的信息查询工具库"},
#     "query_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "获取uptime分析数据的工具"},
#     "statistics_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "统计类工具库"},
#     "device_tool": {"api_key": "", "req_url": "Internal function", "chinese_name": "设备类工具库"},
#     "log_tools": {"api_key": "", "req_url": "Internal function", "chinese_name": "日志类工具库"},
#     "query_knowledge_base": {"api_key": "", "req_url": "Internal function", "chinese_name": "知识库查询工具库"},
# }

# 工具模块路径配置：映射工具分类与对应的模块路径
# TOOL_MODULES = {
#     "COMMON_TOOLS": "src.tools.common_tools",
#     "CONFIGURE_TOOLS": "src.tools.configure_tools",
#     "HEALTH_TOOLS": "src.tools.health_tools",
#     "QUERY_TOOLS": "src.tools.query_tools",
#     "STATISTICS_TOOLS": "src.tools.statistics_tools",
#     "USERS_RELATED_TOOLS": "src.tools.users_related_tools",
#     "INFORMATION_TOOLS": "src.tools.information_tools",
#     "DEVICE_TOOLS": "src.tools.device_tools",
#     "LOG_TOOLS": "src.tools.log_tools",
#     "RAG_TOOLS": "src.tools.rag_tools"
# }

# 动态加载工具模块的函数
# def _load_module(module_path: str):
#     try:
#         return importlib.import_module(module_path)
#     except ImportError:
#         return None

# 自动发现工具：根据TOOL_MODULES配置加载所有工具
# def auto_discover_tools(import_modules: Dict[str, str]) -> Dict[str, List]:
#     discovered_tools = {name.upper(): [] for name in import_modules.keys()}
#     for module_name, module_path in import_modules.items():
#         module = _load_module(module_path)
#         if module:
#             discovered_tools[module_name.upper()].extend(
#                 obj for _, obj in inspect.getmembers(module, lambda x: isinstance(x, StructuredTool))
#             )
#     return discovered_tools

# # 自动加载工具
# auto_tools = auto_discover_tools(TOOL_MODULES)

# 需要认证的工具列表（占位符，实际项目中会通过工具函数的参数注解自动收集）
# 记录哪些业务工具需要注入认证Token
AUTH_REQUIRED_TOOLS = [
    # "configure_tool",   # 配置类工具
    # "health_tool",      # 健康类工具
    # "query_tool",       # 查询类工具
    # "statistics_tool",  # 统计类工具
    # "device_tool",      # 设备类工具
    # "log_tools",        # 日志类工具
    # "information_tool", # 信息查询工具
]

# 代码说明：
# 1. 工具导入与配置：
#    - 导入了`query_tools`中的工具列表；
#    - 注释部分定义了**意图-工具映射**和**工具模块路径**，用于管理不同业务场景对应的工具。

# 2. 动态工具加载：
#    注释的函数（`_load_module`、`auto_discover_tools`）实现了**工具的自动发现**，可根据模块路径动态加载所有`StructuredTool`类型的工具，提升项目的可扩展性。

# 3. 认证工具管理：
#    `AUTH_REQUIRED_TOOLS`是需要认证的工具列表，配合之前的`authToken_inject`函数，可实现这类工具的自动认证Token注入。

# 该文件是工具模块的**统一配置与管理入口**，通过集中配置、动态加载的方式，实现了工具的模块化管理，适配LLM Agent对多工具调用的需求。