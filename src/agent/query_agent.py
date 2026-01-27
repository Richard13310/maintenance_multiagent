"""
工具链 Agent - 学习示例（已迁移到 tool_agent.py）
注意：此文件保留作为向后兼容，新代码请使用 src/agent/tool_agent.py

工具链 Agent = Tool Chain Agent
核心流程：问题 → Agent分析 → 选择工具 → 调用工具 → 处理结果 → 返回答案

与 RAG Agent 的区别：
- 工具链 Agent：调用外部工具（API、数据库等）执行操作
- RAG Agent：从知识库中检索信息，基于文档内容回答
"""
# 向后兼容：导入 tool_agent 中的实现
from src.agent.tool_agent import tool_agent_tool as query_agent_tool

# 定义对外暴露的接口
__all__ = ["query_agent_tool"]

# 代码说明：
# 1. 功能定位：作为历史代码的兼容层，将原query_agent_tool映射到新的tool_agent_tool；
# 2. 设计目的：避免修改原有调用逻辑，实现工具链Agent的平滑迁移；
# 3. 应用场景：供旧模块继续使用query_agent_tool接口，新开发建议直接使用tool_agent.py中的工具。