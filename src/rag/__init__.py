# 导入RAG Agent节点工厂函数
from src.rag.rag_agent import create_simple_rag_node

# 定义对外暴露的接口
__all__ = ["create_simple_rag_node"]

# 代码说明：
# 1. 核心作用：作为rag包的初始化文件，统一管理包内对外导出的接口；
# 2. 功能解析：
#    - 导入create_rag_agent_node函数，简化其他模块的导入操作；
#    - 通过__all__显式指定对外暴露的成员，遵循Python包管理规范；
# 3. 应用场景：让其他模块可通过`from src.rag import create_rag_agent_node`快速调用RAG节点工厂函数。