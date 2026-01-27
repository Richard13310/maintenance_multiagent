# 从闲聊模块的核心文件导入工厂函数
from src.chit_chat.chit_chat import create_chit_chat_node

# 定义对外暴露的接口，简化包的导入方式
__all__ = ["create_chit_chat_node"]

# 代码说明：
# 1. 核心作用：该文件是`chit_chat`包的初始化文件，用于统一管理包内的对外导出接口；
# 2. 功能解析：
#    - 导入`create_chit_chat_node`函数，将其纳入包的命名空间；
#    - 通过`__all__`显式指定包对外暴露的成员，当使用`from src.chit_chat import *`时，仅会导入`create_chit_chat_node`；
# 3. 应用场景：遵循Python的包管理规范，简化其他模块对闲聊节点工厂函数的导入（如`from src.chit_chat import create_chit_chat_node`），提升代码的可读性与维护性。