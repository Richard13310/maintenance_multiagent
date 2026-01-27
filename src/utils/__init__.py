# 导入认证令牌注入工具函数
from src.utils.auth_injection import authToken_inject
# 导入上下文消息裁剪工具函数
from src.utils.model_hook import trim_msg, get_last_user_input

# 定义utils包对外暴露的核心工具接口
__all__ = ["authToken_inject", "trim_msg", "get_last_user_input"]

# 代码说明：
# 1. 核心作用：该文件是`utils`工具包的初始化文件，负责统一对外暴露包内的核心工具函数，简化其他模块的导入操作；
# 2. 功能解析：
#    - 导入`authToken_inject`（认证令牌注入）和`trim_msg`（上下文消息裁剪）两个核心工具函数，纳入包的命名空间；
#    - 通过`__all__`显式指定对外暴露的成员，当使用`from src.utils import *`时，仅会导入这两个函数，避免无关成员的暴露；
# 3. 应用场景：遵循Python包管理的最佳实践，让其他模块可以通过`from src.utils import authToken_inject, trim_msg`快速调用工具函数，提升代码的可读性与维护性，同时实现工具函数的集中管理。