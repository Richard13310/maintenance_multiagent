import logging
from functools import lru_cache

@lru_cache()
def get_logger(name: str) -> logging.Logger:
    """创建日志实例（单例）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 添加控制台处理器
        handler = logging.StreamHandler()
        # 设置日志格式：时间+名称+级别+消息
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        # 设置日志级别为INFO
        logger.setLevel(logging.INFO)
    return logger

# 代码说明：
# 1. 功能定位：创建并缓存日志实例，为系统提供统一的日志输出能力；
# 2. 核心逻辑：
#    - 通过lru_cache实现单例，避免重复创建日志处理器；
#    - 配置控制台输出与日志格式，便于调试与问题排查；
# 3. 技术特点：
#    - 按需创建日志处理器，减少资源占用；
#    - 统一日志级别为INFO，平衡日志信息量与性能；
# 4. 应用场景：在各模块中调用get_logger(__name__)获取日志实例，实现标准化的日志输出。