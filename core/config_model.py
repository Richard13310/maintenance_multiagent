from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class LLLocalConfig(BaseSettings):
    """本地模型配置"""
    model_config = SettingsConfigDict(protected_namespaces=('settings_',))

    model_path: str = ""  # 如"/models/chatglm2-6b"
    device: str = "cuda:0"
    max_length: int = 4096
    temperature: float = 0.1


class LLMInferenceConfig(BaseSettings):
    """LLM推理配置"""
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: int = 30
    retry_count: int = 3


class LLMConfig(BaseSettings):
    """大模型配置"""
    provider: str = ""  # openai, local, custom
    model: str = ""  # 如Qwen3-30B-A3B
    api_key: str = ""
    api_base: str = ""

    local: LLLocalConfig = LLLocalConfig()
    inference: LLMInferenceConfig = LLMInferenceConfig()


class MCPConnectionPoolConfig(BaseSettings):
    """MCP连接池配置"""
    max_connections: int = 50
    max_keepalive_connections: int = 20
    keepalive_expiry: int = 300


class MCPConfig(BaseSettings):
    """MCP协议配置"""
    ops_api_base: str = ""  # 如https://ops.platform.com/api/v1
    api_key: str = Field(default="", env="OPS_API_KEY")
    timeout: int = 30
    retry_count: int = 3
    connection_pool: MCPConnectionPoolConfig = MCPConnectionPoolConfig()


class WorkflowRetryConfig(BaseSettings):
    """工作流重试配置"""
    max_retries: int = 3
    backoff_factor: int = 2
    max_backoff: int = 60


class WorkflowConfig(BaseSettings):
    """工作流配置"""
    max_concurrent: int = 100
    default_timeout: int = 300
    retry_policy: WorkflowRetryConfig = WorkflowRetryConfig()

# 代码说明：
# 1. 功能定位：基于Pydantic定义系统各模块的配置结构，实现配置的类型校验与默认值管理；
# 2. 配置分类：
#    - LLM相关：本地模型、推理参数配置，适配不同部署方式的大模型；
#    - MCP相关：协议与连接池配置，管理外部业务系统的连接；
#    - Workflow相关：工作流并发、重试配置，保障LangGraph的稳定运行；
# 3. 技术特点：
#    - 使用Field绑定环境变量，支持配置的动态注入；
#    - 嵌套配置类，实现复杂配置的结构化管理；
# 4. 应用场景：作为配置读取的基础模型，为config.py提供类型约束，避免配置错误导致的系统异常。