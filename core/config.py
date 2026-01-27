from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from core.config_model import LLMConfig, MCPConfig, WorkflowConfig
from typing import Dict, Any
import os, yaml
from pathlib import Path


class Settings(BaseSettings):
    """系统总配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    llm: LLMConfig = LLMConfig()
    mcp: MCPConfig = MCPConfig()
    workflow: WorkflowConfig = WorkflowConfig()


# 配置文件映射：环境名→配置文件路径
CONF_FILE_MAP = {
    "local": "config/local.yaml",
}


@lru_cache()
def load_yaml_config() -> Dict[str, Any]:
    """加载YAML配置文件，支持多路径查找与异常处理"""
    env = os.environ.get("NS_ENV", "")
    print(f"----------ENVIRONMENT: {env}----------NS_ENV: {env}----------")
    use_config_file = CONF_FILE_MAP.get(env, "config/local.yaml")

    config_file_path = None
    # 方式1：从PYTHONPATH查找配置文件
    if os.environ.get("PYTHONPATH"):
        config_file_path = Path(os.environ.get("PYTHONPATH")) / use_config_file
        if not config_file_path.exists():
            config_file_path = None

    # 方式2：从当前文件路径推断项目根目录查找
    if config_file_path is None or not config_file_path.exists():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # core/ → 项目根目录
        config_file_path = project_root / use_config_file
        if not config_file_path.exists():
            config_file_path = None

    # 方式3：使用相对路径（向后兼容）
    if config_file_path is None or not config_file_path.exists():
        config_file_path = Path(use_config_file)

    # 配置文件不存在则抛出异常
    if not config_file_path.exists():
        raise FileNotFoundError(
            f"配置文件 {use_config_file} 不存在。\n"
            f"尝试的路径:\n"
            f"1. PYTHONPATH路径: {Path(os.environ.get('PYTHONPATH', '')) / use_config_file}\n"
            f"2. 项目根目录路径: {Path(__file__).parent.parent / use_config_file}\n"
            f"3. 相对路径: {Path(use_config_file)}\n"
            f"请检查环境变量 NS_ENV 或 PYTHONPATH 是否正确，或确保配置文件存在"
        )

    # 打印配置文件路径（兼容Windows编码）
    try:
        print(f"{'=' * 50} Config File: {config_file_path} {'=' * 50} env:{env}")
    except UnicodeEncodeError:
        print(f"Config File: {config_file_path}, env: {env}")

    # 加载YAML配置内容
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            return config_data
    except Exception as e:
        print(f"\033[91m警告：无法加载配置文件 {config_file_path}: {e}\033[0m")
        return {}


@lru_cache()
def get_settings() -> Settings:
    """获取系统配置实例（单例），合并YAML与环境变量配置"""
    # 加载YAML配置
    yaml_config = load_yaml_config()

    # 从环境变量创建基础配置实例
    settings = Settings()

    # 将YAML配置转换为对应模型并合并到settings
    for key in yaml_config.keys():
        if not hasattr(settings, key):
            continue
        s_class = getattr(settings, key).__class__
        setattr(settings, key, s_class(**yaml_config[key]))  # YAML配置转模型实例

    return settings

# 代码说明：
# 1. 功能定位：该文件是系统的**配置核心**，实现多源配置（YAML/环境变量）的加载、合并与校验；
# 2. 核心逻辑：
#    - load_yaml_config：按PYTHONPATH→项目根目录→相对路径的优先级查找配置文件，支持编码兼容与异常兜底；
#    - get_settings：通过lru_cache实现单例，将YAML配置转换为Pydantic模型实例并合并到基础配置；
# 3. 技术特点：
#    - 多路径查找：解决不同部署环境下配置文件路径不一致的问题；
#    - 类型安全：通过Pydantic模型将YAML字典转换为强类型实例，避免配置取值错误；
#    - 异常友好：配置文件不存在时抛出详细的路径提示，便于问题排查；
# 4. 应用场景：为整个Agent系统提供统一的配置入口，所有模块通过get_settings()获取标准化的配置实例，是系统配置解耦的关键。