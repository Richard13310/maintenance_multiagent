from typing import TypedDict, Optional, Dict, List
from typing_extensions import Annotated, Any
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

# 状态对象：运行时作为字典，存储Agent工作流的核心数据
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表（支持消息追加）
    intent_name: Optional[str]               # 意图名称
    intent_key: Optional[str]                # 意图标识
    confidence: Optional[float]              # 意图识别置信度（0~1）
    plan: Optional[Dict[str, Any]]           # 执行计划
    authToken: Optional[str]                 # 认证令牌

# 意图Schema：规范意图识别的输出结构
class IntentSchema(BaseModel):# Pydantic 的 BaseModel 是所有 “数据模型类” 的基类，核心能力是 自动数据验证、类型转换和序列化
    intent_name: str = Field(..., description="中文意图名称；业务相关则填写具体业务意图名称（如“设备信息统计”），业务无关则填写“question/chat_chat”")
    intent_key: str = Field(..., description="意图key；业务相关则返回业务意图key（如“chargeStatistics”），业务无关则返回“question/chat_chat”")
    confidence: float = Field(..., ge=0, le=1, description="0~1的置信度")
    reason: str = Field(..., description="简要判断依据")

# 执行步骤：定义单步工具调用的结构（所有字段可选）
class Step(TypedDict, total=False):
    agent_tool: str                          # 工具名称
    params: Dict[str, Any]             # 工具参数
    summary_after: Optional[bool]      # 执行后是否需要总结

# 执行计划：定义多步工具调用的结构（所有字段可选）
class Plan(TypedDict, total=False):
    type: str                          # 计划类型
    steps: list[Step]              # 步骤列表

# 代码说明：
# 1. 核心作用：该文件是Agent工作流的“数据结构定义中心”，通过Pydantic与TypedDict规范状态、意图、执行计划等核心数据的格式；
# 2. 结构分类：
#    - State：存储工作流的运行时数据，是Agent各节点间传递信息的载体；
#    - IntentSchema：约束意图识别的输出，确保LLM输出符合预期结构；
#    - PlanStep/Plan：定义工具执行计划的层级结构，规范多步工具调用的参数与逻辑；
# 3. 技术特点：
#    - 使用TypedDict定义State，兼顾类型约束与运行时的字典灵活性；
#    - 使用Pydantic的BaseModel定义IntentSchema，实现结构化输出的校验；
# 4. 应用场景：作为Agent工作流的“数据契约”，确保各模块（意图分类、计划生成、工具调用）间的数据格式一致，是复杂Agent系统可维护性的基础保障。
