import requests
import json
from typing import Optional
from pydantic import BaseModel, Field
from core.config import get_settings

# 加载项目配置，获取外部API的域名前缀
settings = get_settings()
REQ_DOMAIN_URL = getattr(settings, 'push_config', None) and settings.push_config.energy_domain_url or ""  # 此处URL后缀可能不完整

'''
Pydantic模型定义：用于规范参数格式，同时生成工具描述（适配LangChain工具调用）
'''
# 占位参数模型：仅用于填充占位字段
class DummyParams(BaseModel):
    dummy: str = Field(..., description="占位参数，留空值即可")

# 带认证信息的参数模型：包含自动处理的authToken
class WithAuthInfo(BaseModel):
    authToken: Optional[str] = Field('', description="认证token，系统自动处理")

'''
分离认证信息与请求参数的工具函数
作用：从Pydantic模型中提取authToken（用于请求头），并清理请求体中的认证字段
'''
def get_auth_and_payload(params: BaseModel) -> tuple[dict, str]:
    # 获取认证token（无则为空）
    authorization = getattr(params, 'authToken', '') or ''
    # 将Pydantic模型转换为字典格式的请求体
    payload = params.model_dump()
    # 从请求体中移除authToken（避免重复传递）
    if "authToken" in payload:
        payload.pop("authToken")
    return payload, authorization

'''
外部API的POST请求工具函数
功能：封装HTTP POST请求，自动处理请求头、认证信息，返回API响应（或错误信息）
'''
def post_external_api(url_suffix: str, params: dict, authorization: str = "") -> dict:
    # 基础请求头（指定JSON格式）
    headers = {"Content-Type": "application/json"}
    # 若有认证信息，添加到请求头
    if authorization:
        headers["Authorization"] = authorization
    # 拼接完整请求URL（若配置了域名前缀则拼接，否则直接使用传入的后缀）
    req_url = f"{REQ_DOMAIN_URL}{url_suffix}" if REQ_DOMAIN_URL else url_suffix

    try:
        # 发送POST请求（超时60秒）
        response = requests.post(req_url, headers=headers, json=params, timeout=60)
        # 检查请求是否成功（非2xx状态码会抛出异常）
        response.raise_for_status()
        # 返回JSON格式的响应结果
        return response.json()
    except Exception as e:
        # 捕获异常并返回错误信息
        return {'error': f"请求失败: {str(e)}"}