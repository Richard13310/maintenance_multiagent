"""
多模态 RAG Agent - FastAPI 接口服务
功能：提供 HTTP 接口供前端调用，支持流式响应/会话管理
"""
from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage
from src.graph.graph_simple import graph

from fastapi import FastAPI,WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, Dict, Any, Optional

# ========== FastAPI 初始化 ==========
app = FastAPI(
    title="多模态设备运维 RAG Agent API",
    description="支持文本/图片/PDF多模态问答的设备运维助手",
    version="1.0.0"
)
# 跨域配置（前端对接必需）
app.add_middleware(
    CORSMiddleware, # 跨域资源共享中间件
    allow_origins=["*"],  # 允许的请求源（域名/端口）
    allow_credentials=True,  # 允许携带Cookie/认证信息
    allow_methods=["*"],  # 允许的HTTP方法（GET/POST/PUT等）
    allow_headers=["*"],  # 允许的请求头（Token/Content-Type等）
)
# ========== 数据模型（前端传参规范） ==========
class ChatRequest(BaseModel):
    """前端对话请求参数"""
    user_input: str
    session_id: str
    auth_token: Optional[str] = ""
# ========== 核心函数（原有逻辑改异步） ==========
async def interactive_graph_stream_async(
        user_input: str,
        session_id: str,
        auth_token: str = ""
) -> AsyncGenerator[str, None]:
    """异步版本的 Agent 流式响应生成器"""
    config = {
        "configurable": {
            "thread_id": session_id,
            "auth_token": auth_token
        }
    }
    # 处理中断恢复
    if graph.get_s4tate(config).interrupts:
        send_message = Command([("resume", {"continue": user_input})])
        config["configurable"]["resume"] = True
    else:
        send_message = {"messages": [HumanMessage(content=user_input)]}

    try:
        # 异步流式调用 graph（若原有 graph.stream 是同步的，需包装为异步）
        for event in graph.stream(send_message, config, subgraphs=True, stream_mode=["messages", "custom"]):
            _, event_type, data = event
            if event_type == "messages" and data and len(data) > 0:
                if isinstance(data[0], ToolMessage):
                    yield "\n工具执行完成\n"
                elif hasattr(data[0], "content") and data[0].content:
                    yield data[0].content
    except Exception as e:
        yield f"流式执行错误: {str(e)}"


# ========== HTTP 接口（RESTful） ==========
@app.post("/api/chat", summary="同步对话接口（非流式）")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    同步获取 Agent 回答（适合简单场景）
    - user_input: 用户问题（如"设备显示008通信故障怎么处理？"）
    - session_id: 会话ID（用于区分不同用户，如"user_123"）
    - auth_token: 可选认证令牌（开发环境留空）
    """
    response_content = ""
    async for chunk in interactive_graph_stream_async(request.user_input, request.session_id, request.auth_token):
        response_content += chunk
    return {
        "code": 200,
        "message": "success",
        "data": {
            "session_id": request.session_id,
            "user_input": request.user_input,
            "answer": response_content
        }
    }


# ========== WebSocket 接口（流式响应，推荐前端使用） ==========
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 流式对话接口（推荐前端使用，实时返回回答）
    前端连接后，需发送 JSON 格式数据：
    {
        "user_input": "设备显示008通信故障怎么处理？",
        "session_id": "learning_session",
        "auth_token": ""
    }
    """
    await websocket.accept()
    try:
        while True:
            # 接收前端消息
            data = await websocket.receive_json()
            user_input = data.get("user_input", "")
            session_id = data.get("session_id", "")
            auth_token = data.get("auth_token", "")

            if not user_input or not session_id:
                await websocket.send_json({
                    "code": 400,
                    "message": "user_input 和 session_id 为必填参数"
                })
                continue

            # 流式返回回答
            async for chunk in interactive_graph_stream_async(user_input, session_id, auth_token):
                await websocket.send_json({
                    "code": 200,
                    "message": "success",
                    "data": {
                        "chunk": chunk,
                        "session_id": session_id
                    }
                })

            # 流式结束标记
            await websocket.send_json({
                "code": 200,
                "message": "stream_end",
                "data": {"session_id": session_id}
            })
    except WebSocketDisconnect:
        print(f"会话 {session_id} 已断开")
    except Exception as e:
        await websocket.send_json({
            "code": 500,
            "message": f"服务器错误: {str(e)}"
        })


# ========== 测试接口 ==========
@app.get("/health", summary="健康检查接口")
async def health_check():
    """用于验证服务是否正常运行"""
    return {"status": "healthy", "service": "rag-agent-api"}


if __name__ == "__main__":
    import uvicorn
    # 方式1：启动 FastAPI 服务（推荐）
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 服务端口
        reload=True      # 开发环境自动重载
    )
