# 知识问答和闲聊系统框架
本项目是基于 LangGraph 构建的设备运维场景智能体系统，集成知识问答与闲聊双核心能力，为 LangGraph 多智能体开发的学习版实现。
1.系统整体架构
用户输入经意图分类后，分流至知识问答或闲聊系统处理：
用户输入 → 意图分类（intent_cls）
分支处理：
业务意图 → 知识问答系统（query_agent）：基于 React Agent 调用工具完成业务查询
闲聊意图 → 闲聊系统（chit_chat）：经安全过滤后调用 LLM 生成闲聊回复  
# 多模态设备运维 RAG Agent - 前端对接文档  
## 一、服务部署
### 1. 环境依赖
```bash
# 安装 FastAPI 及运行依赖
pip install fastapi uvicorn python-multipart websockets

# 安装原有 Agent 依赖（如未安装）
pip install langgraph langchain-core langchain-community paddleocr pymupdf numpy
```
### 2. 启动服务
```bash
# 启动 FastAPI 服务（默认端口 8000）
python app.py

# 或手动指定端口
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
### 3. 验证服务
```bash
# 访问健康检查接口：
http://localhost:8000/health

# 返回如下json则服务正常：
{"status":"healthy","service":"rag-agent-api"}
```
## 二、接口说明
1. 核心接口列表

| 接口类型   | 接口地址      | 功能                     | 适用场景               |
| :--------- | :------------ | :----------------------- | :--------------------- |
| RESTful    | POST /api/chat | 同步获取回答（非流式）| 简单问答、测试         |
| WebSocket  | WS /ws/chat   | 流式获取回答（实时返回） | 生产环境、前端聊天框   |
| 健康检查   | GET /health   | 验证服务状态             | 运维监控               |
2. RESTful 接口（/api/chat）

请求参数（JSON）

| 参数名      | 类型   | 必填 | 说明                            |
| :---------- | :----- | :--- |:------------------------------|
| user_input  | string | 是   | 用户问题（如 "设备显示 008 通信故障怎么处理？"）  |
| session_id  | string | 是   | 会话 ID（区分不同用户，如 "user_123456"） |
| auth_token  | string | 否   | 认证令牌（开发环境留空）                  |
请求示例（curl）
```bash
curl -X POST http://localhost:8000/api/chat \
-H "Content-Type: application/json" \
-d '{
    "user_input": "设备显示008通信故障怎么处理？",
    "session_id": "learning_session",
    "auth_token": ""
}'
```
响应示例
```base
{
    "code": 200,
    "message": "success",
    "data": {
        "session_id": "learning_session",
        "user_input": "设备显示008通信故障怎么处理？",
        "answer": "008通信故障的原因主要是通信模块离线...（完整回答）"
    }
}
```
3. WebSocket 接口（/ws/chat）

连接地址
```base
ws://localhost:8000/ws/chat
```
交互流程

    前端建立 WebSocket 连接；
    前端发送 JSON 格式的提问消息；
    后端流式返回回答片段；
    回答结束后返回 stream_end 标记。
前端示例（JavaScript）
```base
// 建立 WebSocket 连接
const ws = new WebSocket('ws://localhost:8000/ws/chat');

// 连接成功回调
ws.onopen = function() {
    console.log('WebSocket 连接成功');
    // 发送提问消息
    ws.send(JSON.stringify({
        user_input: "设备显示008通信故障怎么处理？",
        session_id: "learning_session",
        auth_token: ""
    }));
};

// 接收流式响应
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.code === 200) {
        if (data.message === "stream_end") {
            console.log("回答结束");
            return;
        }
        // 实时追加回答内容到聊天框
        console.log("收到回答片段：", data.data.chunk);
        // document.getElementById("chat-content").innerText += data.data.chunk;
    } else {
        console.error("错误：", data.message);
    }
};

// 连接关闭回调
ws.onclose = function() {
    console.log('WebSocket 连接关闭');
};

// 错误回调
ws.onerror = function(error) {
    console.error('WebSocket 错误：', error);
};
```
