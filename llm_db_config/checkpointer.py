# 创建内存型检查点存储，用于LangGraph的状态持久化
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
# 替换为文件型检查点存储（状态会保存到本地 .langgraph/checkpoints 文件夹）
# from langgraph.checkpoint.local import LocalFileSaver
# checkpointer = LocalFileSaver()
"""
# 生成图
class ChatState(BaseModel):
    messages: List[dict] = Field(default_factory=list)  # 存储{"role": "...", "content": "..."}
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="你的API_KEY")
def generate_response(state: ChatState) -> ChatState:
    response = llm.invoke(state.messages)
    state.messages.append({"role": "assistant", "content": response.content})
    return state
graph_builder = StateGraph(ChatState)
graph_builder.add_node("generate", generate_response)
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)
# 创建内存检查点存储（短期记忆用默认模式）
checkpointer = MemorySaver()
# 编译图（不指定session_id，每次运行用临时ID）
graph = graph_builder.compile(checkpointer=checkpointer)

# 第一次运行：输入用户问题（初始化短期记忆）
result1 = graph.invoke({"messages": [{"role": "user", "content": "我叫小明，记住我的名字"}]})
print("助手回复：", result1.messages[-1]["content"])  # 助手会回应并确认记住名字
# 第二次运行：不传递历史，测试是否保留记忆（短期记忆已丢弃）
result2 = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]})
print("助手回复：", result2.messages[-1]["content"])  # 助手会说“不知道”，因为短期记忆


SESSION_ID = "xiaoming_chat_2024"
# 第一次运行：传入session_id，初始化长期记忆
# 关键：用 invoke 的 config 参数指定 session_id
result1 = graph.invoke({"messages": [{"role": "user", "content": "我叫小明，记住我的名字"}]}, 
                        config={"configurable": {"checkpoint_id": SESSION_ID}})
print("助手回复：", result1.messages[-1]["content"])  # 助手确认记住名字
# 第二次运行：传入相同的session_id，恢复长期记忆
result2 = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, 
                        config={"configurable": {"checkpoint_id": SESSION_ID}})
print("助手回复：", result2.messages[-1]["content"])  # 助手会回答“你叫小明”，记忆保留成功

"""