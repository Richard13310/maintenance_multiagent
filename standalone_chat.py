from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage
from src.graph.graph_simple import graph  # 确保该导入路径正确

class BuiltIn_Chat(object):
    @staticmethod
    def interactive_graph_stream(user_input: str, session_id: str, auth_token: str = ""):
        """静态方法：无需self参数，处理graph流式交互"""
        config = {"configurable": {"thread_id": session_id, "auth_token": auth_token}}
        if graph.get_state(config).interrupts:
            send_message = Command([("resume", {"continue": user_input})])  # 相当于SystemMessage
            config["configurable"]["resume"] = True
        else:
            send_message = {"messages": [HumanMessage(content=user_input)]}
        
        try:
            for event in graph.stream(send_message, config, subgraphs=True, stream_mode=["messages", "custom"]):
                # 解析事件（event 格式：(节点名, 事件类型, 数据)）
                _, event_type, data = event
                if event_type == "messages" and data and len(data) > 0:
                    if isinstance(data[0], ToolMessage):
                        print("\n工具执行完成")
                    elif hasattr(data[0], "content") and data[0].content:
                        print(data[0].content, end="", flush=True)
        except Exception as e:
            print(f"流式执行错误: {str(e)}")

    def chat_stream(self):
        """实例方法：命令行交互入口"""
        print("学习Agent 框架 - 输入/quit/exit/q 退出")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            # 调用静态方法：通过类名调用（推荐），或self调用（兼容）
            BuiltIn_Chat.interactive_graph_stream(user_input, session_id="learning_session", auth_token="")

# 测试运行
if __name__ == "__main__":
    chat_agent = BuiltIn_Chat()
    chat_agent.chat_stream()