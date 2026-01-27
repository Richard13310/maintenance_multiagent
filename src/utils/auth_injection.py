from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from src.tools import AUTH_REQUIRED_TOOLS

def authToken_inject(state, config: RunnableConfig):
    if messages := state.get("messages", []):
        message = messages[-1]
    if isinstance(message, AIMessage) and len(message.tool_calls):
        for tool_call in message.tool_calls:
            if tool_call.get("name") in AUTH_REQUIRED_TOOLS:
                if "params" not in tool_call.get("args", {}):
                    tool_call["args"]["params"] = {}
                tool_call["args"]["params"]["authToken"] = config.get("configurable", {}).get("authToken", "")
            tool_call["arguments"] = tool_call["args"].get("params", tool_call["args"])
    return state
