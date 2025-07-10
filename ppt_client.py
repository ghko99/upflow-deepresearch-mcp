from dotenv import load_dotenv
from langchain.callbacks.base import AsyncCallbackHandler
import uuid
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
import asyncio
from langchain_core.messages import HumanMessage

load_dotenv()
MCP_CONFIG_FILE = "mcp.json"
SESSION_ID = str(uuid.uuid4())  # 재시작마다 새 thread_id → RAM 초기화
checkpointer = MemorySaver()  # 📌 종료 시 모든 상태가 사라집니다

class ToolPrintHandler(AsyncCallbackHandler):
    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else str(serialized)
        print(f"🔧  {name} called with → {input_str}")

    async def on_tool_end(self, output, *, run_id, **kwargs):
        print(f"✅  tool returned ← {output}")

def load_mcp_config():
    try:
        with open(MCP_CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {MCP_CONFIG_FILE}")
        return {}

async def build_agent():

    SERVERS = load_mcp_config().get("mcpServers", {})
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=8000)
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    for t in tools:
        if t.name == "read_pdf":
            import json, pprint
            pprint.pprint(t.as_openai_function())
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    print("툴 목록:", [t.name for t in tools])
    return agent


async def main():
    agent = await build_agent()
    handler = ToolPrintHandler()

    print("🔗 연결됨! 'exit' 입력 시 종료 (재시작하면 기록이 사라집니다)")

    while True:
        question = input("👤  ")
        if question.lower() == "exit":
            break

        result = await agent.ainvoke(
            {
                # ✅ create_react_agent 가 기대하는 형식
                "messages": [HumanMessage(content=question)],
            },
            config={
                "callbacks": [handler],
                "configurable": {"thread_id": SESSION_ID},  # ✅ MemorySaver 요구사항
            },
        )

        print("\n💬 Final answer:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())