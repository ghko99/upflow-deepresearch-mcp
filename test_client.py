"""
chatbot_long_memory.py (RAM‑only 버전)
────────────────────────────────────────
▶ RUN :  python chatbot_long_memory.py

✔️ 핵심 목표  
  • 기존 ReAct 예제(tool‑callback 로깅) 그대로 유지  
  • **프로세스를 재시작하면 메모리 초기화** (대화 기록이 남지 않음)  

🛠️ 설치:
```bash
pip install -U langgraph langchain-openai langchain-core \
               langchain-mcp-adapters tiktoken python-dotenv
```
`.env` 또는 OS 환경 변수에 **OPENAI_API_KEY**를 설정하세요.
────────────────────────────────────────
"""
import uuid   
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.callbacks.base import AsyncCallbackHandler
from langgraph.checkpoint.memory import MemorySaver  # 🗑️ RAM‑only 체크포인터
from langchain_core.messages import HumanMessage

######################################################################
# 0. 환경 변수 로드
######################################################################

load_dotenv()
BASE_DIR = Path(__file__).parent  # 현재 디렉터리 사용 – 파일 저장 없음
SESSION_ID = str(uuid.uuid4())      # 재시작하면 새 ID → RAM 메모리 초기화 컨셉과 잘 맞음
######################################################################
# 1. Callback 핸들러 (변경 없음)
######################################################################

class ToolPrintHandler(AsyncCallbackHandler):
    async def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else str(serialized)
        print(f"🔧  {name} called with → {input_str}")

    async def on_tool_end(self, output, *, run_id, **kwargs):
        print(f"✅  tool returned ← {output}")

######################################################################
# 2. MCP 서버 목록 (예시 그대로)
######################################################################
with open('mcp.json','r') as f:
    SERVERS = json.load(f)

######################################################################
# 3. 에이전트 + 비휘발성 메모리 제거 (MemorySaver만 사용)
######################################################################

checkpointer = MemorySaver()  # 📌 종료 시 모든 상태가 사라집니다

async def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    print("툴 목록:", [t.name for t in tools])
    return agent

######################################################################
# 4. 메인 루프
######################################################################

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
