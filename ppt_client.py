import asyncio, json, uuid
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler

# ------------------------------------------------------------- #
load_dotenv()
SESSION_ID = str(uuid.uuid4())
checkpointer = MemorySaver()
MCP_CONFIG_FILE = "mcp.json"

class ToolPrint(AsyncCallbackHandler):
    async def on_tool_start(self, s, arg, *, run_id, **kw):
        name = s.get("name") if isinstance(s, dict) else str(s)
        print(f"🔧 {name} → {arg}")
    async def on_tool_end(self, out, *, run_id, **kw):
        print(f"✅ 결과 ← {str(out)[:120]}{'...' if len(str(out))>120 else ''}")

def load_servers():
    with open(MCP_CONFIG_FILE) as f:
        return json.load(f)["mcpServers"]

SYSTEM_PROMPT = """
당신은 PPT 초안(JSON) 작성 어시스턴트입니다.

작업 순서
1. 사용자가 PDF파일을 올려 PPT로 만들고자하면, → list_pdfs & chunk_pdf 로 PDF 내용을 확보.
2. 사용자에게 슬라이드 제목(목차) 리스트를 요청.
3. 목차가 주어지면, 사용자의 PDF 내용을 바탕으로 초안을 생성후 저장, 경로를 사용자에게 전달.
4. 이후 수정·보강 요청 시 JSON을 업데이트해 save_outline_json 재호출.
   근거·데이터가 추가적으로 필요하다고 판단되면, sequential thinking tool 및 exa search tool들을 사용해 심층 자료조사를 진행한다.
   자료조사로 얻은 결과는 slide 요소에 추가(표, bulletList 등)한 뒤 다시 저장한다, 
   자료조사를 통해 추가한 요소에는 exa search를 통해 얻은참고 urls와 신뢰도 점수를 반드시 추가한다.

응답은 기본적으로 한국어를 사용하되, 사용자가 바꾸면 따라간다.
"""

async def build_agent():
    servers = load_servers()
    tools   = await MultiServerMCPClient(servers).get_tools()
    llm     = ChatOpenAI(model="gpt-4o", max_tokens=8000)

    # ⬇️ here: 위치 인수(모델, 툴) + checkpointer
    agent = create_react_agent(
        llm,                       # ← 키워드 제거
        tools,                     # ← 두 번째 위치 인수
        checkpointer=checkpointer,  # ← 키워드 OK
        prompt=SYSTEM_PROMPT
    )
    print("로드된 툴:", [t.name for t in tools])
    return agent

async def main():
    agent = await build_agent()
    handler = ToolPrint()
    print("💬 대화를 시작하세요 (exit 입력 시 종료)")

    while True:
        q = input("👤 ")
        if q.lower() == "exit":
            break
        res = await agent.ainvoke(
            {"messages": [HumanMessage(content=q)]},
            config={"callbacks": [handler], "configurable": {"thread_id": SESSION_ID}},
        )
        print("\n🤖", res["messages"][-1].content, "\n")

if __name__ == "__main__":
    asyncio.run(main())
