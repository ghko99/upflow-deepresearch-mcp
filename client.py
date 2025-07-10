# client.py – Terminal PDF→PPT Draft Chatbot (robust JSON save)
# ============================================================
"""
변경 사항
────────
* draft / update 툴이 **문자열(escaped JSON)** 을 반환하는 경우를 자동 보정 → 제대로 파싱해 dict 로 저장
* `ensure_dict()` 헬퍼 추가, `state["ppt_json"]` 저장 전에 모두 검증
"""

from __future__ import annotations

import asyncio, json, uuid, datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from rich import print
from rich.prompt import Prompt
from rich.panel import Panel
from rich.syntax import Syntax

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ---------------------------------------------------------------------------
# MCP 서버 목록 로드
# ---------------------------------------------------------------------------
SERVERS = SERVERS = {"ppt_preview": {
      "transport": "stdio",
      "command": "python",
      "args": ["./server.py"]
    },

    "exa": {
      "transport": "sse",
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "exa",
        "--key",
        "e192bc88-dbf7-4dab-9d77-51e9d405c750"
      ]
    },

    "mcp-sequentialthinking-tools": {
      "transport": "sse",
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@xinzhongyouhai/mcp-sequentialthinking-tools",
        "--key",
        "e192bc88-dbf7-4dab-9d77-51e9d405c750"
      ]
    }
}

# Draft JSON 저장 폴더
DRAFTS_DIR = Path("ppt_drafts")
DRAFTS_DIR.mkdir(exist_ok=True)

# 상태
state: Dict[str, Any] = {
    "stage": "need_pdf",      # need_pdf → need_toc → feedback
    "pdf_text": None,
    "ppt_json": None,
    "json_path": None,         # Path
}

SESSION_ID = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# 헬퍼들
# ---------------------------------------------------------------------------

def ensure_dict(maybe_json) -> Dict[str, Any]:
    """문자열(JSON) → dict 로, 이미 dict 면 그대로 반환."""
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except json.JSONDecodeError:
            # 따옴표/백틱 감싼 코드블록 제거
            cleaned = (
                maybe_json.strip()
                .removeprefix("```json").removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(cleaned)
    raise TypeError("ppt_json 은 dict 또는 JSON 문자열이어야 합니다.")


def print_json(data: Dict[str, Any]):
    pretty = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)
    print(Syntax(pretty, "json", theme="monokai", line_numbers=False))


def default_filename(prefix: str = "draft_") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{ts}.json"


def save_json_to_file(js: Dict[str, Any], path: Path | None = None) -> Path:
    if path is None:
        path = DRAFTS_DIR / default_filename()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)
    return path

# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

async def main():
    # 1) MCP 클라이언트 & 툴
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    tool_dict = {t.name: t for t in tools}
    for req in ("ppt.extract_text", "ppt.draft_ppt_json", "ppt.update_ppt_json"):
        if req not in tool_dict:
            raise RuntimeError(f"필수 MCP 툴 누락: {req}")

    print(Panel("[bold green]PDF → PPT Draft Generator[/bold green]  (type 'exit' to quit)"))

    while True:
        # ------------------------------------------------------------
        # ① PDF 입력 단계
        # ------------------------------------------------------------
        if state["stage"] == "need_pdf":
            pdf_path = Prompt.ask("🗎  PDF 파일 경로")
            if pdf_path.lower() == "exit":
                break
            try:
                text = await tool_dict["ppt.extract_text"].ainvoke({"pdf_path": pdf_path})
                state["pdf_text"] = text
                state["stage"] = "need_toc"
                print(Panel("PDF 추출 완료 – 목차를 입력해 주세요 (줄바꿈)"))
            except Exception as e:
                print(f"[red]PDF 로드 실패: {e}")

        # ------------------------------------------------------------
        # ② 목차 입력 + 초안 생성
        # ------------------------------------------------------------
        elif state["stage"] == "need_toc":
            toc_raw = Prompt.ask("📑  PPT 목차 입력 (줄바꿈). 완료 후 Enter 두 번")
            if toc_raw.lower() == "exit":
                break
            toc_list = [ln.strip() for ln in toc_raw.splitlines() if ln.strip()]
            try:
                resp = await tool_dict["ppt.draft_ppt_json"].ainvoke({
                    "pdf_text": state["pdf_text"],
                    "toc": toc_list,
                })
                state["ppt_json"] = ensure_dict(resp)
                state["json_path"] = save_json_to_file(state["ppt_json"])
                state["stage"] = "feedback"
                print(Panel(f"초안 저장 → {state['json_path'].name}. 수정 지시를 입력하거나 'exit' 로 종료"))
                print_json(state["ppt_json"])
            except Exception as e:
                print(f"[red]초안 생성 실패: {e}")

        # ------------------------------------------------------------
        # ③ 피드백 단계 (업데이트 & 재저장)
        # ------------------------------------------------------------
        elif state["stage"] == "feedback":
            fb = Prompt.ask("🛠️  수정 지시")
            if fb.lower() in {"exit", "done", "finish"}:
                break
            try:
                upd = await tool_dict["ppt.update_ppt_json"].ainvoke({
                    "ppt_json": state["ppt_json"],
                    "feedback": fb,
                })
                state["ppt_json"] = ensure_dict(upd)
                save_json_to_file(state["ppt_json"], state["json_path"])
                print(Panel(f"업데이트 완료 & 저장 → {state['json_path'].name}"))
                print_json(state["ppt_json"])
            except Exception as e:
                print(f"[red]업데이트 실패: {e}")

    print("👋  종료합니다.")


if __name__ == "__main__":
    asyncio.run(main())
