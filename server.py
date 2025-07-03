# ppt_server.py – FastMCP server exposing PDF→PPT tools
# ======================================================
"""
▶ RUN :  python ppt_server.py | mcp 서버로 stdio transport (client.py 에 정의된 mcp.json 과 일치해야 함)

필요 패키지
-----------
$ pip install pdfplumber langchain-openai mcp fastmcp python-dotenv

환경 변수
-----------
OPENAI_API_KEY 가 필요합니다.
"""
import io, json, textwrap, os
from typing import List, Dict, Any

import pdfplumber
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI

load_dotenv()

mcp = FastMCP("PPT")

SCHEMA = textwrap.dedent("""
    {
      "schemaVersion": "1.1.0",
      "docMeta": {
        "titleKo": string,
        "titleEn": string,
        "created": string(YYYY-MM-DD),
        "owner": string,
        "language": [string]
      },
      "slides": [
        {
          "slideDescription": string,
          "elements": [
            {"type": "title" | "subtitle" | "text" | "quote" | "metric" | "bulletList" | "table" | "logo" | "date", "content": any, "references"?: [{"source": string, "snippet": string} ] }
          ]
        }
      ]
    }
""")

MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool(name="ppt.extract_text")
def extract_text(pdf_path: str) -> str:
    """PDF 경로를 받아 텍스트(전체)를 반환합니다."""
    return extract_text_from_pdf(pdf_path)


@mcp.tool(name="ppt.draft_ppt_json")
def draft_ppt_json(pdf_text: str, toc: List[str]) -> Dict[str, Any]:
    """PDF 텍스트와 목차(list[str])를 기반으로 PPT preview JSON 초안을 생성합니다."""
    prompt = f"""
You are a presentation planner. Produce a JSON strictly following the schema below.
1. Cover slide + each TOC item should map to one slide.
2. Pull supportive sentences from the PDF. Add them into each element's `references` array with dummy `source:"PDF"`.
3. If context is insufficient, mark that element's content as "NEED_RESEARCH".

[Schema]
{SCHEMA}

[TOC]
- {'\\n- '.join(toc)}

[PDF]
{pdf_text[:6000]}

Return *ONLY* the JSON.
""".strip()

    response = MODEL.invoke(prompt)

    # 모델 출력이 ```json … ``` 블록일 수 있으므로 파싱 보정
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        cleaned = (
            response.content.strip()
            .removeprefix("```json").removeprefix("```")  # 앞쪽 ```json 또는 ``` 제거
            .removesuffix("```")                          # 뒤쪽 ``` 제거
            .strip()
        )
        return json.loads(cleaned)



@mcp.tool(name="ppt.update_ppt_json")
def update_ppt_json(ppt_json: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """사용자 피드백을 반영해 PPT JSON 을 수정합니다."""
    prompt = f"""
    You are editing an existing PPT JSON. Follow the rules:
    • Incorporate the feedback.
    • Preserve untouched structure.
    • Maintain schema integrity: {SCHEMA[:120]}...
    • For added factual content, append a `references` entry with `source:"USER"`.

    [Current JSON]\n```json\n{json.dumps(ppt_json, ensure_ascii=False, indent=2)}\n```

    [Feedback]\n{feedback}

    Provide the *entire* updated JSON only.
    """
    response = MODEL.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        cleaned = response.content.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)


if __name__ == "__main__":
    # stdio transport → client.py 에서 stdio 로 연결
    mcp.run(transport="stdio")