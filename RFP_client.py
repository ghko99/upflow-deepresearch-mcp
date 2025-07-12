from typing import List, Dict, Any
from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import json
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

STRICT_SYSTEM_PROMPT = """
**목표:** 입력 전문을 분석해 _오로지_ 아래 “Target JSON Schema”와 **완전히 같은 구조·키·배열 순서**로 된 JSON 한 덩어리만을 반환합니다.

──────────────── Target JSON Schema - BEGIN ────────────────
{
  "schemaVersion": "1.1.0",
  "docMeta": {
    "titleKo": (string),
    "titleEn": (string),
    "created": (YYYY-MM-DD),
    "owner": (string),
    "language": ["ko"]
  },
  "slides": [
    {
      "slideDescription": (string, 15자 이내),
      "elements": [
        { "type": "title",      "content": (string) },
        { "type": "subtitle",   "content": (string) },
        { "type": "date",       "content": (YYYY-MM-DD) },
        { "type": "logo",       "content": (string) },
        { "type": "table", "content": {"description":"…","tableType":"matrix|segment|comparison|highlight","highlightColumn":"…","data":[[…],…]}} }
      ]
    }
  ]
}
──────────────── Target JSON Schema -  END  ────────────────
각 elements는 여러개의 {"type":"..." , "content":"..."}로 구성이 되어야 합니다.
table type이 존재하는 slide는 반드시 존재해야하며 table type의 element는 아래 형식을 유지시켜 주세요.
{"type":"table", "content" : {"description":"…","tableType":"matrix|segment|comparison|highlight","highlightColumn":"…","data":[[…],…]}} }
"""

USER_PROMPT_TMPL = """
당신은 'AI 제안서 → PPT JSON' 변환 전담 엔지니어입니다.
표와 제안서 내용을 바탕으로 다음 내용으로 구성된 PPT JSON을 만들어주세요.

### INPUT
{doc_text}

위 제안서를 Target JSON Schema 형식으로 변환하십시오. 목차는 다음순서를 반드시 유지해야하며 생략이 없어야 합니다.
{list_of_contents}

**출력 규칙 (엄격)**  
1. 순수 JSON 하나만 반환 - 코드블록·주석·설명 금지.  
2. 필드 추가·삭제·재배열 **불가**.  
3. elements[*].type 은 "title"|"subtitle"|"text"|"bulletList"|"table"|"metric"|"quote"|"logo"|"date" 중 하나.    
4. 제안서를 바탕으로 elements를 최대한 많이 채워넣어야 하며, table과 수치 데이터를 적극 활용해 길게 작성 합니다.
5. 만약 목차 구성이 올바르지 않은 형태라 판단되면 다시 되물어 주세요.
"""
SUPPLEMENT_PROMPT_TMPL = """
아래는 현재까지 작성된 PPT JSON 입니다.

{existing_json}

### 추가 보강 지시
제안서 전체 내용을 다시 참고해 slides의 elements들의 내용에 table data들을 넣어 **더 풍부하게** 보강해주세요.
slide들이 다음과 같이 올바른 구조로 되어있는지 한번더 검토하고 이에 맞게 수정하세요.
{list_of_contents}
출력 규칙은 이전과 동일하며, **순수 JSON 하나만 반환**해야 합니다.
"""

def _call_gpt(messages: List[Dict[str, str]], model : str) -> Dict[str, Any]:
    """GPT-4o-mini(JSON-mode) 호출 래퍼"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


def chunk_pdf(
    pdf_path: str,
    mode: str = "page",          # page | single
) -> List[str]:
    """
    Load *pdf_path* with PyMuPDF4LLMLoader and return text chunks.
    """
    pdf = Path(pdf_path).expanduser()
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    loader = PyMuPDF4LLMLoader(str(pdf), mode=mode)
    docs   = loader.load()                       # → List[Document]
    pages = [doc.page_content for doc in docs]
    return '\n'.join(pages)


def generate_slide_json_refine(doc_text: str, list_of_contents : str , model : str , rounds: int = 3) -> Dict[str, Any]:
    """제안서 전문을 그대로 사용해 GPT-4o-mini를 `rounds`만큼 호출, JSON을 점진적으로 보강"""

    # 1st pass – 최초 생성
    first_messages = [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TMPL.format(doc_text=doc_text,
                                                              list_of_contents = list_of_contents)},
    ]
    
    slides_json = _call_gpt(first_messages, model=model)
    print("✅ 1/{} pass 완료".format(rounds))

    # Subsequent passes – 보강
    for r in range(2, rounds + 1):
        supplement_messages = [
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SUPPLEMENT_PROMPT_TMPL.format(
                    existing_json=json.dumps(slides_json, ensure_ascii=False),
                    list_of_contents = list_of_contents
                ),
            },
        ]
        slides_json = _call_gpt(supplement_messages, model=model)
        print(f"✅ {r}/{rounds} pass 완료")
        with open("slides.json", "w", encoding="utf-8") as fp:
            json.dump(slides_json, fp, ensure_ascii=False, indent=2)
    return slides_json



def get_ppt_preview(doc_text : str, list_of_contents : str, model : str = "gpt-4o-mini", rounds : int = 3):
    return generate_slide_json_refine(doc_text=doc_text, 
                                      list_of_contents=list_of_contents, model = model, rounds=rounds)



if __name__ == "__main__":

    ## file path : 25년 삼성전자
    file_path = './docs/25년 삼성전자 MX 미국 직영 매장 PMO_입찰공고문_F.pdf'
    pdf_contents = chunk_pdf(file_path)

    #user 요청 결과
    list_of_contents = '프로젝트 배경, 운영 세부 내용, 입찰 일정'
    ppt_preview = get_ppt_preview(pdf_contents, list_of_contents, rounds=1)
    with open("slides.json", "w", encoding="utf-8") as fp:
        json.dump(ppt_preview, fp, ensure_ascii=False, indent=2)
    