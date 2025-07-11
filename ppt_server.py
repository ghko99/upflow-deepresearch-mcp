#!/usr/bin/env python
"""
ppt_server.py  —  FastMCP utility tools (JSON-only, fixed PyMuPDF4LLM)

Tools
1) list_pdfs            → List[str]
2) chunk_pdf            → List[str]  ★ fixed
3) create_outline_json  → str
4) save_outline_json    → str
"""
from __future__ import annotations
import json, datetime
from pathlib import Path
from typing import List, Union

from mcp.server.fastmcp import FastMCP

# ✅ 올바른 로더
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

mcp = FastMCP("PPT-Utility-JSON")

# ------------------------------------------------------------------ #
@mcp.tool()
def list_pdfs(dirpath: str = "./docs", pattern: str = "*.pdf") -> List[str]:
    """Return matched PDF paths."""
    return sorted(str(p) for p in Path(dirpath).expanduser().glob(pattern))

# ------------------------------------------------------------------ #
@mcp.tool()
def chunk_pdf(
    pdf_path: str,
    chunk_size: int = 1500,
    overlap: int = 200,
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(docs)
    return [c.page_content for c in chunks]

# ------------------------------------------------------------------ #
@mcp.tool()
def create_outline_json(titles: List[str], owner: str = "", lang: str = "ko") -> str:
    today = datetime.date.today().isoformat()
    if not titles:
        raise ValueError("`titles` 리스트가 비어 있습니다.")

    slides = [{
        "slideDescription": "표지",
        "elements": [
            {"type": "title", "content": titles[0]},
            {"type": "subtitle", "content": owner or " "},
            {"type": "date", "content": today}
        ]
    }] + [
        {
            "slideDescription": t,
            "elements": [
                {"type": "title", "content": t},
                {"type": "text",  "content": "TODO"}
            ]
        } for t in titles[1:]
    ]

    outline = {
        "schemaVersion": "1.1.0",
        "docMeta": {
            "titleKo": titles[0],
            "created": today,
            "owner": owner,
            "language": [lang]
        },
        "slides": slides
    }
    return json.dumps(outline, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ #
@mcp.tool()
def save_outline_json(ppt_json: Union[str, dict], output_path: str = "outline.json") -> str:
    data = json.loads(ppt_json) if isinstance(ppt_json, str) else ppt_json
    if data.get("schemaVersion") != "1.1.0":
        raise ValueError("schemaVersion 1.1.0 필요")
    out = Path(output_path).with_suffix(".json").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out.resolve())

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    args = parser.parse_args()
    mcp.run(transport="streamable_http" if args.http else "stdio")
