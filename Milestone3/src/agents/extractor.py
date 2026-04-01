"""
Node 3 — Extractor

Input:  state["retrieved_chunks"], state["original_query"]
Output: state["extracted_findings"]  (one string per relevant chunk)

For each retrieved chunk, asks the LLM to extract findings relevant to the
original query. Chunks where the LLM responds with "NOT RELEVANT" are dropped.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ── Load credentials ──────────────────────────────────────────────────────────
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY  = os.getenv("0LLAMA", "")
MODEL           = os.getenv("M3_MODEL", "llama3.2")

# ── LLM ──────────────────────────────────────────────────────────────────────
_client_kwargs = (
    {"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}}
    if OLLAMA_API_KEY else {}
)

_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL,
    temperature=0,
    client_kwargs=_client_kwargs,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
_PROMPT = (
    "You are a research analyst extracting key findings from academic abstracts.\n\n"
    "Research question: {query}\n\n"
    "Paper abstract:\n{text}\n\n"
    "Instructions:\n"
    "- If the abstract contains findings relevant to the research question, "
    "summarise the key finding in 1-3 sentences. Be specific: include "
    "direction of effect, statistical significance, and study context if stated.\n"
    "- If the excerpt is NOT relevant to the research question, respond with "
    "exactly: NOT RELEVANT\n\n"
    "Your response:"
)


# ── Node function ─────────────────────────────────────────────────────────────
def extractor_node(state: dict) -> dict:
    query  = state["original_query"]
    chunks = state["retrieved_chunks"]
    print(f"[Extractor] Extracting findings from {len(chunks)} chunks...")

    findings = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if not text:
            print(f"  [Extractor] Chunk {i+1}: empty text — skipping")
            continue

        prompt = _PROMPT.format(query=query, text=text)
        response = _llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        if result.upper().startswith("NOT RELEVANT"):
            print(f"  [Extractor] Chunk {i+1} ({chunk['title'][:40]}): NOT RELEVANT")
        else:
            print(f"  [Extractor] Chunk {i+1} ({chunk['title'][:40]}): finding extracted")
            findings.append(result)

    print(f"[Extractor] {len(findings)} relevant findings extracted")
    return {"extracted_findings": findings}
