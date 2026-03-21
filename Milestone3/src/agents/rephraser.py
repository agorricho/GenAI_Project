"""
Node 1 — Rephraser

Input:  state["original_query"]
Output: state["rephrased_query"]

Converts the user's natural-language research question into compact academic
search terms optimised for vector similarity search.
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
    "You are an academic research assistant. Convert the following research "
    "question into concise academic search terms. Focus on:\n"
    "- Key concepts and variables\n"
    "- Academic terminology\n"
    "- Alternative phrasings used in research papers\n"
    "- Remove conversational language\n\n"
    "Question: {query}\n\n"
    "Respond with ONLY the optimised search terms (no explanations, no bullet "
    "points, no preamble):"
)


# ── Node function ─────────────────────────────────────────────────────────────
def rephraser_node(state: dict) -> dict:
    print("[Rephraser] Converting query to academic search terms...")
    query = state["original_query"]
    prompt = _PROMPT.format(query=query)
    response = _llm.invoke([HumanMessage(content=prompt)])
    rephrased = response.content.strip()
    print(f"[Rephraser] '{query}' → '{rephrased}'")
    return {"rephrased_query": rephrased}
