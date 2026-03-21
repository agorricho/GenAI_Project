"""
Node 4 — Synthesizer

Input:  state["extracted_findings"], state["retrieved_chunks"]
Output: state["final_answer"], state["citations"]

Combines extracted findings into a coherent answer and builds a numbered
citation list from the source chunks.
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
    "You are a research synthesis expert. Given the following extracted "
    "findings from academic papers, write a concise, evidence-based answer "
    "to the research question.\n\n"
    "Research question: {query}\n\n"
    "Extracted findings:\n{findings}\n\n"
    "Instructions:\n"
    "- Synthesise the findings into a coherent 2-4 paragraph answer\n"
    "- Reference specific findings where possible\n"
    "- Note any contradictions or gaps in the evidence\n"
    "- Write in academic prose\n\n"
    "Answer:"
)


def _deduplicate_citations(chunks: list[dict]) -> list[dict]:
    """Return unique citations by title (preserving order)."""
    seen = set()
    citations = []
    for c in chunks:
        title = c.get("title", "Unknown")
        if title not in seen:
            seen.add(title)
            citations.append({
                "title":   title,
                "authors": c.get("authors", ""),
                "year":    c.get("year", ""),
            })
    return citations


# ── Node function ─────────────────────────────────────────────────────────────
def synthesizer_node(state: dict) -> dict:
    query    = state["original_query"]
    findings = state["extracted_findings"]
    chunks   = state["retrieved_chunks"]
    print(f"[Synthesizer] Synthesising {len(findings)} findings into final answer...")

    if not findings:
        answer = (
            "No relevant findings were extracted from the retrieved documents. "
            "The corpus may not contain papers directly addressing this question."
        )
    else:
        numbered = "\n\n".join(
            f"[{i+1}] {f}" for i, f in enumerate(findings)
        )
        prompt   = _PROMPT.format(query=query, findings=numbered)
        response = _llm.invoke([HumanMessage(content=prompt)])
        answer   = response.content.strip()

    citations = _deduplicate_citations(chunks)
    print(f"[Synthesizer] Answer ready. {len(citations)} citations.")
    return {"final_answer": answer, "citations": citations}
