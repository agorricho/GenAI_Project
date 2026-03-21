"""
state.py — Shared state TypedDict for the M3 LangGraph pipeline.

Every node receives and returns a ResearchState dict.
LangGraph merges returned keys back into the running state.
"""

from typing import TypedDict, List, Optional


class ResearchState(TypedDict):
    original_query:     str
    rephrased_query:    str
    retrieved_chunks:   List[dict]   # {text, title, authors, year, score}
    extracted_findings: List[str]    # one string per relevant chunk
    final_answer:       str
    citations:          List[dict]   # [{title, authors, year}]
    error:              Optional[str]
