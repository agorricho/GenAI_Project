"""
pipeline.py — M3 LangGraph StateGraph

Wires the four agent nodes into a linear pipeline:

    Rephraser → Retriever → Extractor → Synthesizer → END

Public interface:  run_query(question: str) -> dict
"""

from langgraph.graph import StateGraph, END

from src.state import ResearchState
from src.agents.rephraser   import rephraser_node
from src.agents.retriever   import retriever_node
from src.agents.extractor   import extractor_node
from src.agents.synthesizer import synthesizer_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("rephraser",   rephraser_node)
    graph.add_node("retriever",   retriever_node)
    graph.add_node("extractor",   extractor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("rephraser")
    graph.add_edge("rephraser",   "retriever")
    graph.add_edge("retriever",   "extractor")
    graph.add_edge("extractor",   "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Module-level compiled graph (imported by chat_app.py and batch_query.py)
app = build_graph()


def run_query(question: str) -> dict:
    """
    Run the full 4-node pipeline for a research question.

    Returns:
        {
            "answer":    str,
            "citations": list[{title, authors, year}],
            "chunks":    list[{text, title, authors, year, score}],
        }
    """
    initial_state: ResearchState = {
        "original_query":     question,
        "rephrased_query":    "",
        "retrieved_chunks":   [],
        "extracted_findings": [],
        "final_answer":       "",
        "citations":          [],
        "error":              None,
    }

    result = app.invoke(initial_state)

    return {
        "answer":    result["final_answer"],
        "citations": result["citations"],
        "chunks":    result["retrieved_chunks"],
    }
