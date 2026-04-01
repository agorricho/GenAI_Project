import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Walk up from this file to find the nearest .env (same pattern as pipeline agents)
for _p in Path(__file__).resolve().parents:
    if (_p / ".env").exists():
        load_dotenv(_p / ".env")
        break

from src.pipeline import run_query

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

st.set_page_config(page_title="CAIS Demo Prototype", page_icon="🔬", layout="wide")

QDRANT_URL = os.getenv("QDRANT_URL", "")
OPENALEX_URL = "https://api.openalex.org/works"


def check_qdrant(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(f"{url}/collections", timeout=5)
        response.raise_for_status()
        data = response.json()
        return {"ok": True, "collections": data.get("result", {}).get("collections", [])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(show_spinner=True)
def search_openalex(topic: str, per_page: int = 10) -> List[Dict[str, Any]]:
    params = {
        "search": topic,
        "per-page": per_page,
    }
    response = requests.get(OPENALEX_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json().get("results", [])



def extract_abstract_text(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    try:
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words)
    except Exception:
        return ""



def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in r.get("authorships", [])[:4]
            if a.get("author", {}).get("display_name")
        )
        rows.append(
            {
                "title": r.get("display_name", ""),
                "year": r.get("publication_year", ""),
                "authors": authors,
                "doi": r.get("doi", ""),
                "cited_by_count": r.get("cited_by_count", 0),
                "abstract": extract_abstract_text(r.get("abstract_inverted_index")),
            }
        )
    return pd.DataFrame(rows)



def rule_based_framework_row(row: pd.Series) -> Dict[str, str]:
    abstract = row.get("abstract", "") or ""
    title = row.get("title", "") or ""
    year = row.get("year", "") or ""
    authors = row.get("authors", "") or ""
    doi = row.get("doi", "") or ""

    citation = " | ".join(
        part for part in [authors if authors else "Authors unavailable", str(year) if year else "Year unavailable", title, doi] if part
    )

    aims = abstract[:180] + ("..." if len(abstract) > 180 else "") if abstract else "Needs extraction from abstract"
    data = "Look for sample, dataset, participants, or source in abstract"
    analysis = "Look for the variable/intervention and outcome being studied"
    methods = "Look for study design, model, or statistical method"
    findings = "Look for effect, result, significance, or conclusion"

    return {
        "Citation": citation,
        "Aims": aims,
        "Data": data,
        "Analysis": analysis,
        "Methods": methods,
        "Findings": findings,
    }


st.title("🔬 CAIS Demo Prototype")
st.caption("A simple Streamlit interface for your AI Scientist project")

with st.sidebar:
    st.header("Project Controls")
    qdrant_url = st.text_input("Qdrant URL", value=QDRANT_URL)
    topic = st.text_input("Research topic", value="stroke rehabilitation")
    per_page = st.slider("OpenAlex results", 5, 25, 10)
    collection_name = st.text_input("Qdrant collection", value="papers")
    run_search = st.button("Search OpenAlex", type="primary")

status = check_qdrant(qdrant_url)
if status["ok"]:
    st.success(f"Qdrant connected at {qdrant_url}")
else:
    st.warning("Qdrant not connected yet. The interface still works for OpenAlex search.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Qdrant Status", "Connected" if status["ok"] else "Offline")
with col2:
    st.metric("Collections", len(status.get("collections", [])) if status["ok"] else 0)
with col3:
    st.metric("Current Topic", topic)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Paper Search",
    "Research Framework",
    "Qdrant",
    "Chat",
])

with tab1:
    st.subheader("What this interface is for")
    st.markdown(
        """
This app gives your group a clean front door for the project.

**Flow:**
1. Enter a research topic
2. Pull papers from OpenAlex
3. Review abstracts in one place
4. Convert papers into a Research Framework table
5. Later, connect the real extraction pipeline and vector search

Right now, this is a practical UI shell that fits the repo you uploaded. It does not pretend the backend is more finished than it is.
        """
    )

    st.subheader("Suggested next backend milestones")
    st.markdown(
        """
- Replace the rule-based framework output with your real prompt-based extractor
- Add paper embedding and Qdrant upsert
- Add semantic search against stored papers
- Add downloadable CSV output
        """
    )

if run_search:
    try:
        raw_results = search_openalex(topic, per_page=per_page)
        st.session_state["raw_results"] = raw_results
        st.session_state["papers_df"] = results_to_dataframe(raw_results)
    except Exception as e:
        st.error(f"OpenAlex search failed: {e}")

papers_df = st.session_state.get("papers_df")

with tab2:
    st.subheader("OpenAlex Paper Search")
    if papers_df is None or papers_df.empty:
        st.info("Use the sidebar to search for a topic.")
    else:
        st.dataframe(papers_df[["title", "year", "authors", "cited_by_count"]], use_container_width=True)

        selected_index = st.number_input(
            "Select a paper row to inspect",
            min_value=0,
            max_value=max(len(papers_df) - 1, 0),
            value=0,
            step=1,
        )
        row = papers_df.iloc[int(selected_index)]
        st.markdown(f"### {row['title']}")
        st.write(f"**Authors:** {row['authors'] or 'Unavailable'}")
        st.write(f"**Year:** {row['year']}")
        st.write(f"**DOI:** {row['doi'] or 'Unavailable'}")
        st.write("**Abstract:**")
        st.write(row["abstract"] or "Abstract unavailable from OpenAlex for this result.")

with tab3:
    st.subheader("Research Framework Output")
    if papers_df is None or papers_df.empty:
        st.info("Search for papers first.")
    else:
        framework_rows = [rule_based_framework_row(row) for _, row in papers_df.iterrows()]
        framework_df = pd.DataFrame(framework_rows)
        st.dataframe(framework_df, use_container_width=True)

        csv_data = framework_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download framework CSV",
            data=csv_data,
            file_name="research_framework_output.csv",
            mime="text/csv",
        )

        st.caption("This tab uses a rule-based placeholder. For LLM-powered extraction via the full pipeline, use the Chat tab.")

with tab4:
    st.subheader("Qdrant Status")
    if status["ok"]:
        st.json(status)
        st.caption("Next step: wire embeddings + upsert into the selected collection.")
    else:
        st.error(status.get("error", "Unable to connect to Qdrant."))
        st.code("docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")

    st.subheader("Example upsert stub")
    st.code(
        f'''from qdrant_client import QdrantClient\n\nclient = QdrantClient(url="{qdrant_url}")\n# create collection and upsert vectors here''',
        language="python",
    )

with tab5:
    st.subheader("Research Chat")
    st.caption(
        "Ask a research question — the full LangGraph pipeline runs: "
        "Rephraser → Retriever → Extractor → Synthesizer."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    user_question = st.chat_input("Ask a research question...")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

        with st.spinner("Running pipeline: Rephraser → Retriever → Extractor → Synthesizer..."):
            try:
                result = run_query(user_question)
                answer    = result["answer"]
                citations = result["citations"]
                error     = None
            except Exception as exc:
                answer    = None
                citations = []
                error     = str(exc)

        if error:
            st.error(f"Pipeline error: {error}")
            st.session_state.messages.append(AIMessage(f"Error: {error}"))
        else:
            with st.chat_message("assistant"):
                st.markdown(answer)
                if citations:
                    st.markdown("**Citations:**")
                    for c in citations:
                        authors = c.get("authors", "Unknown")
                        year    = c.get("year", "n.d.")
                        title   = c.get("title", "")
                        st.markdown(f"- {authors} ({year}). *{title}*")
            st.session_state.messages.append(AIMessage(answer))
