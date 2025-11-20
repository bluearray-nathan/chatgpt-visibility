import os
from urllib.parse import urlparse  # still used for citations if needed later

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# -----------------------------
# OpenAI client helper
# -----------------------------
def get_openai_client():
    """
    Load API key from Streamlit secrets.
    """
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("❌ OPENAI_API_KEY is missing in `.streamlit/secrets.toml`.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


client = get_openai_client()

# -----------------------------
# Try to import hdbscan
# -----------------------------
try:
    import hdbscan
except ImportError:
    hdbscan = None

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="LLM Topic Visibility (Brand Mentions)", layout="wide")
st.title("LLM Topic Visibility – Brand Mentions")

st.markdown(
    """
This tool:

1. Takes up to **500 queries** and a **tracked brand** (e.g. `\"AA\"`, `\"RAC\"`, `\"Green Flag\"`)  
2. Uses an OpenAI model with **web search** to answer each query  
3. Detects whether the **brand is mentioned in the answer text**, and where  
4. Optionally clusters queries using HDBSCAN + embeddings  
5. Computes **per-topic brand visibility** based on mentions
"""
)

# Sidebar config
st.sidebar.header("Model settings")
chat_model = st.sidebar.text_input("Chat model", value="gpt-4.1-mini")
embedding_model = st.sidebar.text_input(
    "Embedding model",
    value="text-embedding-3-small",
)

max_queries = st.sidebar.number_input(
    "Max queries to process",
    min_value=1,
    max_value=500,
    value=100,
    step=1,
)

clustering_mode = st.sidebar.radio(
    "Clustering mode",
    options=[
        "Use existing 'cluster_label' if present (no re-clustering)",
        "Always run clustering in this app (overwrite labels)",
    ],
)

# Slider for HDBSCAN min_cluster_size
min_cluster_size = st.sidebar.slider(
    "HDBSCAN min_cluster_size",
    min_value=2,
    max_value=50,
    value=8,
    step=1,
)

if clustering_mode == "Always run clustering in this app (overwrite labels)" and hdbscan is None:
    st.sidebar.error("hdbscan is not installed. Run `pip install hdbscan` to enable clustering.")

st.sidebar.markdown("---")
st.sidebar.markdown("**500 queries with web search may incur usage charges.**")

# -----------------------------
# Inputs
# -----------------------------
uploaded = st.file_uploader("Upload CSV of queries (must contain a 'query' column)", type=["csv"])
brand = st.text_input("Tracked brand (e.g. AA, RAC, Green Flag)").strip()
run_button = st.button("Run analysis")

# -----------------------------
# OpenAI call + parsing
# -----------------------------
def call_openai_with_web_search(query: str, model: str):
    """
    Calls the OpenAI Responses API with web_search enabled.

    Returns:
        answer_text: str
        urls: list[str] - URL citations in order of appearance (for context)
    """
    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=query,
        )
    except Exception as e:
        st.error(f"OpenAI error for query: {query[:80]}...\n{e}")
        return "", []

    # Find the assistant message object in the output
    message_item = next(
        (o for o in resp.output if getattr(o, "type", "") == "message"),
        None,
    )
    if not message_item or not getattr(message_item, "content", None):
        return "", []

    content0 = message_item.content[0]
    answer_text = getattr(content0, "text", "") or ""

    # Extract URL citations from annotations (if any)
    urls = []
    annotations = getattr(content0, "annotations", []) or []
    for ann in annotations:
        if getattr(ann, "type", "") == "url_citation":
            url = getattr(ann, "url", None)
            if url:
                urls.append(url)

    return answer_text, urls


def detect_brand_mention(answer_text: str, brand: str):
    """
    Detect whether a brand is mentioned in the answer text (case-insensitive).

    Returns:
        brand_mentioned: bool
        first_position: int | None   # 0-based character index of the first mention
    """
    if not brand:
        return False, None

    brand_lower = brand.lower()
    text_lower = (answer_text or "").lower()

    idx = text_lower.find(brand_lower)
    if idx == -1:
        return False, None

    return True, idx

# -----------------------------
# Clustering
# -----------------------------
def run_clustering(df: pd.DataFrame, embedding_model: str, min_cluster_size: int = 8):
    """
    Perform HDBSCAN clustering over query embeddings.

    Returns:
        df with a 'cluster_label' column (string labels).
    """
    if hdbscan is None:
        st.error("hdbscan is not installed. Run `pip install hdbscan`.")
        st.stop()

    queries = df["query"].astype(str).tolist()

    st.info("Embedding queries for clustering...")
    emb_resp = client.embeddings.create(
        model=embedding_model,
        input=queries,
    )
    vectors = np.array([item.embedding for item in emb_resp.data])

    st.info("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(vectors)

    # Fallback: if EVERYTHING is noise (-1), force a single cluster_0
    if np.all(labels == -1):
        labels = np.zeros_like(labels)

    # Convert numeric labels (incl. -1) to strings
    str_labels = []
    for label in labels:
        if label == -1:
            str_labels.append("noise")
        else:
            str_labels.append(f"cluster_{label}")

    df["cluster_label"] = str_labels
    return df

# -----------------------------
# Topic visibility (brand-based)
# -----------------------------
def build_topic_aggregations(df: pd.DataFrame):
    """
    Build topic-level aggregation for brand visibility.

    Assumes df has columns:
        - query
        - cluster_label
        - brand_mentioned (bool)
        - brand_position_chars (float/NaN)
    """
    # Topic sizes (how many queries in each topic)
    topic_sizes = (
        df.groupby("cluster_label")["query"]
        .nunique()
        .rename("topic_query_count")
        .reset_index()
    )

    # How many queries per topic where brand was mentioned at all
    topic_brand_mentioned = (
        df[df["brand_mentioned"]]
        .groupby("cluster_label")["query"]
        .nunique()
        .rename("queries_with_brand")
        .reset_index()
    )

    topic_summary = topic_sizes.merge(
        topic_brand_mentioned, on="cluster_label", how="left"
    )
    topic_summary["queries_with_brand"] = (
        topic_summary["queries_with_brand"].fillna(0).astype(int)
    )
    topic_summary["visibility_pct"] = (
        topic_summary["queries_with_brand"]
        / topic_summary["topic_query_count"]
        * 100.0
    )

    # Average position of first brand mention (lower = earlier in answer), only where mentioned
    topic_brand_pos = (
        df[df["brand_mentioned"]]
        .groupby("cluster_label")["brand_position_chars"]
        .mean()
        .rename("avg_first_mention_char")
        .reset_index()
    )

    topic_summary = topic_summary.merge(
        topic_brand_pos, on="cluster_label", how="left"
    )

    return topic_summary

# -----------------------------
# MAIN RUN
# -----------------------------
if run_button:
    if uploaded is None:
        st.error("Upload a CSV first.")
        st.stop()

    if not brand:
        st.error("Enter a tracked brand.")
        st.stop()

    df = pd.read_csv(uploaded)
    if "query" not in df.columns:
        st.error("CSV must contain a 'query' column.")
        st.stop()

    df = df.head(max_queries).copy()
    st.write(f"Loaded **{len(df)}** queries")

    # Clustering
    if (
        clustering_mode == "Always run clustering in this app (overwrite labels)"
        or "cluster_label" not in df.columns
    ):
        df = run_clustering(df, embedding_model, min_cluster_size=min_cluster_size)
        st.success("Clustering completed.")
    else:
        st.info("Using existing 'cluster_label' column (no re-clustering).")

    # Prepare columns
    df["brand_mentioned"] = False
    df["brand_position_chars"] = np.nan
    df["all_citations"] = ""  # still useful for debugging / context

    progress = st.progress(0)
    status_text = st.empty()

    # OpenAI calls
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        q = str(row["query"])

        status_text.text(f"Processing {i}/{len(df)}: {q[:80]}...")

        answer_text, urls = call_openai_with_web_search(q, chat_model)
        mentioned, pos = detect_brand_mention(answer_text, brand)

        df.at[idx, "brand_mentioned"] = mentioned
        df.at[idx, "brand_position_chars"] = pos if pos is not None else np.nan
        df.at[idx, "all_citations"] = ";".join(urls)

        progress.progress(i / len(df))

    st.success("OpenAI queries completed.")

    # Aggregations
    topic_summary = build_topic_aggregations(df)

    # -------------------------
    # Display results
    # -------------------------
    st.markdown(f"## Topic-level visibility for brand: `{brand}`")
    if topic_summary.empty:
        st.info("No topics found.")
    else:
        st.dataframe(
            topic_summary.sort_values("visibility_pct", ascending=False),
            use_container_width=True,
        )

    st.markdown("## Query-level results")
    st.dataframe(
        df[
            [
                "query",
                "cluster_label",
                "brand_mentioned",
                "brand_position_chars",
                "all_citations",
            ]
        ],
        use_container_width=True,
    )

    # Downloads
    st.markdown("### Download results")
    st.download_button(
        "Download per-query CSV",
        df.to_csv(index=False).encode("utf-8"),
        "llm_visibility_brand_queries.csv",
        "text/csv",
    )
    st.download_button(
        "Download topic summary CSV",
        topic_summary.to_csv(index=False).encode("utf-8"),
        "llm_visibility_brand_topics.csv",
        "text/csv",
    )





