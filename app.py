import os
import re
from urllib.parse import urlparse  # kept in case you want later URL logic

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
st.set_page_config(page_title="LLM Topic Brand Visibility", layout="wide")
st.title("LLM Topic Visibility (Brand Mentions)")

st.markdown(
    """
This tool:

1. Takes up to **500 queries**  
2. Lets you specify a **brand name** (e.g. `RAC`, `AA`, `Green Flag`)  
3. Uses an OpenAI model with **web search** to answer each query  
4. Detects whether / how often the brand is **mentioned in the answer text**  
5. Optionally clusters queries into topics (HDBSCAN + embeddings)  
6. Computes **per-topic brand visibility** (what % of queries mention the brand)
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

if clustering_mode == "Always run clustering in this app (overwrite labels)" and hdbscan is None:
    st.sidebar.error("hdbscan is not installed. Run `pip install hdbscan` to enable clustering.")

st.sidebar.markdown("---")
st.sidebar.markdown("**500 queries with web search may incur usage charges.**")

# -----------------------------
# Inputs
# -----------------------------
uploaded = st.file_uploader("Upload CSV of queries", type=["csv"])
brand = st.text_input("Tracked brand (e.g. RAC, AA, Green Flag)").strip()

run_button = st.button("Run analysis")

# -----------------------------
# OpenAI call + parsing
# -----------------------------
def call_openai_with_web_search(query: str, model: str):
    """
    Calls the OpenAI Responses API with web_search enabled.

    Returns:
        answer_text: str
    """
    try:
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=query,
        )
    except Exception as e:
        st.error(f"OpenAI error for query: {query[:80]}...\n{e}")
        return ""

    # Find the assistant message object
    message_item = next(
        (o for o in resp.output if getattr(o, "type", "") == "message"),
        None,
    )
    if not message_item or not getattr(message_item, "content", None):
        return ""

    content0 = message_item.content[0]
    answer_text = getattr(content0, "text", "") or ""
    return answer_text


def detect_brand_mentions(answer_text: str, brand: str):
    """
    Detect brand mentions in the model's answer.

    Returns:
        brand_mentioned (bool)
        mention_count (int)
        first_mention_pos (int | None)  # character index
    """
    if not brand or not answer_text:
        return False, 0, None

    answer_lower = answer_text.lower()
    brand_lower = brand.lower().strip()

    # Simple regex search – escapes brand and does a basic substring match.
    # For single-word brands like 'rac' this works well.
    pattern = re.escape(brand_lower)

    matches = list(re.finditer(pattern, answer_lower))
    if not matches:
        return False, 0, None

    brand_mentioned = True
    mention_count = len(matches)
    first_mention_pos = matches[0].start()

    return brand_mentioned, mention_count, first_mention_pos

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
# Topic visibility
# -----------------------------
def build_topic_aggregations(df: pd.DataFrame):
    """
    Build topic-level brand visibility summary.

    Assumes df has columns:
        - query
        - cluster_label
        - brand_mentioned (bool)
        - brand_mention_count (int)
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

    # Average mention count per query in topic (including zeros)
    topic_mentions_avg = (
        df.groupby("cluster_label")["brand_mention_count"]
        .mean()
        .reset_index()
        .rename(columns={"brand_mention_count": "avg_mentions_per_query"})
    )

    topic_summary = topic_summary.merge(
        topic_mentions_avg, on="cluster_label", how="left"
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
        st.error("Enter a brand to track.")
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
        df = run_clustering(df, embedding_model)
        st.success("Clustering completed.")
    else:
        st.info("Using existing 'cluster_label' column (no re-clustering).")

    # Prepare columns
    df["brand_mentioned"] = False
    df["brand_mention_count"] = 0
    df["first_mention_pos"] = np.nan
    # Optional: keep answers if you want to debug / sample them later
    df["answer_text"] = ""

    progress = st.progress(0)
    status_text = st.empty()

    # OpenAI calls
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        q = str(row["query"])

        status_text.text(f"Processing {i}/{len(df)}: {q[:80]}...")

        answer_text = call_openai_with_web_search(q, chat_model)
        brand_mentioned, mention_count, first_pos = detect_brand_mentions(
            answer_text, brand
        )

        df.at[idx, "answer_text"] = answer_text
        df.at[idx, "brand_mentioned"] = brand_mentioned
        df.at[idx, "brand_mention_count"] = mention_count
        df.at[idx, "first_mention_pos"] = first_pos

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
                "brand_mention_count",
                "first_mention_pos",
            ]
        ],
        use_container_width=True,
    )

    # Optionally show only queries where brand is mentioned
    st.markdown("### Queries where brand was mentioned")
    mentioned_df = df[df["brand_mentioned"]].copy()
    if mentioned_df.empty:
        st.info("The brand was not mentioned in any answers.")
    else:
        st.dataframe(
            mentioned_df[
                [
                    "query",
                    "cluster_label",
                    "brand_mention_count",
                    "first_mention_pos",
                ]
            ],
            use_container_width=True,
        )

    # Downloads
    st.markdown("### Download results")
    st.download_button(
        "Download per-query CSV",
        df.to_csv(index=False).encode("utf-8"),
        "llm_brand_visibility_queries.csv",
        "text/csv",
    )
    st.download_button(
        "Download topic summary CSV",
        topic_summary.to_csv(index=False).encode("utf-8"),
        "llm_brand_visibility_topics.csv",
        "text/csv",
    )



