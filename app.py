import os
from urllib.parse import urlparse

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
st.set_page_config(page_title="LLM Topic Visibility", layout="wide")
st.title("LLM Topic Visibility (Tracked Domain)")

st.markdown(
    """
This tool:

1. Takes up to **500 queries** and a **tracked domain**  
2. Uses an OpenAI model with **web search** to answer each query  
3. Detects if / where your domain is cited  
4. Optionally clusters queries using HDBSCAN + embeddings  
5. Computes **per-topic visibility** and **which URLs of your domain appear**
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
domain = st.text_input("Tracked domain (e.g. example.com)").strip().lower()

run_button = st.button("Run analysis")

# -----------------------------
# OpenAI call + parsing
# -----------------------------
def call_openai_with_web_search(query: str, model: str):
    """
    Calls the OpenAI Responses API with web_search enabled.

    Returns:
        answer_text: str
        urls: list[str] - URL citations in order of appearance
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

    # Find the assistant message object
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


def first_domain_position(urls, domain: str):
    """
    Given an ordered list of URLs and a tracked domain,
    return (domain_cited: bool, position: int | None).

    Position is 1-based index of the first URL whose hostname
    contains the tracked domain.
    """
    if not domain:
        return False, None

    for i, url in enumerate(urls, start=1):
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            host = ""
        if domain in host:
            return True, i

    return False, None

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
# Topic visibility + URLs
# -----------------------------
def build_topic_aggregations(df: pd.DataFrame, tracked_domain: str):
    """
    Build:
      - topic_summary: per-topic visibility %
      - topic_urls_tracked: which URLs from tracked domain appear per topic

    Assumes df has columns:
        - query
        - cluster_label
        - domain_cited (bool)
        - all_citations (string of ';'-joined URLs)
    """
    # Topic sizes (how many queries in each topic)
    topic_sizes = (
        df.groupby("cluster_label")["query"]
        .nunique()
        .rename("topic_query_count")
        .reset_index()
    )

    # How many queries per topic where domain was cited at all
    topic_domain_cited = (
        df[df["domain_cited"]]
        .groupby("cluster_label")["query"]
        .nunique()
        .rename("queries_with_domain")
        .reset_index()
    )

    topic_summary = topic_sizes.merge(
        topic_domain_cited, on="cluster_label", how="left"
    )
    topic_summary["queries_with_domain"] = (
        topic_summary["queries_with_domain"].fillna(0).astype(int)
    )
    topic_summary["visibility_pct"] = (
        topic_summary["queries_with_domain"]
        / topic_summary["topic_query_count"]
        * 100.0
    )

    # Explode all_citations into one URL per row, then filter to tracked domain
    df_urls = (
        df.copy()
        .assign(
            url_list=lambda d: d["all_citations"]
            .fillna("")
            .astype(str)
            .str.split(";")
        )
        .explode("url_list")
    )

    df_urls["url"] = df_urls["url_list"].str.strip()
    df_urls = df_urls.drop(columns=["url_list"])
    df_urls = df_urls[df_urls["url"] != ""]  # drop blanks

    def is_tracked(url: str, domain: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        return domain in host

    df_urls_tracked = df_urls[
        df_urls["url"].apply(lambda u: is_tracked(u, tracked_domain))
    ].copy()

    if df_urls_tracked.empty:
        topic_urls_tracked = pd.DataFrame(
            columns=[
                "cluster_label",
                "url",
                "queries_citing_url",
                "topic_query_count",
                "query_coverage_pct",
            ]
        )
        return topic_summary, topic_urls_tracked

    # For each (topic, url) from the tracked domain: in how many distinct queries does it appear?
    topic_urls_tracked = (
        df_urls_tracked.groupby(["cluster_label", "url"])
        .agg(
            queries_citing_url=("query", "nunique"),
        )
        .reset_index()
    )

    # Add topic sizes to compute per-URL % coverage within topic
    topic_urls_tracked = topic_urls_tracked.merge(
        topic_sizes, on="cluster_label", how="left"
    )
    topic_urls_tracked["query_coverage_pct"] = (
        topic_urls_tracked["queries_citing_url"]
        / topic_urls_tracked["topic_query_count"]
        * 100.0
    )

    return topic_summary, topic_urls_tracked


# -----------------------------
# MAIN RUN
# -----------------------------
if run_button:
    if uploaded is None:
        st.error("Upload a CSV first.")
        st.stop()

    if not domain:
        st.error("Enter a tracked domain.")
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
    df["domain_cited"] = False
    df["citation_position"] = np.nan
    df["all_citations"] = ""

    progress = st.progress(0)
    status_text = st.empty()

    # OpenAI calls
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        q = str(row["query"])

        status_text.text(f"Processing {i}/{len(df)}: {q[:80]}...")

        answer_text, urls = call_openai_with_web_search(q, chat_model)
        cited, pos = first_domain_position(urls, domain)

        df.at[idx, "domain_cited"] = cited
        df.at[idx, "citation_position"] = pos
        df.at[idx, "all_citations"] = ";".join(urls)

        progress.progress(i / len(df))

    st.success("OpenAI queries completed.")

    # Aggregations
    topic_summary, topic_urls_tracked = build_topic_aggregations(df, domain)

    # -------------------------
    # Display results
    # -------------------------
    st.markdown("## Topic-level visibility (tracked domain only)")
    if topic_summary.empty:
        st.info("No topics found.")
    else:
        st.dataframe(
            topic_summary.sort_values("visibility_pct", ascending=False),
            use_container_width=True,
        )

    st.markdown("## Tracked-domain URLs per topic")
    if topic_urls_tracked.empty:
        st.info("No URLs from your domain were cited.")
    else:
        topics = sorted(topic_urls_tracked["cluster_label"].dropna().unique())
        selected_topic = st.selectbox("Select topic", topics)

        topic_view = topic_urls_tracked[
            topic_urls_tracked["cluster_label"] == selected_topic
        ].sort_values("query_coverage_pct", ascending=False)

        st.dataframe(
            topic_view[
                [
                    "url",
                    "queries_citing_url",
                    "topic_query_count",
                    "query_coverage_pct",
                ]
            ],
            use_container_width=True,
        )

    st.markdown("## Query-level results")
    st.dataframe(
        df[
            [
                "query",
                "cluster_label",
                "domain_cited",
                "citation_position",
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
        "llm_visibility_queries.csv",
        "text/csv",
    )
    st.download_button(
        "Download topic summary CSV",
        topic_summary.to_csv(index=False).encode("utf-8"),
        "llm_visibility_topics.csv",
        "text/csv",
    )
    st.download_button(
        "Download tracked URLs per topic CSV",
        topic_urls_tracked.to_csv(index=False).encode("utf-8"),
        "llm_visibility_tracked_urls.csv",
        "text/csv",
    )


