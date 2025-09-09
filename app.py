# app.py

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import re

# === Load Metadata and FAISS Index (safe to cache) ===
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_rugby_index.index")

@st.cache_data
def load_metadata():
    return pd.read_csv("rugby_context_metadata_with_embeddings.csv")

index = load_faiss_index()
metadata_df = load_metadata()

# === UI ===
st.set_page_config(page_title="Rugby Stats Query", layout="wide")
st.title("🏉 Rugby Stats Search Engine")
st.markdown("Ask questions like:")
st.markdown("- `Average defenders beaten by Adam Hastings vs Sharks`")
st.markdown("- `Tackles by Jamie Ritchie against Saracens`")
st.markdown("- `Meters run by a Fly Half against Exeter`")

query = st.text_input("🔍 Enter your query here")

# === Core Search Logic ===
def extract_filters(query, df):
    filters = {}
    for col in ['name', 'team', 'team_vs', 'position']:
        values = df[col].dropna().unique()
        for val in values:
            if isinstance(val, str) and val.lower() in query.lower():
                filters[col] = val
    return filters

def search_and_respond(query, top_k=5):
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Step 1: FAISS semantic search
    query_embedding = encoder.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)
    matched_contexts = metadata_df.iloc[I[0]].copy()

    # Step 2: Re-rank
    scores = reranker.predict([[query, ctx] for ctx in matched_contexts['context_str']])
    matched_contexts['relevance'] = scores
    top_matches = matched_contexts.sort_values("relevance", ascending=False)

    # Step 3: Optional filter
    filters = extract_filters(query, metadata_df)
    if filters:
        st.info(f"🔍 Applied Filters: {filters}")
        for col, val in filters.items():
            top_matches = top_matches[top_matches[col] == val]

    return top_matches.head(1)  # show top result

# === Display Results ===
if query:
    with st.spinner("⚙️ Searching..."):
        result = search_and_respond(query)

    if not result.empty:
        row = result.iloc[0]
        player = row.get('name')
        team = row.get('team')
        match = row.get('team_vs')
        metric = 'defenders_beaten'  # fallback
        value = row.get(metric, 'N/A')

        st.markdown(
            f"**{player}**, from **{team}**, recorded **{value}** in a match vs **{match}**."
        )
    else:
        st.warning("No relevant results found.")
