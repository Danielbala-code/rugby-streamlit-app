# app.py

import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import re

# === Load Models and Index ===
@st.cache_resource
def load_models():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return encoder, reranker

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_rugby_index.index")

@st.cache_data
def load_metadata():
    return pd.read_csv("rugby_context_metadata_with_embeddings.csv")

encoder, reranker = load_models()
index = load_faiss_index()
metadata_df = load_metadata()

# === Utility: Extract filters from query ===
def extract_filters(query, df):
    filters = {}
    for col in ['name', 'team', 'team_vs', 'position']:
        values = df[col].dropna().unique()
        for val in values:
            if isinstance(val, str) and val.lower() in query.lower():
                filters[col] = val
    return filters

# === Search & Filter Logic ===
def search_and_respond(query, top_k=15):
    query_embedding = encoder.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)
    matched_contexts = metadata_df.iloc[I[0]].copy()

    # Re-rank
    scores = reranker.predict([[query, ctx] for ctx in matched_contexts['context_str']])
    matched_contexts['relevance'] = scores
    top_matches = matched_contexts.sort_values("relevance", ascending=False)

    # Optional filter based on query content
    filters = extract_filters(query, metadata_df)
    if filters:
        st.info(f"🔍 Applied Filters: {filters}")
        for col, val in filters.items():
            top_matches = top_matches[top_matches[col] == val]

    return top_matches

# === UI ===
st.set_page_config(page_title="Rugby Stats Query", layout="wide")
st.title("🏉 Rugby Stats Search Engine")
st.markdown("Ask questions like:")
st.markdown("- `Average defenders beaten by Adam Hastings vs Sharks`")
st.markdown("- `Tackles by Jamie Ritchie against Saracens`")
st.markdown("- `Meters run by a Fly Half against Exeter`")

query = st.text_input("🔍 Enter your query here")

if query:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index = faiss.read_index("faiss_rugby_index.index")
    metadata_df = pd.read_csv("rugby_context_metadata_with_embeddings.csv")
    
    results = search_and_respond(query, top_k=5)

            f"**{player}**, playing for **{team}**, recorded a **{target_metric.replace('_', ' ')}** of **{value}**."
        )
        st.caption(f"🔍 Top Match Score: {round(score, 2)} | Opponent: {match}")

