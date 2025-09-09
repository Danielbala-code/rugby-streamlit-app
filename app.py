import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_models():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("faiss_rugby_index.index")

@st.cache_data
def load_metadata():
    return pd.read_csv("rugby_context_metadata_with_embeddings.csv")

encoder = load_models()
index = load_faiss_index()
metadata_df = load_metadata()

def search(query, top_k=3):
    query_embedding = encoder.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)
    return metadata_df.iloc[I[0]].copy()

st.set_page_config(page_title="Rugby Q&A", layout="wide")
st.title("🏉 Lite Rugby Stats App")

query = st.text_input("🔍 Ask a question")

if query:
    results = search(query, top_k=3)

    top_context = results.iloc[0]
    player = top_context.get('name') or top_context.get('player_name')
    team = top_context['team']
    match = top_context['team_vs']
    metric = 'defenders_beaten'
    value = top_context.get(metric, 'N/A')

    st.markdown(f"**{player}**, from **{team}**, beat **{value}** defenders vs **{match}**.")
