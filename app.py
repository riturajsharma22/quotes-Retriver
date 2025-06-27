import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re
import spacy
import chromadb
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from chromadb import PersistentClient
from dotenv import load_dotenv

# === 0. Load environment variables
import streamlit as st


# ────── Load OpenAI key from Streamlit Secrets ──────
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

# ────── Initialize the OpenAI client ──────
openai_client = OpenAI(api_key=api_key)

# === 1. Load models and DB (cache to avoid reloading)
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('rituraj18/quote-embedder-finetuned_ver_8')
    client = PersistentClient(path="chroma_storage")
    col = client.get_collection(name='english_quotes_minilm')
    nlp = spacy.load("en_core_web_sm")

    # Normalize author list once
    all_authors = set(
        meta['author'].strip(", ").strip()
        for meta in col.get()['metadatas']
        if meta.get('author', '').strip()
    )
    last_name_index = {}
    for author in all_authors:
        last_name = author.strip().split()[-1].lower()
        last_name_index.setdefault(last_name, []).append(author)

    return embedder, col, nlp, OpenAI(), last_name_index

embedder, col, nlp, openai_client, LAST_NAME_INDEX = load_resources()

# === 2. Author & keyword extraction logic (same as your code)
def extract_author(query: str) -> str | None:
    query_lc = query.lower()
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ent_text = ent.text.lower()
            for last, authors in LAST_NAME_INDEX.items():
                if last in ent_text or ent_text in last:
                    return authors[0]
    match = re.search(r'\bby\s+([a-z]+)', query_lc)
    if match:
        lname = match.group(1).lower()
        if lname in LAST_NAME_INDEX:
            return LAST_NAME_INDEX[lname][0]
    for lname in LAST_NAME_INDEX:
        if lname in query_lc:
            return LAST_NAME_INDEX[lname][0]
    return None

def extract_keywords(query: str) -> List[str]:
    return re.findall(r'\b\w+\b', query.lower())

# === 3. Quote retrieval logic
def retrieve_quotes(query: str, top_k: int = 5, overfetch: int = 25) -> List[Tuple[str, Dict[str, str], float]]:
    author = extract_author(query)
    keywords = extract_keywords(query)
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    where_filter = {"author": author} if author else None
    results = col.query(
        query_embeddings=[q_emb],
        n_results=overfetch,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]

    filtered = []
    for doc, meta, dist in zip(docs, metas, dists):
        tag_str = meta.get('tags', '').lower()
        tag_words = set(tag.strip() for tag in tag_str.split(','))
        if any(keyword in tag_words or keyword in tag_str for keyword in keywords):
            filtered.append((doc, meta, dist))

    filtered.sort(key=lambda x: x[2])
    return filtered[:top_k]

# === 4. Build system prompt
SYSTEM_PROMPT = (
    "You are a quote assistant. Use only the quotes provided below to answer the user's query. "
    "Return a JSON array with keys 'quote', 'author', 'tags' and write summary of the json output. Do not invent or modify quotes. "
    "If none match, respond with: not found"
)

def build_prompt(query: str, hits: List[Tuple[str, Dict[str, str], float]]) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nHere are some candidate quotes:"}
    ]
    for i, (doc, meta, dist) in enumerate(hits, 1):
        messages.append({
            "role": "user",
            "content": f"{i}. “{doc}” — {meta['author']} (tags: {meta['tags']}, score: {dist:.2f})"
        })
    messages.append({
        "role": "user",
        "content": "Return matching quotes as JSON array with 'quote', 'author', 'tags' and summary. If none match, reply: not found"
    })
    return messages

# === 5. Final RAG pipeline
def rag_quotes(query: str, top_k: int = 5) -> str:
    hits = retrieve_quotes(query, top_k=top_k)
    if not hits:
        return "not found"
    messages = build_prompt(query, hits)
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content

# === 6. Streamlit UI
st.set_page_config(page_title="Quote Retriever", page_icon="📚")
st.title("🎙️ Quote Search with Tag + Author Enforcement")

query = st.text_input("Enter your query (e.g. Love quotes Shakespeare)", "humor quotes by Oscar Wilde")

if st.button("Search"):
    with st.spinner("Searching..."):
        result = rag_quotes(query)
    st.subheader("Results")
    st.code(result, language="json")
