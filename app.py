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

# Load OpenAI API key from Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"].strip()
openai = OpenAI(api_key=api_key)

# Load models and DB 
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

    return embedder, col, nlp, last_name_index

embedder, col, nlp, LAST_NAME_INDEX = load_resources()

# Author & keyword extraction
def extract_author(query: str) -> str | None:
    query_lc = query.lower().strip()

    
    m = re.search(r'\bby\s+([A-Za-z ]+)$', query_lc)
    if m:
        candidate = m.group(1).strip()
        last = candidate.split()[-1].lower()
        if last in LAST_NAME_INDEX:
            matches = [
                a for a in LAST_NAME_INDEX[last]
                if a.lower() == candidate or candidate in a.lower()
            ]
            return matches[0] if matches else LAST_NAME_INDEX[last][0]

    # 2) SpaCy PERSON entity
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            last = ent.text.split()[-1].lower()
            if last in LAST_NAME_INDEX:
                return LAST_NAME_INDEX[last][0]

    # 3) Last-resort substring on last names
    for last, authors in LAST_NAME_INDEX.items():
        if last in query_lc:
            return authors[0]

    return None

def extract_keywords(query: str) -> List[str]:
    return re.findall(r'\b\w+\b', query.lower())

# Retrieve from ChromaDB with author + tag fallback 
def retrieve_quotes(query: str, top_k: int = 5, overfetch: int = 20):
    author   = extract_author(query)
    keywords = extract_keywords(query)

    # Pull back an author‐filtered batch
    results = col.query(
        query_embeddings=[embedder.encode([query], normalize_embeddings=True)[0]],
        n_results=overfetch,
        where={"author": author} if author else None,
        include=["documents", "metadatas", "distances"]
    )
    docs, metas, dists = (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )

    # First pass: require tag match
    filtered: List[Tuple[str, Dict[str,str], float]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        tag_str  = meta.get("tags", "").lower()
        tag_list = [t.strip() for t in tag_str.split(",") if t.strip()]
        if any(kw in tag_list or kw in tag_str for kw in keywords):
            filtered.append((doc, meta, dist))

    # Fallback: if no author+tag hits, show all author docs
    if not filtered and author:
        filtered = list(zip(docs, metas, dists))

    # Sort by similarity and return top_k
    filtered.sort(key=lambda x: x[2])
    return filtered[:top_k]

# Build system prompt for LLM 
SYSTEM_PROMPT = (
    "You are a quote assistant. Use only the quotes provided below to answer the user's query. "
    "Return a not more than 3 JSON array with keys 'quote', 'author', 'tags' and summary of json array. Do not invent or modify quotes. "
    "If none match, respond with: not found"
)

def build_prompt(query: str, hits: List[Tuple[str, Dict[str, str], float]]) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Query: {query}\n\nHere are some candidate quotes:"}
    ]
    for i, (doc, meta, dist) in enumerate(hits, 1):
        messages.append({
            "role": "user",
            "content": f"{i}. “{doc}” — {meta['author']} (tags: {meta['tags']}, score: {dist:.2f})"
        })
    messages.append({
        "role": "user",
        "content": "Return matching quotes as JSON array with 'quote', 'author','tags' and summary of json array. If none match, reply: not found"
    })
    return messages

# Query LLM 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-...")
openai = OpenAI()

def rag_quotes(query: str, top_k: int = 5) -> str:
    hits = retrieve_quotes(query, top_k=top_k)
    if not hits:
        return "not found"
    messages = build_prompt(query, hits)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content
# Streamlit UI
st.set_page_config(page_title="Quote Retriever", page_icon="📚")
st.title("RAG-Based Semantic Quote Retrieval and Structured QA")

query = st.text_input("Enter your query (e.g. Love quotes Shakespeare)", "humor quotes by Oscar Wilde")

if st.button("Search"):
    with st.spinner("Searching..."):
        result = rag_quotes(query)
    st.subheader("Results")
    st.code(result, language="json")
