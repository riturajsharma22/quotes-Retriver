import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import os

# 1. Load & clean the quotes
dataset = load_dataset('Abirate/english_quotes')
df = (
    dataset['train']
    .to_pandas()
    .dropna(subset=['quote'])
    .drop_duplicates(subset=['quote'])
    .reset_index(drop=True)
)

df['quote_clean'] = df['quote'].str.strip()
df['author'] = df['author'].fillna('')
df['tags'] = df['tags'].apply(lambda tags: ', '.join(tags) if isinstance(tags, (list, np.ndarray)) else '')

# 2. Clean author names (strip trailing commas/spaces)
df['author_clean'] = (
    df['author']
      .astype(str)
      .str.strip()
      .str.rstrip(',')
)

# 3. Load your fine-tuned SentenceTransformer model
model = SentenceTransformer('rituraj18/quote-embedder-finetuned_ver_8')

# 4. Encode quotes in batches
batch_size = 128
embeddings = []
for start in range(0, len(df), batch_size):
    batch = df['quote_clean'][start:start+batch_size].tolist()
    embs = model.encode(batch, normalize_embeddings=True)
    embeddings.append(embs)
embeddings = np.vstack(embeddings)

# 5. Create & populate ChromaDB collection (persistent mode)
client = PersistentClient(path="chroma_storage")
collection_name = "english_quotes_minilm"

# Delete old collection if it exists
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

# Create collection
col = client.create_collection(name=collection_name)

# 6. Prepare metadata (clean author + tags string)
metadatas = df[['author_clean', 'tags']].rename(columns={'author_clean': 'author'}).to_dict(orient='records')

# 7. Add data to ChromaDB
col.add(
    ids=df.index.astype(str).tolist(),
    documents=df['quote_clean'].tolist(),
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print(f"✅ Indexed {len(df)} quotes into ChromaDB at 'chroma_storage'.")
