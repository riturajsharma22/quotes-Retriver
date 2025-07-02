# 📚 Quote Retriever – RAG-Based Semantic Quote Search

A retrieval-augmented generation (RAG) based Streamlit application that allows users to semantically search for English quotes by keyword, tag, or author. Built using a fine-tuned `SentenceTransformer`, OpenAI's GPT-4o, and ChromaDB for persistent vector search.

---

## ✨ Features

* 🔍 Semantic quote retrieval using custom fine-tuned embeddings
* 🧠 Named-entity author recognition and fallback keyword filtering
* 💬 GPT-powered structured quote formatting with reasoning
* 📁 ChromaDB persistent vector store for scalable storage
* 🖼️ Streamlit UI for interactive exploration

---

## 🛠️ Installation & Setup

Follow these steps to get started:

### 1. Clone the repository

```bash
git clone https://github.com/riturajsharma22/quotes-Retriver.git
cd quotes-Retriver
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up OpenAI API key

You can either:

* Add a `.streamlit/secrets.toml` file with your key:

```toml
OPENAI_API_KEY = "your-openai-key"
```

* Or set it as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-key"
```

---

## 🚀 Running the App

To launch the Streamlit interface:

```bash
streamlit run app.py
```

Use the text input to query quotes like:

```
Love quotes by Shakespeare
truth memory quote by Mark Twain
```

---

## 📆 Quote Indexing (Optional)

If you'd like to regenerate the vector DB:

```bash
python collection.py
```

This will:

* Load and clean quotes from the `Abirate/english_quotes` dataset
* Use a fine-tuned embedding model: `rituraj18/quote-embedder-finetuned_ver_8`
* Store vectors and metadata in a persistent ChromaDB collection (`chroma_storage`)

---

## 📊 Evaluation Summary (RAGAS)

```text
RAGAS Evaluation:
{
  "context_recall":      0.5000,
  "context_precision":   0.3333,
  "faithfulness":        1.0000,
  "answer_relevancy":    0.1245,
  "answer_correctness":  1.0000
}
```

### 🧠 Insight:

* ✅ High **faithfulness** and **correctness** show that the model returns accurate results when relevant data exists.
* ⚠️ Lower **context precision** and **answer relevancy** stem from the LLM occasionally failing to match tags and author info—especially for lesser-known authors.
* ⏳ Time constraints and academic commitments limited optimization of summarization and retrieval quality.
* 📍 With more tuning, precision and relevancy can be improved significantly.

---

## 📝 Limitations & Future Improvements

* 🔍 Some author names (especially partial last names) aren't always correctly matched.
* 📉 A few relevant quotes may be excluded if tags are missing or improperly labeled.
* ⏳ Time constraints limited final optimization and robust quote summarization via LLM.
* 📚 Certain author-specific queries (e.g. “Einstein on imagination”) return partial matches.

> “This is a work in progress, and the current results highlight the potential. Given more time, I’m confident the system can be improved significantly.” – *Author*

---

## 🙏 Acknowledgements

* OpenAI for LLM inference
* HuggingFace Datasets & Transformers
* ChromaDB for vector search
* Streamlit for the UI
* Quote dataset: [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)

---

## 📬 Final Note

This project demonstrates my understanding of RAG pipelines, embedding models, and structured data retrieval. While time constraints prevented full-scale refinement, the functional pipeline and system design are solid. I welcome any extension, feedback, or further time for improvement!

---
