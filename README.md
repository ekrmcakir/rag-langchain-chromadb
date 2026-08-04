# 🤖 GitHub Issues RAG Assistant & Vector Space Visualizer

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with **LangChain**, **Streamlit**, **ChromaDB**, and **Hugging Face Inference API**. This project indexes technical GitHub issues from the **PEFT (State-of-the-art Parameter-Efficient Fine-Tuning)** repository to answer complex engineering queries and visualizes the vector retrieval space in 2D using UMAP.

---

## ✨ Features

- **Interactive AI Assistant:** Uses LLMs (`Qwen2.5-Coder-32B-Instruct` via Hugging Face) combined with conversational history to answer technical questions about PEFT issues.
- **Context Retrieval:** Queries local ChromaDB embeddings (`all-MiniLM-L6-v2`) to pull top-K relevant repository documents.
- **2D Vector Space Map:** Real-time 2D projection (via UMAP) showing repository chunks, the active user query, matched Top-K context chunks, and vector similarity distance connections.
- **Built-in Sidebar Prompts:** Interactive quick-ask buttons in the UI for effortless testing.

---

## 📐 System Architecture

1. **Ingestion & Indexing:** GitHub issues parsed and chunked -> Embedded using Hugging Face Embeddings -> Stored in local ChromaDB.
2. **Retrieval & RAG Chain:** User query -> History-aware Query Reformulation -> Cosine Similarity Search in ChromaDB -> Top-3 Context Assembly -> LLM Synthesis.
3. **Visualization:** 384-dimensional embeddings projected to 2D space using UMAP via Plotly.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone repository
git clone https://github.com/ekrmcakir/rag-langchain-chromadb.git
cd rag-langchain-chromadb

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate

# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
# On Windows (CMD):
# venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Setup

Create a `.env` file in the root directory and add your API tokens:

```env
HF_TOKEN=your_huggingface_api_token
GITHUB_TOKEN=your_github_personal_access_token
```

### 3. Usage & Execution

```bash
# Step 1: Fetch GitHub Issues and build ChromaDB Vector Database
python main.py

# Step 2: Run the Streamlit Interactive Interface
streamlit run app.py
```