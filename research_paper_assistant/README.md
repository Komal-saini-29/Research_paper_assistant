# Research Paper Q&A Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) project that lets a user upload a research paper in PDF format and ask questions about its content.

## Tech Stack

- Python
- NLP
- Sentence Transformers
- Embeddings
- RAG
- OpenAI API
- PyPDF
- Streamlit

## How it works

```text
Research Paper PDF
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Sentence Embeddings
       ↓
Semantic Retrieval
       ↓
Relevant Paper Passages
       ↓
LLM
       ↓
Grounded Answer + Sources
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

For local development, set:

```text
OPENAI_API_KEY=your_api_key_here
```

Or create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Do NOT commit your real API key to GitHub.

### 4. Run

```bash
streamlit run app.py
```

## Notes

The first run downloads the `all-MiniLM-L6-v2` embedding model. A text-based PDF works best in this first version; scanned image-only PDFs may not contain extractable text.

## Future improvements

- Add a local vector database such as FAISS or Chroma
- Support multiple research papers
- Add conversation memory
- Add citation highlighting
- Add page previews
- Add hybrid keyword + semantic search
- Add reranking
- Compare multiple papers
- Add evaluation metrics for retrieval quality
- Add a local/open-source LLM option
