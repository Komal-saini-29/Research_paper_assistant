import os
import io
import re
import hashlib
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PaperQ&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f8fc;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .subtitle {
            color: #6b7280;
            font-size: 17px;
            margin-bottom: 24px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 750;
            margin-top: 12px;
            margin-bottom: 8px;
        }

        .source-box {
            padding: 14px 16px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: white;
            margin-bottom: 10px;
        }

        .status-box {
            padding: 12px 14px;
            border-radius: 10px;
            background: white;
            border: 1px solid #e5e7eb;
            margin-bottom: 10px;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5e7eb;
            padding: 16px;
            border-radius: 12px;
        }

        .small-muted {
            color: #6b7280;
            font-size: 13px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "gpt-4.1-mini"
TOP_K = 4
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "chunks": [],
    "embeddings": None,
    "paper_name": "",
    "paper_hash": "",
    "messages": [],
    "num_pages": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# =========================================================
# PDF FUNCTIONS
# =========================================================

def extract_pdf_text(pdf_bytes):
    """Extract text page-by-page from a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages


def clean_text(text):
    """Clean extracted PDF text while preserving readable spacing."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split each page into overlapping text chunks.

    Chunk size is character-based rather than token-based.
    """
    chunks = []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    for page_data in pages:
        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "page": page_number,
                        "text": chunk,
                    }
                )

            if end >= text_length:
                break

            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks


# =========================================================
# EMBEDDINGS / RETRIEVAL
# =========================================================

def build_embeddings(chunks, model):
    """Create normalized embeddings for all chunks."""
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def retrieve_chunks(question, chunks, embeddings, model, top_k=TOP_K):
    """Retrieve the most relevant chunks using cosine similarity."""
    if not chunks or embeddings is None:
        return []

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = np.dot(embeddings, question_embedding)

    top_k = min(top_k, len(chunks))
    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append(
            {
                "page": chunks[index]["page"],
                "text": chunks[index]["text"],
                "score": float(scores[index]),
            }
        )

    return results


# =========================================================
# QUESTION CLASSIFICATION
# =========================================================

def is_summary_question(question):
    """Detect questions that should summarize the complete paper."""
    q = question.lower().strip()

    summary_phrases = [
        "summarise",
        "summarize",
        "summary",
        "summarise this",
        "summarize this",
        "summarise the paper",
        "summarize the paper",
        "summarise this paper",
        "summarize this paper",
        "summarise the pdf",
        "summarize the pdf",
        "what is this paper about",
        "what is the paper about",
        "what is this pdf about",
        "what is the pdf about",
        "overview",
        "give me an overview",
        "provide an overview",
        "main findings",
        "key findings",
        "main points",
        "key points",
        "key takeaways",
        "main takeaways",
        "explain this paper",
        "explain the paper",
        "briefly explain this paper",
        "briefly explain the paper",
    ]

    return any(phrase in q for phrase in summary_phrases)


# =========================================================
# OPENAI API KEY
# =========================================================

def get_api_key(manual_key=""):
    """
    Get the OpenAI API key.

    Priority:
    1. Key entered in the sidebar for this app session
    2. Streamlit Secrets: OPENAI_API_KEY
    3. Environment variable: OPENAI_API_KEY
    """
    manual_key = (manual_key or "").strip()

    if manual_key:
        return manual_key, "Sidebar"

    secret_key = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret_key = None

    if secret_key:
        return str(secret_key).strip(), "Streamlit Secrets"

    env_key = os.getenv("OPENAI_API_KEY")

    if env_key:
        return env_key.strip(), "Environment Variable"

    return None, "Not configured"


# =========================================================
# OPENAI NORMAL Q&A
# =========================================================

def generate_answer(question, retrieved_chunks, client, model_name):
    """Answer a normal question using only retrieved paper context."""

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are PaperQ&A, a research-paper question answering assistant.

Answer the user's question using ONLY the provided paper excerpts.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the answer cannot be determined from the excerpts, say that the provided
  excerpts do not contain enough information.
- Be accurate and concise.
- When useful, mention the page number.
- Prefer clear headings and bullet points for structured answers.

USER QUESTION:
{question}

PAPER EXCERPTS:
{context}
"""

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )

    return response.output_text.strip()


# =========================================================
# COMPLETE PAPER SUMMARY
# =========================================================

def generate_summary(chunks, client, model_name, batch_size=5):
    """
    Summarize the complete paper using a map-reduce approach.

    Every chunk is processed, rather than retrieving only the top 4 chunks.
    """

    if not chunks:
        return "No readable text was found in the PDF."

    partial_summaries = []

    total_batches = (len(chunks) + batch_size - 1) // batch_size

    progress = st.progress(0, text="Preparing complete-paper summary...")

    for batch_number, start in enumerate(
        range(0, len(chunks), batch_size),
        start=1,
    ):
        batch = chunks[start:start + batch_size]

        context_parts = []

        for chunk in batch:
            context_parts.append(
                f"[Page {chunk['page']}]\n{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are analyzing part {batch_number} of {total_batches}
of a research paper.

Use ONLY the paper text provided below.

Create a factual summary of this section.

Focus on:
- topic
- research problem
- objective
- methodology
- important findings
- conclusions
- limitations or future work if mentioned

Do not use outside knowledge.
Do not invent information.

PAPER TEXT:

{context}
"""

        response = client.responses.create(
            model=model_name,
            input=prompt,
        )

        partial_summaries.append(response.output_text.strip())

        progress_value = batch_number / total_batches
        progress.progress(
            progress_value,
            text=f"Summarizing paper section {batch_number}/{total_batches}...",
        )

    progress.empty()

    combined = "\n\n".join(
        f"SECTION SUMMARY {i + 1}:\n{summary}"
        for i, summary in enumerate(partial_summaries)
    )

    final_prompt = f"""
You are creating the final summary of a research paper.

The text below contains summaries of different sections of the same paper.

Create ONE coherent final summary.

Include:

1. What the paper is about
2. Research problem
3. Main objective
4. Methodology
5. Important findings
6. Conclusion
7. Limitations or future work, if mentioned

Use ONLY the information provided below.

Do not invent information.
Do not use outside knowledge.

Write clearly using headings and bullet points where useful.

SECTION SUMMARIES:

{combined}
"""

    final_response = client.responses.create(
        model=model_name,
        input=final_prompt,
    )

    return final_response.output_text.strip()


# =========================================================
# ERROR HANDLING
# =========================================================

def show_openai_error(error):
    """Display a useful error message without exposing the API key."""

    error_text = str(error)
    lower_error = error_text.lower()

    if (
        "account_deactivated" in lower_error
        or "account deactivated" in lower_error
    ):
        st.error(
            "❌ The OpenAI account associated with the API key is deactivated."
        )

        st.warning(
            "This is an OpenAI account/API issue, not a PDF or RAG issue. "
            "Create/use an API key belonging to an active OpenAI API account "
            "and replace OPENAI_API_KEY in your Streamlit Secrets."
        )

    elif (
        "invalid_api_key" in lower_error
        or "incorrect api key" in lower_error
        or "authentication" in lower_error
        or "401" in lower_error
    ):
        st.error("❌ The OpenAI API key is invalid or not authorized.")

        st.info(
            "Check OPENAI_API_KEY in your Streamlit Secrets or environment "
            "variables. Do not paste the key into this chat."
        )

    elif (
        "quota" in lower_error
        or "insufficient_quota" in lower_error
        or "billing" in lower_error
        or "credit" in lower_error
    ):
        st.error("❌ OpenAI API billing/quota is not available for this account.")

        st.info(
            "Check that API billing/credits are active for the OpenAI API "
            "account associated with this key."
        )

    elif "rate limit" in lower_error or "429" in lower_error:
        st.error("❌ OpenAI rate limit reached.")

        st.info(
            "Wait a little and try again."
        )

    else:
        st.error("❌ OpenAI request failed.")

        with st.expander("Technical error details"):
            st.code(error_text)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📚 PaperQ&A")
    st.caption("Research paper Q&A and complete-paper summarization")

    st.divider()

    st.markdown("### ⚙️ Configuration")

    st.write(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
    st.write(f"**LLM:** `{OPENAI_MODEL_NAME}`")

    manual_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-...",
        help="Optional: enter your OpenAI API key here for this app session. "
             "The key is not displayed.",
    )

    api_key, api_key_source = get_api_key(manual_api_key)

    if api_key:
        st.success("✅ OpenAI key loaded")
        st.caption(f"Key source: {api_key_source}")
    else:
        st.warning("⚠️ OpenAI API key not configured")
        st.caption(
            "Enter your key above, or configure OPENAI_API_KEY in "
            "Streamlit Secrets/environment variables."
        )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        **1. Upload PDF**

        The PDF is extracted page-by-page.

        **2. Chunking**

        The text is split into overlapping chunks.

        **3. Embeddings**

        MiniLM creates vector embeddings.

        **4. Q&A**

        Normal questions retrieve the most relevant chunks.

        **5. Summary**

        Summary questions process the **entire paper** in batches.
        """
    )

    st.divider()

    if st.button("🗑️ Clear current paper", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.paper_name = ""
        st.session_state.paper_hash = ""
        st.session_state.messages = []
        st.session_state.num_pages = 0
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">📚 PaperQ&A</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">'
    "Upload a research paper, ask questions, or generate a summary of the complete paper."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# PDF UPLOAD
# =========================================================

st.markdown('<div class="section-title">📄 Upload Research Paper</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    help="Upload a research paper in PDF format.",
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_hash != st.session_state.paper_hash:
        with st.spinner("Reading and processing the PDF..."):
            try:
                pages = extract_pdf_text(pdf_bytes)

                readable_pages = [
                    page
                    for page in pages
                    if clean_text(page["text"])
                ]

                chunks = chunk_text(
                    readable_pages,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )

                if not chunks:
                    st.error(
                        "❌ No readable text could be extracted from this PDF."
                    )
                    st.info(
                        "If this is a scanned/image-only PDF, OCR is required "
                        "before this app can answer questions from it."
                    )
                else:
                    model = load_embedding_model()
                    embeddings = build_embeddings(chunks, model)

                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.paper_hash = current_hash
                    st.session_state.messages = []
                    st.session_state.num_pages = len(pages)

                    st.success(
                        f"✅ {uploaded_file.name} processed successfully."
                    )

            except Exception as error:
                st.error("❌ Failed to process the PDF.")

                with st.expander("Technical error details"):
                    st.code(str(error))

    else:
        st.success(
            f"✅ {st.session_state.paper_name} is already processed."
        )


# =========================================================
# PAPER STATISTICS
# =========================================================

if st.session_state.chunks:
    st.divider()

    st.subheader("📊 Paper Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="📄 Pages",
            value=st.session_state.num_pages,
        )

    with c2:
        st.metric(
            label="🧩 Text Chunks",
            value=len(st.session_state.chunks),
        )

    with c3:
        st.metric(
            label="🔎 Embeddings",
            value="MiniLM",
        )

    with c4:
        st.metric(
            label="🤖 LLM",
            value=OPENAI_MODEL_NAME,
        )


# =========================================================
# QUESTION AREA
# =========================================================

st.divider()

st.markdown('<div class="section-title">💬 Ask Your Paper</div>', unsafe_allow_html=True)

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Examples:\n"
        "• Summarize this paper\n"
        "• What problem does the paper solve?\n"
        "• What methodology was used?\n"
        "• What are the main findings?\n"
        "• What dataset was used?"
    ),
    height=130,
)

ask = st.button(
    "🚀 Ask PaperQ&A",
    type="primary",
    use_container_width=True,
)


# =========================================================
# ASK / ANSWER FLOW
# =========================================================

if ask:

    if not st.session_state.chunks:
        st.error("❌ Please upload and process a PDF first.")

    elif not question.strip():
        st.warning("⚠️ Please enter a question.")

    elif not api_key:
        st.error("❌ OpenAI API key is not configured.")

        st.info(
            "Enter your OpenAI API key in the sidebar, or configure "
            "OPENAI_API_KEY in Streamlit Secrets/environment variables."
        )

    else:
        question = question.strip()

        try:
            client = OpenAI(api_key=api_key)

            # -------------------------------------------------
            # COMPLETE PAPER SUMMARY
            # -------------------------------------------------

            if is_summary_question(question):

                with st.spinner(
                    "Generating a summary from the complete paper..."
                ):
                    answer = generate_summary(
                        st.session_state.chunks,
                        client,
                        OPENAI_MODEL_NAME,
                        batch_size=5,
                    )

                unique_pages = sorted(
                    set(
                        chunk["page"]
                        for chunk in st.session_state.chunks
                    )
                )

                sources = [
                    {
                        "page": page,
                        "text": "",
                        "score": None,
                    }
                    for page in unique_pages
                ]

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "summary": True,
                    }
                )

            # -------------------------------------------------
            # NORMAL RAG QUESTION
            # -------------------------------------------------

            else:

                model = load_embedding_model()

                with st.spinner("Searching the paper..."):
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        model,
                        top_k=TOP_K,
                    )

                if not retrieved:
                    st.warning(
                        "No relevant text could be retrieved from the paper."
                    )

                else:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(
                            question,
                            retrieved,
                            client,
                            OPENAI_MODEL_NAME,
                        )

                    st.session_state.messages.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": retrieved,
                            "summary": False,
                        }
                    )

        except Exception as error:
            show_openai_error(error)


# =========================================================
# RESULTS
# =========================================================

if st.session_state.messages:

    st.divider()

    latest = st.session_state.messages[-1]

    if latest.get("summary", False):

        st.subheader("📄 Paper Summary")

        st.markdown(latest["answer"])

        st.divider()

        st.subheader("📚 Pages Used")

        pages = [
            source["page"]
            for source in latest["sources"]
        ]

        if pages:
            st.caption(
                "The summary was generated by processing the complete paper "
                "in smaller batches rather than retrieving only a few chunks."
            )

            page_text = ", ".join(
                f"Page {page}"
                for page in pages
            )

            st.write(page_text)

    else:

        st.subheader("💬 Answer")

        st.markdown(latest["answer"])

        st.divider()

        st.subheader("🔎 Retrieved Sources")

        for index, source in enumerate(
            latest["sources"],
            start=1,
        ):
            score = source.get("score")

            if score is not None:
                title = (
                    f"Source {index} — Page {source['page']} "
                    f"— Score {score:.3f}"
                )
            else:
                title = f"Source {index} — Page {source['page']}"

            with st.expander(title):
                st.write(source["text"])

                if score is not None:
                    st.caption(
                        f"Cosine similarity / retrieval score: {score:.3f}"
                    )


# =========================================================
# PREVIOUS QUESTIONS
# =========================================================

if len(st.session_state.messages) > 1:

    st.divider()

    st.subheader("🕘 Previous Questions")

    for index, message in enumerate(
        reversed(st.session_state.messages[:-1]),
        start=1,
    ):
        question_text = message["question"]

        with st.expander(
            f"{index}. {question_text}"
        ):
            st.markdown(message["answer"])


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PaperQ&A • PDF extraction + MiniLM embeddings + OpenAI Responses API"
)
