## Topic Extraction using LLama:



import streamlit as st
import os
from pathlib import Path
from collections import defaultdict
import traceback

import hdbscan
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://10.129.182.50:11434"
LLM_MODEL = "llama3.1:8b-instruct-q8_0"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CLUSTER_SIZE = 3


# ============================================================
# STEP 1: LOAD SINGLE PDF
# ============================================================

def load_single_pdf(file_path):
    """
    Load a single PDF and return list of Documents.
    Extracts page content + metadata.
    """

    documents = []

    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        for page in pages:
            if page.page_content.strip():
                documents.append(
                    Document(
                        page_content=page.page_content,
                        metadata={
                            "source": Path(file_path).name,
                            "page": page.metadata.get("page", 0) + 1,
                            "author": page.metadata.get("author", "Not Available")
                        }
                    )
                )
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")

    return documents


# ============================================================
# STEP 2: CHUNKING
# Why RecursiveCharacterTextSplitter?
# - Preserves semantic flow
# - Handles long PDFs safely
# - Adds overlap to maintain context
# ============================================================

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for doc in docs:
        splits = splitter.split_text(doc.page_content)

        for split in splits:
            chunks.append(
                Document(
                    page_content=split,
                    metadata=doc.metadata
                )
            )

    return chunks


# ============================================================
# STEP 3: CLUSTERING
# Why HDBSCAN?
# - Finds natural topic groups
# - No need to define number of topics
# - Handles noise well
# ============================================================

def cluster_chunks(chunks):

    if len(chunks) < MIN_CLUSTER_SIZE:
        return {}

    try:
        model = SentenceTransformer(EMBED_MODEL)
        texts = [c.page_content for c in chunks]
        embeddings = model.encode(texts)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            metric='euclidean'
        )

        labels = clusterer.fit_predict(embeddings)

        grouped = defaultdict(list)

        for label, chunk in zip(labels, chunks):
            if label != -1:
                grouped[label].append(chunk)

        return grouped

    except Exception as e:
        st.error(f"Clustering error: {e}")
        return {}


# ============================================================
# STEP 4: LLM TOPIC LABELING
# Why LLM?
# - Generates meaningful human-readable titles
# - Better than keyword extraction
# ============================================================

# def generate_topic_label(llm, chunks):

#     combined = "\n\n".join([c.page_content[:400] for c in chunks])

#     prompt = f"""
# Generate a concise professional topic title (3–6 words)
# for the following content.

# Return ONLY the title.

# Content:
# {combined}
# """

#     try:
#         response = llm.invoke(prompt)
#         return response.content.strip()
#     except:
#         return "Untitled Topic"

def generate_topic_label(llm, chunks):
    """
    Generate a STRICT short topic title.
    Prevents LLM from returning summary.
    """

    combined = "\n\n".join([c.page_content[:300] for c in chunks])

    prompt = f"""
You are generating ONLY a topic title.

STRICT RULES:
- Maximum 6 words.
- No sentences.
- No explanation.
- No bullet points.
- No paragraph.
- No extra text.
- Output ONLY the title.
- If unsure, generate a short descriptive label.

CONTENT:
{combined}

FINAL OUTPUT:
"""

    try:
        response = llm.invoke(prompt)

        # 🔒 Force strict cleaning
        title = response.content.strip().split("\n")[0]
        title = title.replace("-", "").strip()

        # 🔒 Hard length control
        if len(title.split()) > 6:
            title = " ".join(title.split()[:6])

        # 🔒 Remove long accidental summaries
        if len(title) > 80:
            title = title[:80]

        return title

    except Exception as e:
        return "Untitled Topic"

# ============================================================
# STEP 4B: TOPIC SUMMARY GENERATION
# ============================================================

def generate_topic_summary(llm, chunks):
    """
    Generate a professional summary for a topic
    using all clustered chunks.
    """

    combined_text = "\n\n".join(
        [c.page_content[:500] for c in chunks]
    )

    prompt = f"""
You are an enterprise document analyst.

Generate a clear, professional summary (5-10 sentences)
for the following topic content.

Focus on:
- Main theme
- Key insights
- Important points
- Business relevance

CONTENT:
{combined_text}

SUMMARY:
"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except:
        return "Summary could not be generated."


# ============================================================
# STEP 5: PROCESS SINGLE PDF
# ============================================================

def process_pdf(file_path, llm):

    docs = load_single_pdf(file_path)

    if not docs:
        return None

    chunks = chunk_documents(docs)

    clusters = cluster_chunks(chunks)

    topics = {}

    for cluster_id, cluster_chunks_list in clusters.items():
        title = generate_topic_label(llm, cluster_chunks_list)
        topics[title] = cluster_chunks_list

    return {
        "chunks": chunks,
        "topics": topics
    }


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(layout="wide")
st.title("📘 Enterprise PDF Topic Explorer")

folder_path = st.text_input("Enter Folder Path Containing PDFs")

if folder_path and os.path.exists(folder_path):

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.2
    )

    # Initialize session storage
    if "pdf_data" not in st.session_state:

        st.session_state.pdf_data = {}

        pdf_files = list(Path(folder_path).glob("*.pdf"))

        if not pdf_files:
            st.error("No PDF files found.")
            st.stop()

        for pdf in pdf_files:
            with st.spinner(f"Processing {pdf.name}..."):
                result = process_pdf(pdf, llm)

                if result:
                    st.session_state.pdf_data[pdf.name] = result

    # ============================================================
    # SIDEBAR - SELECT PDF
    # ============================================================

    st.sidebar.title("📂 Select PDF")

    pdf_names = list(st.session_state.pdf_data.keys())

    selected_pdf = st.sidebar.selectbox("Choose PDF", pdf_names)

    pdf_info = st.session_state.pdf_data[selected_pdf]

    # ============================================================
    # SHOW ALL CHUNKS BUTTON
    # ============================================================

    if st.button("📄 Show All Chunks of This PDF"):

        for i, chunk in enumerate(pdf_info["chunks"]):
            with st.expander(
                f"Chunk {i+1} | Page {chunk.metadata['page']}"
            ):
                st.write(chunk.page_content)
                st.json(chunk.metadata)

    st.divider()

    # ============================================================
    # SIDEBAR - SELECT TOPIC
    # ============================================================

    st.sidebar.title("📌 Topics")

    topics = pdf_info["topics"]

    if not topics:
        st.warning("No topics detected for this PDF.")
        st.stop()

    topic_names = sorted(topics.keys())

    selected_topic = st.sidebar.radio("Select Topic", topic_names)

    # ============================================================
    # SHOW RELATED CHUNKS AUTOMATICALLY
    # ============================================================

    related_chunks = topics[selected_topic]

    st.header(f"📌 {selected_topic}")

    # ============================================================
    # GENERATE & SHOW SUMMARY
    # ============================================================

    with st.spinner("Generating topic summary..."):
        summary = generate_topic_summary(llm, related_chunks)

    st.subheader("📝 Topic Summary")

    st.info(summary)

    st.write(f"Total Related Chunks: {len(related_chunks)}")

    st.divider()


    for i, chunk in enumerate(related_chunks):
        with st.expander(
            f"Chunk {i+1} | Page {chunk.metadata['page']}"
        ):
            st.write(chunk.page_content)
            st.json({
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "author": chunk.metadata["author"]
            })

else:
    st.info("Please provide a valid folder path containing PDFs.")
