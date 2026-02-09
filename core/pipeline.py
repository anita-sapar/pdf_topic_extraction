from core.pdf_loader import load_pdfs_from_folder
from core.chunker import chunk_pages
from core.embeddings import embed_texts
from core.topic_model import build_topic_model,extract_topics
from core.vector_store import store_chunks
import streamlit as st

def run_pipeline(folder_path):
        pages=load_pdfs_from_folder(folder_path)
        chunks= chunk_pages(pages)
        st.write(f"Total pages loaded:{len(pages)}")
        st.write(f"Total chunks created:{len(chunks)}")
        texts=[c["text"] for c in chunks]
        embeddings=embed_texts(texts)
        topic_model=build_topic_model()
        #topics,_=extract_topics(topic_model,texts,embeddings)
       # topics, topic_info, topic_model=extract_topics(topic_model,texts,embeddings)
        topics, topic_info, topic_model, debug = extract_topics(topic_model, texts, embeddings)
        print("chunks:", len(chunks))
        print("embeddings:", len(embeddings))
        print("Topics:", len(topics))

        store_chunks(chunks,embeddings,topics,topic_model)
            
        return chunks, embeddings, topics, topic_model