
# import streamlit as st

# from core.vector_store import reset_collection
# from ui.streamlit_ui import render_ui
# from pipeline import run_pipeline
# import os


# collection = None 
# st.set_page_config(page_title="PDF Topic Extractor",layout="wide")

# st.title("PDF Folder Topic Extraction")
# folder_path=st.text_input("Enter folder path containing PDFs")

# if folder_path and os.path.isdir(folder_path):
#     if st.button("Extract Topics"):
#         reset_collection()
        
#         with st.spinner("Processing PDFs....."):
#             run_pipeline(folder_path)
#             collection = store_chunks(chunks, embeddings,topics,topic_model)
#         st.success("Topics extracted successfully!")

# if collection is not None:
#     render_ui(collection)


import os
import streamlit as st

from core.vector_store import reset_collection, store_chunks
from ui.streamlit_ui import render_ui
from core.pipeline import run_pipeline  # assuming you have this

st.set_page_config(page_title="PDF Topic Extractor", layout="wide")
st.title("PDF Folder Topic Extraction")

# Persist across reruns
if "collection" not in st.session_state:
    st.session_state.collection = None

folder_path = st.text_input("Enter folder path containing PDFs")

if folder_path:
    if not os.path.isdir(folder_path):
        st.error("Invalid folder path. Please enter a valid folder containing PDFs.")
    else:
        if st.button("Extract Topics"):
            # Reset DB collection (fresh start)
            reset_collection()

            with st.spinner("Processing PDFs....."):
                # ✅ IMPORTANT: capture returned values
                chunks, embeddings, topics, topic_model = run_pipeline(folder_path)

                # Store to Chroma
                st.session_state.collection = store_chunks(
                    chunks, embeddings, topics, topic_model
                )

            st.success("Topics extracted successfully!")

# Render UI only if collection is available
if st.session_state.collection is not None:
    render_ui(st.session_state.collection)
else:
    st.info("Enter a folder path and click **Extract Topics** to begin.")




