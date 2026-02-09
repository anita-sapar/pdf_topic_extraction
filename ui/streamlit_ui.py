
import streamlit as st
from core.vector_store import get_chunks_by_topic  # remove get_topics import
# from core.pipeline import run_pipeline  # not used here, ok to keep if needed

def render_ui(collection):
    if collection is None:
        st.warning("No data available")
        return

    st.sidebar.header("Topics")

    topics = get_topics_from_collection(collection)

    if not topics:
        st.sidebar.info("No Topics found yet")
        return

    selected_topic = st.sidebar.selectbox("Select Topic", topics)

    st.subheader(f"Topic: {selected_topic}")

    results = get_chunks_by_topic(selected_topic)

    for doc, meta in zip(results["documents"], results["metadatas"]):
        with st.expander(f"{meta.get('source','Unknown')} | Page {meta.get('page_no','?')}"):
            st.write(doc)

    return selected_topic


def get_topics_from_collection(collection):
    results = collection.get(include=["metadatas"])

    topics = {
        m.get("topic_name")
        for m in results.get("metadatas", [])
        if m.get("topic_name") and m.get("topic_name") != "Miscellaneous"
    }

    return sorted(list(topics))






#---------------------------------------------------------------------

# import streamlit as st
# from core.vector_store import get_topics, get_chunks_by_topic
# from core.pipeline import run_pipeline
# def render_ui(collection):

#     if collection is None:
#         st.warning("No data available")


#     st.sidebar.header("Topics")
#     topics=get_topics(collection)

#     if not topics:
#         st.info("No Topics found yet")
#         selected_topic = st.sidebar.selectbox("select Topic",topics)
#         return selected_topic

#         st.subheader(f"Topic:{selected_topic}")

#         results=get_chunks_by_topic(selected_topic)

#         for doc,meta in zip(results["documents"],results["metadatas"]):
#             with st.expander(f"{meta['source']} | Page{meta['page_no']}"):
#                 st.write(doc)


# def get_topics(collection):
#     results=collection.get(include=["metadatas"])
#     topics={
#         m["topic_name"] for m in results["metadatas"] if m["topic_name"] !="Miscellaneous"
#     }

#     return sorted(list(topics))

