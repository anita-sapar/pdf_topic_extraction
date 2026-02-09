import os
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

COLLECTION_NAME = "pdf_topics"


def get_client():
    # Persistent client is the recommended way for local persistence
    return chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection(client):
    # Always get a fresh handle (prevents stale UUID issues)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection():
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # If it doesn't exist, ignore
        pass
    return get_collection(client)


def store_chunks(chunks, embeddings, topics, topic_model):
    client = get_client()
    collection = get_collection(client)

    documents = []
    metadatas = []
    ids = []

    # Ensure embeddings is a plain list (Chroma accepts list[list[float]])
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        topic_id = int(topics[idx]) if topics is not None else -1

        # Safe topic naming
        if topic_id == -1:
            topic_name = "Miscellaneous"
        else:
            topic_words = topic_model.get_topic(topic_id) if topic_model is not None else None
            topic_name = topic_words[0][0] if topic_words else f"Topic {topic_id}"

        documents.append(text)
        metadatas.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "page_no": chunk.get("page_no"),
            "source": chunk.get("source"),
        })
        #ids.append(f"{chunk.get('source','unknown')}_{idx}")
        ids.append(f"{chunk.get('source','unknown')}_{chunk.get('page_no','x')}_{idx}")

    # Final consistency check once (outside loop)
    if len(embeddings) != len(documents):
        raise ValueError(
            f"Embeddings count ({len(embeddings)}) does not match documents ({len(documents)})."
        )

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    return collection


def get_topics():
    client = get_client()
    collection = get_collection(client)

    results = collection.get(include=["metadatas"])
    topics = {}

    for meta in results.get("metadatas", []):
        tname = meta.get("topic_name", "Unknown")
        topics[tname] = topics.get(tname, 0) + 1

    return topics


def get_chunks_by_topic(topic_name):
    client = get_client()
    collection = get_collection(client)

    results = collection.get(
        where={"topic_name": topic_name},
        include=["documents", "metadatas"]
    )
    return results
#---------------------------------------------------------------------------------
# import chromadb
# from chromadb.config import Settings
# import os
# from dotenv import load_dotenv

# load_dotenv()

# CHROMA_DB_DIR =os.getenv("CHROMA_DB_DIR","chroma_db")
# os.makedirs(CHROMA_DB_DIR,exist_ok=True)

# client= chromadb.Client(chromadb.Settings(
#     persist_directory=CHROMA_DB_DIR
# ))

# collection= client.get_or_create_collection(name="pdf_topics")

# def store_chunks(chunks, embeddings, topics, topic_model):
    
#     documents=[]
#     metadatas=[]
#     ids =[]

#     for idx,chunk in enumerate(chunks):
#         documents.append(chunk["text"])
#         topic_id=int(topics[idx])
#         topic_name=(
#             topic_model.get_topic(topic_id)[0][0]
#             if topic_id != -1 else "Miscellaneous"
#         )

        
#         metadatas.append({
#             "topic_id":int(topic_id),
#             "topic_name":topic_name,
#             "page_no":chunk["page_no"],
#             "source": chunk["source"]
#             })
#         ids.append(f"{chunk['source']}_{idx}")

#         # embeddings = embeddings.tolist()
#         embeddings =embeddings[:len(documents)]

#         assert len(documents)==len(embeddings)==len(metadatas)==len(ids),(len(documents),len(embeddings),len(metadatas),len(ids))
        
#         print(
#             len(documents),
#             len(embeddings),
#             len(metadatas),
#             len(ids)

#         )
    
    
#         collection.add(
#             documents=documents,
#             embeddings=embeddings,
#             metadatas=metadatas,
#             ids=ids

    
#     )
#     return collection
 

# def get_topics():
#     # results= collection.get(
#     #     where={"topic_name":topic_name},
#     #     include=["documents","metadatas"]
#     results=collection.get(include=['metadatas'])
#     topics={}

#     for meta in results["metadatas"]:
#         tname= meta["topic_name"]
#         topics[tname]=topics.get(tname,0)+1
#         return topics
#     # )


# def get_chunks_by_topic(topic_name):
#     results= collection.get(
#         where={"topic_name": topic_name},
#         include=["documents","metadatas"]
#     )
#     return results

# def reset_collection():
#     global collection
#     client.delete_collection("pdf_topics")
#     collection= client.get_or_create_collection("pdf_topics")




