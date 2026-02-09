from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages):
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks= []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for chunk in splits:
            chunks.append({
                "text":chunk,
                "page_no":page["page_no"],
                "source":page["source"]
            })
    return chunks