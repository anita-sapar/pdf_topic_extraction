
import fitz
import os  #pyMuPDF

def load_pdfs_from_folder(folder_path: str):
    all_pages = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path=os.path.join(folder_path,file)
            doc=fitz.open(pdf_path)
            for page_num,page in enumerate(doc):
                all_pages.append({
                    "text":page.get_text(),
                    "page_no":page_num+1,
                    "source":file
                })
            doc.close()

    return all_pages