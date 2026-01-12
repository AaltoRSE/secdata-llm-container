import pymupdf4llm
import pathlib
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_md_txt_from_pdf(pdf_path, group_paragraphs=False):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    if group_paragraphs:
        md_text = md_text.replace('\n\n', '<PARA_SPLIT>').replace('\n', ' ').replace('<PARA_SPLIT>', '\n')
    return md_text


def convert_txt_to_chunks(text_str, chunk_size=200, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_list = text_splitter.split_text(text_str)
    return chunk_list


def get_all_pdfs_as_chunks(pdf_dir, chunk_size=200, chunk_overlap=0):
    doc_chunks_dict = {}
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            md_text = extract_md_txt_from_pdf(pdf_path, group_paragraphs=False)
            chunk_list = convert_txt_to_chunks(md_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            doc_chunks_dict[filename] = chunk_list
    return doc_chunks_dict


if __name__ == '__main__':
    pdf_dir = "./"
    doc_chunks_dict = get_all_pdfs_as_chunks(pdf_dir, chunk_size=200, chunk_overlap=0)
    print(doc_chunks_dict)
