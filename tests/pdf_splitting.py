#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://pubmed.ncbi.nlm.nih.gov/7216444/ This paper has some non-digital content and is chanlleging to pdf to markdown conversion

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
pdf_path = "iai00165-0202.pdf"
# Configuration to speed up processing
config = {
  "extract_images": False,  # Skip image extraction
  "output_format": "markdown",
}
# Initialize optimized PDF converter
pdf_converter = PdfConverter(
  artifact_dict=create_model_dict(),
  # processor_list=optimized_processors,
  config=config
)

rendered = pdf_converter(pdf_path)
text, _, _ = text_from_rendered(rendered)
text


# In[7]:


from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Header 1"),
    # ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(text)
md_header_splits


# In[9]:


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter
)

def test_splitters(text, splitter_type="recursive", chunk_size=512, chunk_overlap=30):
 
    if splitter_type == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif splitter_type == "nltk":
        import nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK punkt resource...")
            nltk.download('punkt_tab')
        splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    return splitter.split_documents(text)


# In[10]:


# splits = test_splitters(md_header_splits, splitter_type="recursive")
splits = test_splitters(md_header_splits, chunk_size=512, chunk_overlap=30, splitter_type="nltk")
splits

