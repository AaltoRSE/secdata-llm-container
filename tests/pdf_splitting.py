import pymupdf4llm
pdf_path = "tests/data_parallel_cpp.pdf"
# Convert PDF to markdown using pymupdf4llm
text = pymupdf4llm.to_markdown(pdf_path)


from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Header 1"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(text)
print(md_header_splits)


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


splits = test_splitters(md_header_splits, chunk_size=512, chunk_overlap=30, splitter_type="nltk")
print(splits)
