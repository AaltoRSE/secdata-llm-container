import os
from pdf2text_PyMuPDF4LLM import get_all_pdfs_as_chunks
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from model_utils import get_local_model_path

# Where to store the FAISS index
faiss_index_path = "faiss_scientific_papers.index"

# Chunk size to use when splitting the documents
chunk_size = 500
chunk_overlap = 25

# Initialize embeddings with local model
embedding_model_path = get_local_model_path("sentence-transformers/all-MiniLM-L6-v2")
embeddings = SentenceTransformerEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"local_files_only": True}
)
dimensions: int = len(embeddings.embed_query("dummy"))

# Generate embeddings and create the FAISS index
print("Creating FAISS index ...")

vector_store = FAISS(
    embedding_function=embeddings,
    index=IndexFlatL2(dimensions),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=True,
)

text_chunks_dict = get_all_pdfs_as_chunks("./", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

for filename, chunk_list in text_chunks_dict.items():
    filenames = [{"source": filename}]*len(chunk_list)
    vector_store.add_texts(texts=chunk_list, metadatas=filenames)

# Save the FAISS index for later use
vector_store.save_local(faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")

# Example: How to load the index later
print(f'Testing by trying to load the created vector index ...')
loaded_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

print(loaded_vector_store)
print('vector_store distance strategy:', vector_store.distance_strategy, '. loaded_vector_store distance strategy:', loaded_vector_store.distance_strategy)
