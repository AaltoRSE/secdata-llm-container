import os
import sys
import re
from pdf2text_PyMuPDF4LLM import get_all_pdfs_as_chunks
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, DistanceStrategy
from faiss import IndexFlatL2

from langchain_community.docstore.in_memory import Docstore, InMemoryDocstore


from langchain_community.embeddings import SentenceTransformerEmbeddings
# Where to store the FAISS index
faiss_index_path = "faiss_scientific_papers.index"



# Chunk size to use when splitting the documents
chunk_size = 500
chunk_overlap = 25  # This parameter doesn't seem to have any effect ...?


distant_strat = DistanceStrategy.COSINE  # DistanceStrategy.EUCLIDEAN



# Ensure secrets are valid

# Initialize OpenAI embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
dimensions: int = len(embeddings.embed_query("dummy"))

# Generate embeddings and create the FAISS index
print("Creating FAISS index ...")
#vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)

vector_store = FAISS(
    embedding_function=embeddings,
    index=IndexFlatL2(dimensions),
    docstore=InMemoryDocstore(),
    #docstore=Docstore(),
    index_to_docstore_id={},
    normalize_L2=True,  # False
    #distance_strategy = distant_strat
)


text_chunks_dict = get_all_pdfs_as_chunks("./", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


for filename, chunk_list in text_chunks_dict.items():
    filenames = [{"source": filename}]*len(chunk_list)
    vector_store.add_texts(texts=chunk_list, metadatas=filenames)

# Prepare texts and associate them with filenames
#texts = [text for _, text in pdf_texts]
#metadata = [{"source": filename} for filename, _ in pdf_texts]  # To track which PDF the text is from


# Save the FAISS index for later use
vector_store.save_local(faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")


# Example: How to load the index later
print(f'Testing by trying to load the created vector index ...')
loaded_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
# distance_strategy is always DistanceStrategy.EUCLIDEAN for loaded models. Why? Bug?
#loaded_vector_store.distance_strategy = distant_strat  # Setting it manually not seem to have any effect at retrieval

print(loaded_vector_store)

print('vector_store distance strategy:', vector_store.distance_strategy, '. loaded_vector_store distance strategy:', loaded_vector_store.distance_strategy)