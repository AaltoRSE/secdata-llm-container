import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
import os
import getpass
from typing_extensions import List, TypedDict
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import dotenv
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

# Use local documents instead of web loading
local_docs = [
    Document(page_content="Task decomposition is the process of breaking down complex tasks into smaller, more manageable sub-tasks."),
    Document(page_content="There are several approaches to task decomposition, including LLM-based and human-specified methods.")
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(local_docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Remove hub.pull (which requires internet) and define prompt locally
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

model_id = "google/gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    device_map="auto"
)

# Initialize LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    print("Retrieved docs: ", retrieved_docs)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # Format the input according to Gemma's chat template
    messages = [
        {"role": "user", "content": prompt.invoke({"question": state["question"], "context": docs_content}).text}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)
    response = llm.invoke(formatted_input)
    print("Response: ", response)
    # Extract the generated text from the response
    if isinstance(response, list) and len(response) > 0:
        return {"answer": response[0]["generated_text"]}
    elif isinstance(response, dict) and "generated_text" in response:
        return {"answer": response["generated_text"]}
    return {"answer": str(response)}  # fallback to string representation


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
response = graph.invoke({"question": "What is Task Decomposition?"})
print("Answer: ", response["answer"])