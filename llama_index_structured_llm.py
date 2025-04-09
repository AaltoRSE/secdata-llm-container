from llama_index.readers.file import PDFReader
from pathlib import Path

from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import math

from datetime import datetime
from pydantic import BaseModel, Field
import json

class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )
pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("./data_parallel_cpp.pdf"))
text = documents[0].text
model_id = "google/gemma-3-27b-it"
llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    device_map="auto",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True
    },
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.3,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "num_beams": 1,
        "top_k": 50
    }
)
sllm = llm.as_structured_llm(Invoice)

try:
    response = sllm.complete(text)
    if any(math.isnan(x) or math.isinf(x) for x in response.logprobs):
        raise ValueError("Model output contains NaN or infinite values")
    json_response = json.loads(response.text)
    print(json.dumps(json_response, indent=2))
except ValueError as e:
    print(f"Error: {e}")
    print("Raw LLM output:")
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")