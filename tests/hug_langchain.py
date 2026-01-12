from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch
from model_utils import get_local_model_path

model_id = "Qwen/Qwen2.5-14B-Instruct-1M"
model_path = get_local_model_path(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
hf = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10
)

print("Loading model.............. done")

from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

question = "What is electroencephalography?"
output = hf(f"Question: {question}\n\nAnswer: Let's think step by step.")
print(output[0]['generated_text'])

# Create a streamer
streamer = TextStreamer(tokenizer)

# Use streamer parameter instead of model_kwargs
for chunk in hf(f"Question: {question}\n\nAnswer: Let's think step by step.", streamer=streamer):
    print(chunk['generated_text'], end="", flush=True)

bloom_model_path = get_local_model_path("bigscience/bloom-1b7")
gpu_llm = pipeline(
    "text-generation",
    model=bloom_model_path,
    device=0 if torch.cuda.is_available() else -1,
    batch_size=2,
    model_kwargs={"temperature": 0, "max_length": 64, "local_files_only": True}
)

questions = []
for i in range(4):
    questions.append(f"Question: What is the number {i} in french?\n\nAnswer:")

answers = gpu_llm(questions, stop_sequence="\n\n")
for answer in answers:
    print(answer[0]['generated_text'])