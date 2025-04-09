from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


model_id = "google/gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer=tokenizer,
    device_map="auto",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50
    },
    max_new_tokens=512,
    generate_kwargs={
        "temperature": 0.7,
        "repetition_penalty": 1.1
    }
)
workflow = AgentWorkflow.from_tools_or_functions(
    [multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())