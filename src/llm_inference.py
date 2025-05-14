from llama_cpp import Llama
import os

# Load quantized model (es: TinyLlama GGUF)
MODEL_PATH = "./models/tiny-llama-1.1b-chat.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_batch=8,
    verbose=False
)

def describe_leaf_condition(species: str, disease: str) -> str:
    prompt = f"""
    ### Instruction:
    A {species} plant is affected by {disease}.
    Write a sentence describing the condition of the leaf and suggest a possible cure.

    ### Response:
    """
    output = llm(prompt, max_tokens=100, temperature=0.7, stop=["###"])
    return output["choices"][0]["text"].strip()
