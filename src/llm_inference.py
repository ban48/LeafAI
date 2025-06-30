import os
import multiprocessing
from llama_cpp import Llama

# IMPORTANT: 
# download the correct model https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# put the file in the folder "models"

class LeafConditionDescriber:
    def __init__(self, n_ctx=512, n_batch=8):
        """
        Wrapper class for TinyLLaMA to describe plant condition based on species and disease.

        Args:
            model_path (str): Path to the quantized GGUF model file.
            n_ctx (int): Max context window size.
            n_batch (int): Batch size for inference.
        """
        self.n_threads = multiprocessing.cpu_count()
        model_path = os.path.join(os.path.dirname(__file__), "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,                    # Max number of context Tokens
            n_threads=self.n_threads,       # CPU cores to use
            n_batch=n_batch,                # Batch size
            verbose=False
        )

    def describe(self, species: str, disease: str) -> str:
        """
        Generate a description and cure suggestion for a plant based on its species and disease.
        If there is no desease, just say that the plant is healthy.

        Args:
            species (str): Name of the plant species.
            disease (str): Name of the disease affecting the plant.

        Returns:
            str: Description and suggested cure.
        """
        if disease == "healthy":
            prompt = f"""
            ### Instruction:
            Write a sentence describing the {disease} condition of the {species} plant's leaf.

            ### Response:
            """
        else:   
            prompt = f"""
            ### Instruction:
            A {species} plant is affected by {disease}.
            Write a sentence describing the condition of the leaf and suggest a possible cure.

            ### Response:
            """
        # (prompt, max token used in the response, grade of creativity in range [0, 1], stops the model where it sees "###")
        response = self.llm(prompt, max_tokens=256, temperature=0.7, stop=["###"])
        
        # The LLM returns a dictionary, we have to take the intresting parts
        return response["choices"][0]["text"].strip()
