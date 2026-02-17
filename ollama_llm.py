from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import subprocess
# from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms import Ollama
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

OLLAMA_PATH = r"C:\Users\ru14272\AppData\Local\Programs\Ollama\ollama.exe"
OLLAMA_TIMEOUT = None

MODEL_NAME = "llama3.1:8b"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/ask_ollama")
def ask_ollama(request: PromptRequest):
    try:
        prompt=request.prompt
        print(prompt,'prompt')
        result =run_ollama(prompt)
        print(result.stdout.strip(),"result")
        output = result.stdout.strip()
        response = output.replace("###","").replace("**","")
        # Generate a response using the local model

        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

def run_ollama(prompt):
    result = subprocess.run(
        [OLLAMA_PATH, "run", MODEL_NAME],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=OLLAMA_TIMEOUT
    )
    return result