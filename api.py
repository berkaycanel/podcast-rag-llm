# api.py — thin HTTP wrapper around chat.ask()
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from chat import ask   # uses OLLAMA_URL + LLM_NAME from env

class AskIn(BaseModel):
    question: str
    top_k: int = 10
    episode_type: str | None = None
    source: str = "meta"          # 'meta' | 'transcript' | 'both'

class AskOut(BaseModel):
    answer: str
    hits: list[dict]

app = FastAPI()

@app.post("/ask", response_model=AskOut)
def ask_endpoint(body: AskIn):
    answer, hits = ask(
        question=body.question,
        top_k=body.top_k,
        episode_type=body.episode_type,
        source=body.source,
        model_name=os.environ.get("LLM_NAME", "qwen2.5:14b-instruct-q4_K_M"),
    )
    return AskOut(answer=answer, hits=hits)

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, workers=1)
