from fastapi import FastAPI, Request
import uvicorn
import requests
from backend import process_and_index, query_llm

app = FastAPI()

@app.post("/hackrx/run")
async def run_query(payload: dict):
    doc_url = payload.get("documents")
    questions = payload.get("questions", [])

    # Download file
    file_bytes = requests.get(doc_url).content
    filename = doc_url.split("/")[-1]

    # Process & index
    process_and_index(file_bytes, filename)

    # Answer all questions
    answers = [query_llm(q) for q in questions]

    return {"answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
