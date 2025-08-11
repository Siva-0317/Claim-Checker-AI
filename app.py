import streamlit as st
import json
import threading
import uvicorn
from fastapi import FastAPI, Request
from backend import process_and_index, process_and_index_from_url, query_llm

# -------------------------------
# FASTAPI WEBHOOK ENDPOINT
# -------------------------------
fastapi_app = FastAPI()

@fastapi_app.post("/api/v1/hackrx/run")
async def run_submission(request: Request):
    """
    Webhook endpoint for HackRx.
    Accepts:
    {
        "documents": "<doc_url>",
        "questions": ["Q1", "Q2", ...]
    }
    """
    data = await request.json()
    document_url = data.get("documents")
    questions = data.get("questions", [])

    if not document_url or not questions:
        return {"error": "Missing 'documents' URL or 'questions' list."}

    # Process document from URL
    process_and_index_from_url(document_url)

    # Answer all questions
    answers = []
    for q in questions:
        ans = query_llm(q)
        try:
            ans_json = json.loads(ans.split("```")[0].strip())
            answers.append(ans_json)
        except:
            answers.append({"raw_response": ans})

    return {"answers": answers}

def start_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# -------------------------------
# START FASTAPI IN BACKGROUND
# -------------------------------
threading.Thread(target=start_fastapi, daemon=True).start()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="📄 LLM Document QA", layout="centered")
st.title("📄 LLM Document Processing System (Cohere API)")

st.markdown("Upload a document and ask questions. The system uses semantic search + Cohere LLM to give answers.")

# Upload section
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    with st.spinner("Processing..."):
        result = process_and_index(uploaded_file.read(), uploaded_file.name)
        st.success(result)

# Query section
query = st.text_input("Enter your query:")
if st.button("Ask"):
    with st.spinner("Generating answer..."):
        response = query_llm(query)
        try:
            parsed = json.loads(response.split("```")[0].strip())
            st.json(parsed)
        except:
            st.markdown("📜 Raw LLM Output:")
            st.text(response)
