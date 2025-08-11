import os
import faiss
import tempfile
import requests
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import cohere
from utils import extract_text_from_file

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB location
VECTOR_DB_PATH = "vector_index"

# Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
co = cohere.Client(COHERE_API_KEY)


# ---------------------------
# PROCESS & INDEX FUNCTIONS
# ---------------------------
def process_and_index(file_bytes, filename):
    """Process uploaded file and index into FAISS DB."""
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    raw_text = extract_text_from_file(temp_path)

    # Larger chunks with overlap to preserve context
    chunks = [raw_text[i:i+1500] for i in range(0, len(raw_text), 1200)]

    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local(VECTOR_DB_PATH)

    return "✅ Document processed and indexed."


def process_and_index_from_url(doc_url):
    """Download document from URL and index."""
    file_ext = ".pdf" if ".pdf" in doc_url.lower() else ".docx"
    file_path = Path(tempfile.gettempdir()) / f"temp_doc{file_ext}"

    r = requests.get(doc_url)
    r.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(r.content)

    raw_text = extract_text_from_file(str(file_path))
    chunks = [raw_text[i:i+1500] for i in range(0, len(raw_text), 1200)]

    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)

    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local(VECTOR_DB_PATH)

    return "✅ Document downloaded, processed, and indexed from URL."


# ---------------------------
# QUERY FUNCTION
# ---------------------------
def query_llm(user_query):
    """Search indexed docs, build context, and query Cohere."""
    if not os.path.exists(VECTOR_DB_PATH):
        return "❌ No indexed document found."

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(user_query, k=5)  # More results for richer context
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a claim processing assistant. Use ONLY the provided context to answer. 
If the answer is not in the context, say "Not specified in the document".

Context:
{context}

Query:
{user_query}

Respond in **valid JSON** with:
- Decision: Approval, Denial, Conditional, or Not specified
- Amount: Number or range, or null if not specified
- Justification: Short reason referencing the clause
"""

    response = co.generate(
        model="command-r-plus",  # Cohere's reasoning model
        prompt=prompt,
        max_tokens=300,
        temperature=0.2
    )

    return response.generations[0].text.strip()
