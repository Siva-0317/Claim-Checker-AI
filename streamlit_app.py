import streamlit as st
import json
from backend import process_and_index, process_and_index_from_url, query_llm

st.set_page_config(page_title="📄 LLM Document QA", layout="centered")
st.title("📄 LLM Document Processing System")

st.markdown("Upload a document **or** paste a document URL, then ask questions. The system uses semantic search + LLM to give answers.")

# === Upload Section ===
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
doc_url = st.text_input("Or paste document URL (PDF/DOCX)")

if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        result = process_and_index(uploaded_file.read(), uploaded_file.name)
        st.success(result)

elif doc_url:
    with st.spinner("Downloading & processing from URL..."):
        try:
            result = process_and_index_from_url(doc_url)
            st.success(result)
        except Exception as e:
            st.error(f"❌ Failed to process from URL: {str(e)}")

# === Query Section ===
query = st.text_input("Enter your query:")
if st.button("Ask"):
    with st.spinner("Generating answer..."):
        response = query_llm(query)

        # Attempt to parse JSON from LLM output
        try:
            # Handle cases where LLM wraps JSON in code blocks
            cleaned_response = response.strip()
            if "```" in cleaned_response:
                cleaned_response = cleaned_response.split("```")[0].strip()
            
            parsed = json.loads(cleaned_response)
            st.json(parsed)
        except:
            st.warning("⚠ Could not parse as JSON. Showing raw output instead:")
            st.text(response)
