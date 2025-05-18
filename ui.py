import streamlit as st
from app.retrieval import search_docs

st.set_page_config(page_title="Scalable Semantic Search")
st.title("Scalable Semantic Search")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Searching..."):
        results = search_docs(query)
        for i, res in enumerate(results):
            st.markdown(f"**Chunk {i+1}:** {res.page_content}")
