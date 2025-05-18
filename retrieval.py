from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def load_faiss_index(path="embeddings/faiss_index"):
    embedding = OpenAIEmbeddings()
    return FAISS.load_local(path, embedding)

def search_docs(query, k=5):
    index = load_faiss_index()
    return index.similarity_search(query, k=k)
