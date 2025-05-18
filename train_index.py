from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Load and split documents
loader = DirectoryLoader("data/sample_docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(docs)

# Generate embeddings
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(splits, embedding)
db.save_local("embeddings/faiss_index")

print("FAISS index built and saved.")
