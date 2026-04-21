import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# load API Key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_KEY")
# Load the dataset
df = pd.read_csv('main_memory_shortage_dataset.csv')

# Convert DataFrame rows into LangChain Document objects
# combine Title and Text for better context, and put URL/Source in metadata
documents = []
for index, row in df.iterrows():
    content = f"Title: {row['title']}\n\nContent: {row['text']}"
    metadata = {
        "source": row['source'],
        "url": row['url'],
        "publish_date": row['publish_date']
    }
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

# Split the documents into chunks
# This prevents hitting token limits and helps find more specific answers
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Create Embeddings and Save to ChromaDB
# This will create a local folder named 'memory_db' to store your data
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./memory_db"
)

print(f"Ingested {len(docs)} chunks into the vector store.")