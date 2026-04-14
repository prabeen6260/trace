import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 1. Setup Page Config
st.set_page_config(page_title="Memory Shortage AI Assistant", layout="wide")

# --- CSS to fix horizontal overflow ---
st.markdown("""
    <style>
    .stChatMessage {
        overflow-wrap: break-word;
        word-break: break-word;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Memory Shortage Q&A System")

# It's safer to use st.secrets or environment variables for this!
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]
# 2. Initialize Vector Store
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory="./memory_db", embedding_function=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# 3. Setup History
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Add a "Clear Chat" button to the sidebar (Production Feature)
if st.sidebar.button("Clear Chat History"):
    msgs.clear()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

template = """Answer the question based ONLY on the context below. 
If you don't know, say you don't know. 

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"], 
    template=template
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_history_string(inputs):
    return "\n".join([f"{m.type}: {m.content}" for m in msgs.messages])

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "chat_history": RunnableLambda(get_history_string),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Chat UI Logic
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if query := st.chat_input("What would you like to know?"):
    st.chat_message("human").write(query)
    msgs.add_user_message(query)

    with st.chat_message("ai"):
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)
            response = rag_chain.invoke({"question": query})
            
            # --- IMPROVED SOURCE FORMATTING ---
            # We create clickable links instead of raw URLs
            source_links = []
            for doc in docs:
                source_name = doc.metadata.get('source', 'Source')
                url = doc.metadata.get('url', '#')
                source_links.append(f"[{source_name}]({url})")
            
            # Remove duplicates and join with a bullet or pipe
            unique_sources = list(set(source_links))
            source_text = "\n\n**Sources:** " + " | ".join(unique_sources)
            
            full_response = response + source_text
            st.markdown(full_response)
            
            msgs.add_ai_message(full_response)