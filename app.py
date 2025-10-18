import streamlit as st
import os
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Knowledge Chat", layout="centered")
st.title("💬 Chat with Your Documents")

# Input your Gemini API key
api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API key to continue.")
    st.stop()

# Initialize Gemini client
os.environ["GEMINI_API_KEY"] = api_key
client = genai.Client()

# File uploader
uploaded_files = st.file_uploader(
    "📁 Upload .txt files", 
    type=["txt"], 
    accept_multiple_files=True
)

# If files uploaded
if uploaded_files:
    texts = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        texts.append(text)
        st.success(f"Loaded: {file.name}")

    # Create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents(texts)

    # Create vector store
    st.write("🔍 Creating vector embeddings...")
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("✅ Knowledge base created!")

    # Chat section
    st.subheader("💡 Ask a question about your files")
    query = st.text_input("Type your question:")

    if query:
        with st.spinner("Thinking..."):
            retrieved_docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            prompt = f"""
            You are an expert AI assistant trained to answer questions **only** using the provided context.
            Use the exact facts in the context; if something is not found, politely say you don't know.
            
            Context:
            {context}
            
            Question: {query}
            
            Guidelines:
            - Answer clearly and concisely in full sentences.
            - Summarize or combine relevant points.
            - Do not say "context says" — integrate it naturally.
            - Avoid repeating the question.
            - Use bullet points if it helps clarity.
            
            Answer:
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.markdown("### 🧠 Answer")
            st.write(response.text)

else:
    st.info("Upload one or more `.txt` files to get started.")
