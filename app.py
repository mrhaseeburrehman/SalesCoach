import streamlit as st
import os
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(page_title="AI Sales Coach", layout="centered")
st.title("💬 AI Sales Coach — Chat with Your Docs")

# -------------------------------
# GEMINI API KEY
# -------------------------------
api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
if not api_key:
    st.info("Please enter your Gemini API key to continue.")
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key
client = genai.Client()

# -------------------------------
# FILE UPLOAD & VECTORSTORE
# -------------------------------
uploaded_files = st.file_uploader("📁 Upload .txt files", type=["txt"], accept_multiple_files=True)

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []  # store chat messages
if "summary" not in st.session_state:
    st.session_state["summary"] = ""  # running summary

if uploaded_files:
    texts = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        texts.append(text)
        st.success(f"Loaded: {file.name}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings()
    st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)
    st.success("✅ Knowledge base ready!")

# -------------------------------
# CHAT INTERFACE
# -------------------------------
if st.session_state["vectorstore"]:
    query = st.chat_input("Type your question...")

    # Display chat history (like a real chatbot)
    for msg in st.session_state["history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            # --- Retrieve relevant docs ---
            retrieved_docs = st.session_state["vectorstore"].similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            # --- Prepare chat history and summary ---
            last_turns = st.session_state["history"][-6:]
            chat_snippet = "\n".join(
                [f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" for msg in last_turns]
            )

            prompt = f"""
You are an AI Sales Coach trained to answer based on company materials.

Conversation summary so far:
{st.session_state['summary']}

Recent messages:
{chat_snippet}

Relevant context from documents:
{context}

New question:
User: {query}

Guidelines:
- Answer clearly and conversationally.
- Use the summary and context naturally.
- Don’t say “context says”.
- If unsure, politely say you don’t know.
"""

            # --- Gemini response ---
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text.strip()

            with st.chat_message("assistant"):
                st.markdown(answer)

            # --- Update history ---
            st.session_state["history"].append({"role": "user", "content": query})
            st.session_state["history"].append({"role": "assistant", "content": answer})

            # --- Update summary memory ---
            summary_prompt = f"""
Update this ongoing summary of the chat in 3–5 sentences.

Previous summary:
{st.session_state['summary']}

New exchange:
User: {query}
Assistant: {answer}
"""
            summary_update = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=summary_prompt
            )
            st.session_state["summary"] = summary_update.text.strip()

else:
    st.info("Upload your `.txt` files to start chatting.")

