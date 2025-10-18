import streamlit as st
import os
import zipfile
import shutil
from google import genai
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ------------------------------------
# STREAMLIT SETUP
# ------------------------------------
st.set_page_config(page_title="AI Sales Coach with Persistent Memory", layout="centered")
st.title("💬 AI Sales Coach — Persistent Chat with Google Drive")


# ------------------------------------
# GEMINI API KEY
# ------------------------------------
api_key = st.text_input("🔑 Enter your Gemini API Key", type="password")
if not api_key:
    st.info("Please enter your Gemini API key to continue.")
    st.stop()
os.environ["GEMINI_API_KEY"] = api_key
client = genai.Client()


# ------------------------------------
# GOOGLE DRIVE AUTH
# ------------------------------------
@st.cache_resource
def connect_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens a browser for OAuth on first run
    return GoogleDrive(gauth)

drive = connect_drive()


# ------------------------------------
# STREAMLIT STATE
# ------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "summary" not in st.session_state:
    st.session_state["summary"] = ""


# ------------------------------------
# GOOGLE DRIVE FOLDER INPUT
# ------------------------------------
folder_id = st.text_input("📂 Enter your Google Drive Folder ID")

if not folder_id:
    st.info("Please enter a Google Drive folder ID where your vectors/files are stored.")
    st.stop()

query = f"'{folder_id}' in parents and title contains 'vectorstore.zip'"
existing_vectors = drive.ListFile({'q': query}).GetList()

if existing_vectors:
    st.success("✅ Found an existing vectorstore in Drive!")
    choice = st.radio("Choose an action:", ["Load existing vectorstore", "Create new vectorstore"])
else:
    choice = "Create new vectorstore"


# ------------------------------------
# LOAD EXISTING VECTORSTORE
# ------------------------------------
if choice == "Load existing vectorstore":
    file_id = existing_vectors[0]['id']
    file_name = existing_vectors[0]['title']
    st.info(f"Downloading vectorstore `{file_name}` from Google Drive...")
    downloaded = drive.CreateFile({'id': file_id})
    downloaded.GetContentFile(file_name)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall("vectorstore_data")

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.load_local("vectorstore_data", embeddings, allow_dangerous_deserialization=True)
    st.session_state["vectorstore"] = vectorstore
    st.success("✅ Vectorstore loaded successfully!")


# ------------------------------------
# CREATE NEW VECTORSTORE
# ------------------------------------
if choice == "Create new vectorstore":
    uploaded_files = st.file_uploader("📁 Upload .txt files", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        texts = [file.read().decode("utf-8") for file in uploaded_files]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents(texts)

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vectorstore_data")

        # Zip and upload to Google Drive
        shutil.make_archive("vectorstore", 'zip', "vectorstore_data")
        file = drive.CreateFile({"title": "vectorstore.zip", "parents": [{"id": folder_id}]})
        file.SetContentFile("vectorstore.zip")
        file.Upload()

        st.session_state["vectorstore"] = vectorstore
        st.success("✅ Vectorstore created and uploaded to Google Drive!")


# ------------------------------------
# CHAT INTERFACE
# ------------------------------------
if st.session_state["vectorstore"]:
    st.divider()
    st.markdown("### 💬 Chat with your knowledge base")

    query = st.chat_input("Ask a question...")

    # Show full conversation
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
            vectorstore = st.session_state["vectorstore"]
            retrieved_docs = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            # Prepare last messages and summary
            last_turns = st.session_state["history"][-6:]
            chat_snippet = "\n".join(
                [f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}" for msg in last_turns]
            )

            prompt = f"""
You are an AI Sales Coach helping a sales rep understand company materials.

Summary of previous discussion:
{st.session_state['summary']}

Recent messages:
{chat_snippet}

Relevant context from documents:
{context}

New question:
User: {query}

Guidelines:
- Be helpful, professional, and contextual.
- Use the summary and document context naturally.
- Do not repeat previous answers.
- If unsure, politely say you don’t know.
"""

            # Generate response
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text.strip()

            with st.chat_message("assistant"):
                st.markdown(answer)

            # Update history
            st.session_state["history"].append({"role": "user", "content": query})
            st.session_state["history"].append({"role": "assistant", "content": answer})

            # Update summary memory
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
    st.info("Upload or load a vectorstore to start chatting.")
