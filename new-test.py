import os
from dotenv import load_dotenv
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="PDF Chat with Memory")
st.title("💬 Chat with or without your PDF")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF (optional)", type="pdf")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

# Default: no retriever
retriever = None

# If PDF is uploaded, process it and build retriever
if pdf_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Input box
user_question = st.text_input("Ask a question:")

if user_question:
    if retriever:
        # If PDF uploaded, use retrieval-augmented chat
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
        response = qa_chain.run(user_question)
    else:
        # Fallback: general conversation without retrieval
        convo_chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        response = convo_chain.run(user_question)

    st.write("### Answer:")
    st.write(response)

    # Save to chat history
    st.session_state.chat_history.append((user_question, response))

    st.write("---")
    st.write("### Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")
