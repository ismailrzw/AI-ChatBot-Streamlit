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

# Load .env for API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit app
st.set_page_config(page_title="PDF Chat with Memory")
st.title("📄 Chat with your PDF (Memory Included)")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if pdf_file is not None:
    # Save PDF temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Embeddings + Chroma vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vectordb.persist()

    # Memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LLM model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    # Conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True
    )

    # Chat input
    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        response = qa_chain.run(user_question)
        st.write("### Answer:")
        st.write(response)

        # Update history
        st.session_state.chat_history.append((user_question, response))

        st.write("---")
        st.write("### Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}:** {q}")
            st.write(f"**A{i+1}:** {a}")
