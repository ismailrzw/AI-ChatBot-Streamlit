import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from a .env file (if it exists in the same directory)
load_dotenv()

# Get the OpenAI API key from the environment variables
openai_key = os.environ.get("OPENAI_API_KEY")

# Set PDF path manually (you can modify this as needed)
pdf_path = r"C:\Users\lenovo\Desktop\LangChain\Monopoly_Rules.pdf"

# Load and process the document
def process_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

vector_store = process_document(pdf_path)

# Initialize QA system
def create_qa_system(vector_store, openai_key):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)

    prompt_template = ChatPromptTemplate.from_template(
        """Use the following context to answer the question:
        {context}

        Question: {question}
        Answer in detail:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt_template}
    )

# Only proceed if API key is available
if openai_key:
    qa_system = create_qa_system(vector_store, openai_key)

    # Example usage
    while True:
        question = input("Ask a question about the document (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        result = qa_system({"query": question})
        print("\nAnswer:\n", result["result"])

        print("\nRelevant Sections:\n")
        for i, doc in enumerate(result["source_documents"]):
            print(f"Section {i+1}:")
            print(doc.page_content)
            print("-" * 40)
else:
    print("OpenAI API key not found. Please set it in your environment variables or .env file.")
