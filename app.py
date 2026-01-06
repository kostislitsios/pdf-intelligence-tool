import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# 1. Page Setup
st.set_page_config(page_title="Ask your PDF", page_icon="📄")
st.header("📄 AI Analyst: Chat with your PDF")

# 2. Sidebar for API Key (Security Best Practice)
with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.markdown("---")
    st.write("This tool allows you to upload a PDF and ask questions about its content.")

# 3. File Uploader
pdf = st.file_uploader("Upload your PDF here", type="pdf")

# 4. Main Logic
if pdf is not None and api_key:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # Split text into chunks (AI can't read whole books at once)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create Embeddings (The "Brain")
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # 5. User Question Input
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        
        st.success(response)

elif pdf and not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")