import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
st.header("📄 Multi-PDF AI Analyst (100% Free)")

# --------------------------------------------------
# Shared OpenRouter API key (only key needed)
# --------------------------------------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.subheader("Configuration")

    model_name = st.selectbox(
        "Choose a free model",
        [
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen-3-235b-a22b:free",
            "mistralai/devstral-2512:free",
        ]
    )

    st.markdown("---")
    st.write("Upload PDFs and ask questions.")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
pdfs = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# --------------------------------------------------
# Build knowledge base (LOCAL embeddings)
# --------------------------------------------------
@st.cache_resource(show_spinner="Indexing PDFs...")
def build_knowledge_base(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if pdfs:
    full_text = ""

    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        st.error("No readable text found in the uploaded PDFs.")
        st.stop()

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(full_text)
    knowledge_base = build_knowledge_base(chunks)

    question = st.text_input("Ask a question about your PDFs:")

    if question:
        docs = knowledge_base.similarity_search(question, k=4)

        llm = OpenAI(
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model_name=model_name,
            default_headers={
                "HTTP-Referer": "https://your-app-name.streamlit.app",
                "X-Title": "Multi-PDF AI Analyst"
            },
            temperature=0
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        with st.spinner("Thinking..."):
            answer = chain.run(
                input_documents=docs,
                question=question
            )

        st.success(answer)
