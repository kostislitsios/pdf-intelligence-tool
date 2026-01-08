import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
st.header("📄 Multi-PDF AI Analyst (OpenRouter)")

# --------------------------------------------------
# Shared OpenRouter API key
# --------------------------------------------------
api_key = st.secrets["OPENROUTER_API_KEY"]

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
    st.write("Upload one or more PDFs and ask questions about them.")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
pdfs = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# --------------------------------------------------
# Build knowledge base (cached)
# --------------------------------------------------
@st.cache_resource(show_spinner="Indexing PDFs...")
def build_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",  # ✅ FIX
        model="text-embedding-3-small"
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
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(full_text)

    knowledge_base = build_knowledge_base(chunks)

    question = st.text_input("Ask a question about your PDFs:")

    if question:
        docs = knowledge_base.similarity_search(question, k=4)

        llm = OpenAI(
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",  # ✅ FIX
            model_name=model_name,
            temperature=0
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        with st.spinner("Thinking..."):
            answer = chain.run(
                input_documents=docs,
                question=question
            )

        st.success(answer)
