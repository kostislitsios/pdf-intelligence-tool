import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="PDF AI Analyst", page_icon="📄")

# Expand upload area & sidebar
st.markdown(
    """
    <style>
    .upload-box {
        border: 2px dashed #999;
        border-radius: 12px;
        padding: 80px 20px;
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 20px;
    }
    .css-1r6slb0.e1fqkh3o3 {
        min-width: 280px !important;
        max-width: 280px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("📄 PDF AI Analyst")

st.markdown(
    """
    <div class="upload-box">
        📄 Drag & drop your PDFs<br>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# API Key
# --------------------------------------------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.subheader("Configuration")

    model_name = st.selectbox(
        "Select model",
        [
            "meta-llama/llama-3.3-70b-instruct:free",
            "gpt-4o-mini:free",
            "mistralai/devstral-2512:free",
        ]
    )

    st.markdown("---")
    st.caption("Upload PDFs and ask questions")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
pdfs = st.file_uploader(
    "",
    type="pdf",
    accept_multiple_files=True
)

# --------------------------------------------------
# Build vector store
# --------------------------------------------------
@st.cache_resource(show_spinner="Indexing PDFs...")
def build_vectorstore(chunks):
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
    vectorstore = build_vectorstore(chunks)

    question = st.text_input("Ask a question about your PDFs:")

    if question:
        # 🔍 Retrieve relevant chunks
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)

        # 🤖 LLM (OpenRouter via OpenAI-compatible API)
        llm = ChatOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://your-app-name.streamlit.app",
                "X-Title": "PDF AI Analyst"
            },
        )

        # 🧾 Prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are an expert AI analyst.
Answer the question ONLY using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
        )

        # 🔗 LCEL chain
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            answer = chain.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )

        st.success(answer)
