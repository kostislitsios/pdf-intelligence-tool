import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# Page Config & CSS
# --------------------------------------------------
st.set_page_config(page_title="PDF AI Analyst", page_icon="📄")

st.markdown("""
    <style>
    [data-testid='stFileUploader'] section {
        min-height: 200px;
        padding: 40px;
    }
    .stSidebar {
        min-width: 300px;
        max-width: 300px;
    }
    </style>
""", unsafe_allow_html=True)

st.header("📄 PDF AI Analyst (Multi-Provider)")

# --------------------------------------------------
# Sidebar: Model & Key Management
# --------------------------------------------------
with st.sidebar:
    st.subheader("1. Choose Your Model")
    
    # Map friendly names to (Provider_ID, Official_Model_Name)
    model_options = {
        "Llama 3.3 70B (Groq)": ("groq", "llama-3.3-70b-versatile"),
        "Gemini 2.0 Flash (Google)": ("google", "gemini-2.0-flash-exp"),
        "Codestral (Mistral)": ("mistral", "codestral-latest"), # Replaces Devstral
    }
    
    selected_label = st.selectbox("Select Model", list(model_options.keys()))
    provider, model_name = model_options[selected_label]
    
    # Dynamic Input for the specific key
    if provider == "groq":
        api_key = st.secrets["GROQ_API_KEY"]
        base_url = "https://api.groq.com/openai/v1"
        
    elif provider == "google":
        api_key = st.secrets["GOOGLE_API_KEY"]
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        
    elif provider == "mistral":
        api_key = st.secrets["MISTRAL_API_KEY"]
        base_url = "https://api.mistral.ai/v1"

    # st.markdown("---")
    # st.info(f"**Current Limit:** You are using {provider.capitalize()}'s direct free tier.")

# --------------------------------------------------
# File Uploader
# --------------------------------------------------
pdfs = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

# --------------------------------------------------
# RAG Pipeline (Cached)
# --------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_vectorstore(chunks):
    return FAISS.from_texts(chunks, get_embeddings())

if pdfs:
    full_text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text: full_text += text + "\n"

    if not full_text.strip():
        st.error("No text found in PDFs.")
        st.stop()

    # Smart Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    
    if chunks:
        vectorstore = build_vectorstore(chunks)
        
        question = st.text_input("Ask a question about your PDFs:")

        if question:
            if not api_key:
                st.error(f"Please enter your {provider.capitalize()} API key in the sidebar!")
                st.stop()

            # 🔍 Retrieval
            docs = vectorstore.similarity_search(question, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)

            # 🤖 LLM Setup (Dynamic)
            try:
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template("""
                You are an expert analyst. Answer strictly based on the context provided.
                
                Context:
                {context}
                
                Question: {question}
                """)

                chain = prompt | llm | StrOutputParser()

                with st.spinner(f"Thinking with {model_name}..."):
                    response = chain.invoke({"context": context, "question": question})
                
                st.success(response)
                    
            except Exception as e:
                st.error(f"API Error: {e}")