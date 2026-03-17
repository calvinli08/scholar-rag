"""ScholarRAG Frontend - Main Application."""

import streamlit as st

st.set_page_config(
    page_title="ScholarRAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ScholarRAG")
st.markdown("### AI-Powered Academic Research Assistant")
st.markdown(
    """
    Welcome to ScholarRAG! Upload academic papers and chat with them using 
    retrieval-augmented generation (RAG) to get accurate, cited answers.
    
    **Getting Started:**
    - 📄 **Upload**: Go to the Upload page to add PDF papers to the knowledge base
    - 💬 **Chat**: Go to the Chat page to ask questions and generate reports
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📄 Upload Papers")
    st.markdown("Upload PDF documents to build your research knowledge base.")
    if st.button("Go to Upload →", key="upload_btn"):
        st.switch_page("pages/upload.py")

with col2:
    st.markdown("#### 💬 Chat & Reports")
    st.markdown("Ask questions and generate RAG-based research reports.")
    if st.button("Go to Chat →", key="chat_btn"):
        st.switch_page("pages/chat.py")
