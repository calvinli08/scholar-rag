"""ScholarRAG Frontend - Document Upload Page."""

import streamlit as st
import httpx
import asyncio
from pathlib import Path
from config import settings

st.set_page_config(
    page_title="Upload Papers - ScholarRAG",
    page_icon="📄",
    layout="wide",
)

def upload_pdf(file):
    """Upload a PDF file to the ingestion API."""
    try:
        with httpx.Client(timeout=60.0) as client:
            files = {"file": (file.name, file.getvalue(), "application/pdf")}

            response = client.post(f"{settings.app_host}:{settings.app_port}/ingest/upload", files=files)

            if response.status_code != 200:
                return {"error": "Upload failed"}

            return response.json()
    except httpx.ConnectError:
        return {"error": "Could not connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


st.title("📄 Upload Papers")
st.markdown(
    """
    Upload academic papers in PDF format to build your research knowledge base.
    The system will parse, chunk, and embed the documents for retrieval.
    """
)

st.divider()

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to upload",
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)}** file(s) selected")
    
    if st.button("🚀 Upload Files", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        success_count = 0
        error_count = 0

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Uploading {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            result = upload_pdf(uploaded_file)
            
            if "error" in result:
                error_count += 1
                with results_container:
                    st.error(f"❌ **{uploaded_file.name}**: {result['error']}")
            else:
                success_count += 1
                with results_container:
                    st.success(
                        f"✅ **{uploaded_file.name}** uploaded successfully! "
                        f"Chunks: {result.get('chunks', 'N/A')}"
                    )
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("")
        progress_bar.empty()
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Successful", success_count)
        with col2:
            st.metric("Failed", error_count)
        
        if success_count > 0:
            st.info(
                "💡 Your papers are now being processed. You can start chatting with them "
                "on the Chat page once indexing is complete."
            )
else:
    st.info("👆 Select PDF files above to begin uploading")

st.divider()

st.markdown("### 📋 Upload Guidelines")
st.markdown(
    """
    - **Supported Format**: PDF only
    - **Recommended**: Academic papers, research articles, technical reports
    - **Size Limit**: Depends on API configuration (typically < 50MB per file)
    - **Processing Time**: Varies by document length (typically 5-30 seconds per paper)
    
    The system will:
    1. Parse the PDF and extract text
    2. Split into semantic chunks
    3. Generate embeddings for each chunk
    4. Store in the vector database with BM25 index
    """
)
