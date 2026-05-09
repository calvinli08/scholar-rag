"""ScholarRAG Frontend - Papers Management Page."""

import streamlit as st
import httpx
from datetime import datetime
from config import settings

st.set_page_config(
    page_title="My Papers - ScholarRAG",
    page_icon="📚",
    layout="wide",
)

def fetch_papers():
    """Fetch the list of uploaded papers from the API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{settings.app_host}:{settings.app_port}/api/papers")
            
            if response.status_code != 200:
                return {"error": "Failed to fetch papers"}
            
            return response.json()
    except httpx.ConnectError:
        return {"error": "Could not connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}

def fetch_ingestion_status():
    """Fetch the status of pending ingestion jobs."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{settings.app_host}:{settings.app_port}/api/ingestion/status")
            
            if response.status_code != 200:
                return {"error": "Failed to fetch ingestion status"}
            
            return response.json()
    except httpx.ConnectError:
        return {"error": "Could not connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}

st.title("📚 My Papers")
st.markdown(
    """
    View all your uploaded papers and monitor the status of ingestion jobs.
    Papers are processed asynchronously and become available for chat once ingestion completes.
    """
)

st.divider()

# Fetch data
papers_result = fetch_papers()
ingestion_result = fetch_ingestion_status()

# Display ingestion status
st.subheader("📥 Pending Ingestion Jobs")

if "error" in ingestion_result:
    st.warning(f"Could not fetch ingestion status: {ingestion_result['error']}")
elif not ingestion_result.get("jobs", []):
    st.info("No pending ingestion jobs. All papers are processed and ready!")
else:
    for job in ingestion_result["jobs"]:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File**: {job.get('filename', 'Unknown file')}")
        with col2:
            st.write(f"**Status**: {job.get('status', 'unknown')}")

st.divider()

# Display uploaded papers
st.subheader("📚 Uploaded Papers")

if "error" in papers_result:
    st.error(f"Could not fetch papers: {papers_result['error']}")
elif not papers_result.get("papers", []):
    st.info("No papers uploaded yet. Upload your first paper on the Upload page!")
else:
    # Sort papers by upload date (newest first)
    papers = sorted(papers_result["papers"], key=lambda x: x.get("uploaded_at", ""), reverse=True)
    
    for paper in papers:
        with st.expander(f"📄 {paper.get('filename', 'Unknown file')}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Uploaded**: {paper.get('uploaded_at', 'N/A')}")
            with col2:
                st.write(f"**Status**: {paper.get('status', 'unknown')}")
            with col3:
                st.write(f"**Chunks**: {paper.get('chunk_count', 0)}")
            with col4:
                st.write(f"**Tokens**: {paper.get('token_count', 0):,}")
            
            if paper.get("metadata"):
                st.markdown("**Metadata**")
                for key, value in paper.get("metadata", {}).items():
                    st.write(f"- **{key}**: {value}")

st.divider()

st.markdown("### 💡 Tips")
st.markdown(
    """
    - **Processing Time**: Ingestion jobs typically complete in 5-30 seconds depending on document size
    - **Status Updates**: Refresh this page to see the latest status of your ingestion jobs
    - **Ready for Chat**: Once a paper shows as "completed", it's available for questioning in the Chat interface
    - **Troubleshooting**: If a job appears stuck, try re-uploading the file
    """
)