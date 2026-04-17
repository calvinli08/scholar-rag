"""ScholarRAG Frontend - Chat and RAG Report Generation Page."""

import streamlit as st
import httpx
from datetime import datetime
from config import settings

st.set_page_config(
    page_title="Chat - ScholarRAG",
    page_icon="💬",
    layout="wide",
)


def search_query(query: str, top_k: int = 5):
    """Search for relevant chunks using the query."""
    try:
        with httpx.Client(timeout=30.0) as client:
            payload = {"query": query, "top_k": top_k}
            response = client.post(f"{settings.app_host}/search", json=payload)
            return response.json()
    except httpx.ConnectError:
        return {"error": "Could not connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


def generate_rag_response(
    query: str, 
    use_reranker: bool = True,
    top_k: int = 5
):
    """Generate a RAG response with citations."""
    try:
        with httpx.Client(timeout=60.0) as client:
            payload = {
                "query": query,
                "use_reranker": use_reranker,
                "top_k": top_k
            }

            response = client.post(f"{settings.app_host}:{settings.app_port}/query", json=payload)
            if response.status_code != 200:
                return {"error": f"Query failed"}
            
            return response.json()
    except httpx.ConnectError:
        return {"error": "Could not connect to API. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")


st.title("💬 Chat & RAG Reports")
st.markdown(
    """
    Ask questions about your uploaded research papers. The system will retrieve
    relevant information and generate accurate, cited answers.
    """
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to retrieve for each query"
    )
    
    use_reranker = st.checkbox(
        "Use reranker",
        value=True,
        help="Re-rank retrieved results for better relevance (slower but more accurate)"
    )
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown(
        """
        - Be specific in your questions
        - Ask about concepts, methods, or findings
        - Request comparisons between papers
        - Ask for summaries of specific sections
        """
    )


# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message.get("sources"):
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(message["sources"]):
                    st.markdown(source)
                    st.markdown(f"**Source {i}** (Score: {source.get('score', 'N/A'):.3f})")
                    st.markdown(f"*Paper*: {source.get('chunk.paper_id', 'Unknown')}")
                    st.markdown(f"*Chunk*: {source.get('text', '')[:200]}...")
                    st.divider()


# Chat input
if prompt := st.chat_input("Ask a question about your research papers..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and generating response..."):
            result = generate_rag_response(
                query=prompt,
                use_reranker=use_reranker,
                top_k=top_k
            )
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
                assistant_message = f"Sorry, I encountered an error: {result['error']}"
                sources = []
            else:
                # Debug: print full API response
                if settings.debug:
                    with st.expander("🔧 Debug: Full API Response", expanded=False):
                        st.json(result)

                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)

                sources = result.get("sources", [])
                if sources:
                    with st.expander("📚 View Sources", expanded=True):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}** (Score: {source.get('score', 'N/A'):.3f})")
                            st.markdown(f"*Paper*: {source.get('chunk', {}).get('paper_id', 'Unknown')}")
                            
                            if source.get('chunk', {}).get('text'):
                                with st.expander("📄 View Chunk Text", expanded=False):
                                    st.markdown(f"*Excerpt*: {source.get('chunk', {}).get('text')}")

                            st.divider()
                
                assistant_message = answer
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_message,
        "sources": sources
    })


# Show conversation info
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Conversation ID**: `{st.session_state.chat_id}`")
with col2:
    st.markdown(f"**Messages**: {len(st.session_state.messages)}")
