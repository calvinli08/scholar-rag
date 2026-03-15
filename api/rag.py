import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from api.retriever import Retriever
from api.embedder import Embedder

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GROUNDING_THRESHOLD = 0.9
MAX_RETRIES = 2


class RAGState(TypedDict):
    query: str
    hyde_query: str
    chunks: List[Dict[str, Any]]
    generated_answer: str
    grounding_score: float
    retry_count: int
    error: Optional[str]


def build_rag_graph():
    """Builds the LangGraph workflow for RAG with grounding evaluation."""
    
    # Initialize components
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    retriever = Retriever(embedder=embedder)
    
    # --- Nodes ---
    
    def hyde_node(state: RAGState) -> Dict:
        """Generate a hypothetical document answer to improve retrieval."""
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Write a hypothetical answer to the following question based on your internal knowledge. This will be used to retrieve relevant documents."),
            ("human", "{query}")
        ])
        chain = hyde_prompt | llm
        response = chain.invoke({"query": state["query"]})
        return {"hyde_query": response.content}

    def retrieve_node(state: RAGState) -> Dict:
        """Retrieve relevant chunks using the HyDE query."""
        query_text = state.get("hyde_query") or state["query"]
        chunks = retriever.search(query_text, top_k=5, mode="hybrid")
        return {"chunks": chunks}

    def generate_node(state: RAGState) -> Dict:
        """Generate an answer based on retrieved chunks."""
        context = "\n\n".join([f"[Source {i+1}]: {c['content']}" for i, c in enumerate(state["chunks"])])
        
        retry_feedback = ""
        if state["retry_count"] > 0:
            retry_feedback = f"\n\nNOTE: Previous attempts contained hallucinations (Grounding Score: {state['grounding_score']:.2f}). Ensure your answer is STRICTLY derived from the provided sources. Do not invent information."

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful research assistant. Answer the user's question using ONLY the provided context. Cite your sources using [Source X] notation. If the context doesn't contain the answer, state that you don't know.{retry_feedback}"),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": state["query"]})
        return {"generated_answer": response.content, "retry_count": state["retry_count"] + 1}

    def evaluate_node(state: RAGState) -> Dict:
        """Evaluate the generated answer for grounding/hallucination using DeepEval."""
        if not state["chunks"]:
            return {"grounding_score": 0.0}
            
        context = "\n".join([c["content"] for c in state["chunks"]])
        
        # Import DeepEval here to avoid issues if not installed
        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            print("DeepEval not installed, skipping grounding evaluation")
            return {"grounding_score": 1.0}
        
        test_case = LLMTestCase(
            input=state["query"],
            actual_output=state["generated_answer"],
            retrieval_context=[context]
        )
        
        metric = FaithfulnessMetric(threshold=GROUNDING_THRESHOLD)
        try:
            metric.measure(test_case)
            score = metric.score
        except Exception as e:
            print(f"Evaluation error: {e}")
            score = 0.0
            
        return {"grounding_score": score}

    def should_retry(state: RAGState) -> str:
        """Decide whether to retry generation or end."""
        if state["grounding_score"] >= GROUNDING_THRESHOLD:
            return "end"
        if state["retry_count"] >= MAX_RETRIES:
            return "end"
        return "generate"

    # --- Graph Construction ---
    
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("hyde", hyde_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    
    # Set entry point
    workflow.set_entry_point("hyde")
    
    # Add edges
    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    
    # Conditional edge for retry loop
    workflow.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    return workflow.compile()


# Create the compiled graph instance
rag_graph = build_rag_graph()


def run_rag_workflow(query: str) -> Dict[str, Any]:
    """Execute the RAG workflow for a given query."""
    initial_state = {
        "query": query,
        "hyde_query": "",
        "chunks": [],
        "generated_answer": "",
        "grounding_score": 0.0,
        "retry_count": 0,
        "error": None
    }
    
    try:
        final_state = rag_graph.invoke(initial_state)
        return {
            "answer": final_state["generated_answer"],
            "sources": final_state["chunks"],
            "grounding_score": final_state["grounding_score"],
            "retries": final_state["retry_count"] - 1 if final_state["retry_count"] > 0 else 0
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": [],
            "grounding_score": 0.0,
            "retries": 0,
            "error": str(e)
        }
