import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from api.retriever import retrieve_chunks
from ingestion.embedder import get_embedder
from config import settings
from logger import get_logger, configure_logging

configure_logging()

log = get_logger(__name__)

# Configuration
GROUNDING_THRESHOLD = 0.9
MAX_RETRIES = 2


def get_llm_instance():
    """Initialize the LLM based on settings.llm_backend."""
    backend = settings.llm_backend.lower()

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key
        )
    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_llm_url
        )
    elif backend == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.vllm_model,
            temperature=settings.llm_temperature,
            base_url=settings.vllm_url,
            api_key="ollama"  # vLLM uses OpenAI-compatible API
        )
    elif backend == "cohere":
        from langchain_cohere import ChatCohere
        return ChatCohere(
            model=settings.cohere_model,
            temperature=settings.llm_temperature,
            cohere_api_key=settings.cohere_api_key
        )
    elif backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            api_key=settings.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")


def get_eval_llm_instance():
    """Initialize the LLM for evaluation based on settings.eval_backend using DeepEval native models."""
    backend = settings.eval_backend.lower()

    if backend == "openai":
        from deepeval.models import GPTModel
        return GPTModel(
            model=settings.openai_eval_model,
            api_key=settings.openai_api_key
        )
    elif backend == "ollama":
        from deepeval.models import OllamaModel
        return OllamaModel(
            model=settings.ollama_eval_model,
            base_url=settings.ollama_eval_url
        )
    elif backend == "vllm":
        from deepeval.models import OllamaModel
        # vLLM uses OpenAI-compatible API, accessible via OllamaModel with custom base_url
        return OllamaModel(
            model=settings.vllm_eval_model,
            base_url=settings.vllm_eval_url
        )
    elif backend == "gemini":
        from deepeval.models import GeminiModel
        return GeminiModel(
            model=settings.gemini_eval_model,
            api_key=settings.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported evaluation backend: {backend}")


class RAGState(TypedDict):
    query: str
    hyde_query: str
    chunks: List[Dict[str, Any]]
    generated_answer: str
    grounding_score: float
    top_k: int
    use_reranker: bool
    retry_count: int
    error: Optional[str]


def build_rag_graph():
    """Builds the LangGraph workflow for RAG with grounding evaluation."""
    
    # Initialize components using settings from config
    llm = get_llm_instance()
    
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

    async def retrieve_node(state: RAGState) -> Dict:
        """Retrieve relevant chunks using the HyDE query."""
        query_text = state.get("hyde_query") or state["query"]

        chunks = await retrieve_chunks(
            query_text,
            top_k=state["top_k"],
            use_reranker=state["use_reranker"]
        )

        return {"chunks": chunks}

    def generate_node(state: RAGState) -> Dict:
        """Generate an answer based on retrieved chunks."""

        context = "\n\n".join([f"[Source {i+1}]: {c.text}" for i, c in enumerate(state["chunks"])])
        
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

        context = "\n".join([c.text for c in state["chunks"]])

        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=state["query"],
            actual_output=state["generated_answer"],
            retrieval_context=[context]
        )

        # Use the configured evaluation LLM backend
        evaluation_llm = get_eval_llm_instance()
        
        metric = FaithfulnessMetric(
            threshold=GROUNDING_THRESHOLD,
            model=evaluation_llm
        )

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


async def run_rag_workflow(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True
) -> Dict[str, Any]:
    """Execute the RAG workflow for a given query."""
    initial_state = RAGState({
        "query": query,
        "hyde_query": "",
        "chunks": [],
        "top_k": top_k,
        "use_reranker": use_reranker,
        "generated_answer": "",
        "grounding_score": 0.0,
        "retry_count": 0,
        "error": None
    })

    try:
        final_state = await rag_graph.ainvoke(initial_state)
        return {
            "answer": final_state["generated_answer"],
            "sources": final_state["chunks"],
            "grounding_score": final_state["grounding_score"],
            "retries": final_state["retry_count"] - 1 if final_state["retry_count"] > 0 else 0
        }
    except Exception as e:
        import traceback

        log.error("Query failed: %s\n%s", e, traceback.format_exc())

        raise e
