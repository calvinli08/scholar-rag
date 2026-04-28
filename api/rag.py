import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from api.retriever import retrieve_chunks
from ingestion.embedder import get_embedder
from api.api_logger import APICallLogger, _is_hosted_backend
from config import settings, ModelBackend
from logger import get_logger, configure_logging

configure_logging()

log = get_logger(__name__)
from api.api_logger import APICallLogger, _is_hosted_backend
from config import settings, ModelBackend
from logger import get_logger, configure_logging

configure_logging()

log = get_logger(__name__)


def get_llm_instance():
    """Initialize the LLM based on settings.model_backend."""
    backend = settings.model_backend

    # Create callback handler for hosted backends
    callbacks = []
    if _is_hosted_backend(backend.value):
        callbacks.append(APICallLogger(backend=backend.value))

    if backend == ModelBackend.OPENAI:
    """Initialize the LLM based on settings.model_backend."""
    backend = settings.model_backend

    # Create callback handler for hosted backends
    callbacks = []
    if _is_hosted_backend(backend.value):
        callbacks.append(APICallLogger(backend=backend.value))

    if backend == ModelBackend.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            callbacks=callbacks,
            api_key=settings.openai_api_key,
            callbacks=callbacks,
        )
    elif backend == ModelBackend.QWEN:
        # Qwen models hosted on vLLM server - uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.qwen_model,
    elif backend == ModelBackend.QWEN:
        # Qwen models hosted on vLLM server - uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.qwen_model,
            temperature=settings.llm_temperature,
            base_url=settings.qwen_url,
            api_key="vllm",
            callbacks=callbacks,
            base_url=settings.qwen_url,
            api_key="vllm",
            callbacks=callbacks,
        )
    elif backend == ModelBackend.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
    elif backend == ModelBackend.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            api_key=settings.gemini_api_key,
            callbacks=callbacks,
            api_key=settings.gemini_api_key,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")


def get_eval_llm_instance():
    """Initialize the LLM for evaluation based on settings.model_backend using DeepEval native models.
    
    Uses the same configured LLM as the main RAG pipeline (no separate eval model settings).
    """
    backend = settings.model_backend

    if backend == ModelBackend.OPENAI:
        from deepeval.models import GPTModel
        return GPTModel(
            model=settings.openai_model,
            api_key=settings.openai_api_key
        )
    elif backend == ModelBackend.QWEN:
        # Qwen models hosted on vLLM server - uses OpenAI-compatible API for DeepEval
        from deepeval.models import DeepEvalBaseLLM

        class QWENWrapper(DeepEvalBaseLLM):
            def __init__(self, model: str, base_url: str) -> None:
                self._model = model
                self._base_url = base_url

            def load_model(self):
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self._model,
                    base_url=self._base_url,
                    api_key="vllm",
                )

            def generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                from langchain_core.messages import HumanMessage
                response = chat_model.invoke([HumanMessage(content=prompt)])
                return response.content

            async def a_generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                from langchain_core.messages import HumanMessage
                response = await chat_model.ainvoke([HumanMessage(content=prompt)])
                return response.content

            def get_model_name(self) -> str:
                return self._model

        return QWENWrapper(
            model=settings.qwen_model,
            base_url=settings.qwen_url
        )
    elif backend == ModelBackend.GEMINI:
        from deepeval.models import GeminiModel
        return GeminiModel(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported evaluation backend: {backend}")


# Configuration
GROUNDING_THRESHOLD = 0.9
MAX_RETRIES = 2


def get_eval_llm_instance():
    """Initialize the LLM for evaluation based on settings.model_backend using DeepEval native models.
    
    Uses the same configured LLM as the main RAG pipeline (no separate eval model settings).
    """
    backend = settings.model_backend

    if backend == ModelBackend.OPENAI:
        from deepeval.models import GPTModel
        return GPTModel(
            model=settings.openai_model,
            api_key=settings.openai_api_key
        )
    elif backend == ModelBackend.QWEN:
        # Qwen models hosted on vLLM server - uses OpenAI-compatible API for DeepEval
        from deepeval.models import DeepEvalBaseLLM

        class QWENWrapper(DeepEvalBaseLLM):
            def __init__(self, model: str, base_url: str) -> None:
                self._model = model
                self._base_url = base_url

            def load_model(self):
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self._model,
                    base_url=self._base_url,
                    api_key="vllm",
                )

            def generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                from langchain_core.messages import HumanMessage
                response = chat_model.invoke([HumanMessage(content=prompt)])
                return response.content

            async def a_generate(self, prompt: str) -> str:
                chat_model = self.load_model()
                from langchain_core.messages import HumanMessage
                response = await chat_model.ainvoke([HumanMessage(content=prompt)])
                return response.content

            def get_model_name(self) -> str:
                return self._model

        return QWENWrapper(
            model=settings.qwen_model,
            base_url=settings.qwen_url
        )
    elif backend == ModelBackend.GEMINI:
        from deepeval.models import GeminiModel
        return GeminiModel(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported evaluation backend: {backend}")


# Configuration
GROUNDING_THRESHOLD = 0.9
MAX_RETRIES = 2


class RAGState(TypedDict):
    query: str
    hyde_query: str
    chunks: List[Dict[str, Any]]
    generated_answer: str
    grounding_score: float
    eval_feedback: str
    supervisor_decision: str
    top_k: int
    use_reranker: bool
    retry_count: int
    error: Optional[str]


def build_rag_graph():
    llm = get_llm_instance()

    def hyde_node(state: RAGState) -> Dict[str, Any]:
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant. Write a hypothetical answer to the following question based on your internal knowledge. This will be used to retrieve relevant documents."),
            ("human", "{query}")
        ])

        chain = hyde_prompt | llm
        response = chain.invoke({"query": state["query"]})


        return {"hyde_query": response.content}

    async def retrieve_node(state: RAGState) -> Dict[str, Any]:
        query_text = state.get("hyde_query") or state["query"]

        chunks = await retrieve_chunks(
            query_text,
            top_k=state["top_k"],
            use_reranker=state["use_reranker"]
        )


        chunks = await retrieve_chunks(
            query_text,
            top_k=state["top_k"],
            use_reranker=state["use_reranker"]
        )

        return {"chunks": chunks}

    def generation_agent(state: RAGState) -> Dict[str, Any]:
        context = "\n\n".join(
            [f"[Source {i+1}]: {c.text}" for i, c in enumerate(state["chunks"])]
        )

        feedback = ""
        if state.get("eval_feedback"):
            feedback = f"\n\nSupervisor feedback from prior evaluation:\n{state['eval_feedback']}"

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the Generation Agent in a multi-agent RAG system. "
                    "Answer the user's question using ONLY the provided context. "
                    "Cite sources using [Source X]. "
                    "If the answer is not supported by the context, say you do not know."
                    f"{feedback}"
                ),
                (
                    "human", 
                    "Context:\n{context}\n\nQuestion: {query}"
                )
            ],
            template_format="mustache"
        )

        chain = prompt | llm
        response = chain.invoke({"context": context, "query": state["query"]})

        return {
            "generated_answer": response.content,
            "retry_count": state["retry_count"] + 1
        }

    def deepeval_agent(state: RAGState) -> Dict[str, Any]:
        if not state["chunks"]:
            return {
                "grounding_score": 0.0,
                "eval_feedback": "No retrieval context was available, so the answer could not be grounded."
            }

        context = "\n".join([c.text for c in state["chunks"]])

        test_case = LLMTestCase(
            input=state["query"],
            actual_output=state["generated_answer"],
            retrieval_context=[context]
        )

        evaluation_llm = get_eval_llm_instance()

        metric = FaithfulnessMetric(
            threshold=GROUNDING_THRESHOLD,
            model=evaluation_llm
        )

        try:
            metric.measure(test_case)

            score = metric.score
            feedback = metric.reason[:1000]  # Limit feedback length for prompt
        except Exception as e:
            log.error("Evaluation error: %s", e)
            score = 0.0
            feedback = "Evaluation failed with error"

        return {
            "grounding_score": score,
            "eval_feedback": feedback
        }

    def supervisor_agent(state: RAGState) -> Dict[str, Any]:
        if state["retry_count"] >= MAX_RETRIES:
            return {
                "supervisor_decision": "end",
                "eval_feedback": (
                    state["eval_feedback"]
                    + " Maximum retries reached. Returning best available answer."
                )
            }

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the supervisor agent in a multi-agent RAG system. "
                "Before you were invoked, an evaluation agent assessed the RAG-generated answer for faithfulness and produced a numeric grounding score and feedback. "
                "Examine both the numeric faithfulness score and the feedback. Determine whether the score is high enough and whether the justification is strong enough to terminate the workflow, or whether to re-generate the answer. "
                "The numeric faithfulness score is a float number between 0 and 1, where a higher score indicates the answer is more faithful to the retrieved context. A score above 0.9 is generally considered good, but also consider the feedback justification provided by the evaluation agent."
            ),
            (
                "human", 
                "The numeric faithfulness score is: {grounding_score}. The evaluation agent feedback is: {eval_feedback}. "
                "Based on the above, should the generation agent try to generate a new answer (type 'generate') or is the current answer good enough to return to the user (type 'end')? "
                "Respond with only 'generate' or 'end'."
            )
        ])

        chain = prompt | llm
        response = chain.invoke({"grounding_score": state["grounding_score"], "eval_feedback": state["eval_feedback"]})

        return {
            "supervisor_decision": response.content.strip().lower()
        }

    def route_supervisor(state: RAGState) -> str:
        return state["supervisor_decision"]

    workflow = StateGraph(RAGState)

    workflow.add_node("hyde", hyde_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generation_agent", generation_agent)
    workflow.add_node("deepeval_agent", deepeval_agent)
    workflow.add_node("supervisor_agent", supervisor_agent)

    workflow.set_entry_point("hyde")

    workflow.add_edge("hyde", "retrieve")
    workflow.add_edge("retrieve", "generation_agent")
    workflow.add_edge("generation_agent", "deepeval_agent")
    workflow.add_edge("deepeval_agent", "supervisor_agent")

    workflow.add_conditional_edges(
        "supervisor_agent",
        route_supervisor,
        {
            "generate": "generation_agent",
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
async def run_rag_workflow(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True
) -> Dict[str, Any]:
    """Execute the RAG workflow for a given query."""
    initial_state = RAGState({
    initial_state = RAGState({
        "query": query,
        "hyde_query": "",
        "chunks": [],
        "top_k": top_k,
        "use_reranker": use_reranker,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "generated_answer": "",
        "grounding_score": 0.0,
        "retry_count": 0,
        "error": None
    })

    })

    try:
        final_state = await rag_graph.ainvoke(initial_state)
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
        import traceback

        log.error("Query failed: %s\n%s", e, traceback.format_exc())

        raise e
