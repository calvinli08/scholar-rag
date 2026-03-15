"""
LLM generation module for answer synthesis.
Supports multiple backends: Ollama (local), HuggingFace, vLLM, OpenAI, Anthropic.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings, LLMBackend
from logger import get_logger
from models import RetrievedChunk, QueryResult

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_llm():
    """Get the configured LLM client/model."""
    backend = settings.llm_backend
    
    if backend == LLMBackend.OLLAMA:
        log.info("Using Ollama LLM backend: %s", settings.ollama_model)
        return "ollama"
    
    elif backend == LLMBackend.HF:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        log.info("Loading HF LLM model: %s", settings.hf_llm_model)
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_llm_model)
        model = AutoModelForCausalLM.from_pretrained(
            settings.hf_llm_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        return {"tokenizer": tokenizer, "model": model}
    
    elif backend == LLMBackend.VLLM:
        from openai import OpenAI
        log.info("Using vLLM backend at %s", settings.vllm_url)
        # vLLM provides OpenAI-compatible API
        return OpenAI(base_url=settings.vllm_url, api_key="not-needed")
    
    elif backend == LLMBackend.OPENAI:
        from openai import OpenAI
        log.info("Using OpenAI LLM backend")
        return OpenAI(api_key=settings.openai_api_key)
    
    elif backend == LLMBackend.ANTHROPIC:
        import anthropic
        log.info("Using Anthropic LLM backend")
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    use_hyde: bool = True,
) -> QueryResult:
    """
    Generate an answer to a query using retrieved context.
    
    Args:
        query: User's question
        chunks: Retrieved chunks to use as context
        use_hyde: Whether to use HyDE for query expansion (not yet implemented)
        
    Returns:
        QueryResult with answer and token counts
    """
    llm_client = _get_llm()
    backend = settings.llm_backend
    
    # Build context from chunks
    context = _build_context(chunks)
    
    # Build prompt
    prompt = _build_prompt(query, context)
    
    # Generate answer based on backend
    if backend == LLMBackend.OLLAMA:
        answer, prompt_tokens, completion_tokens = await _generate_ollama(llm_client, prompt)
        
    elif backend == LLMBackend.HF:
        answer, prompt_tokens, completion_tokens = _generate_hf(llm_client, prompt)
        
    elif backend == LLMBackend.VLLM:
        answer, prompt_tokens, completion_tokens = await _generate_vllm(llm_client, prompt)
        
    elif backend == LLMBackend.OPENAI:
        answer, prompt_tokens, completion_tokens = await _generate_openai(llm_client, prompt)
        
    elif backend == LLMBackend.ANTHROPIC:
        answer, prompt_tokens, completion_tokens = await _generate_anthropic(llm_client, prompt)
        
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
    
    return QueryResult(
        query=query,
        answer=answer,
        sources=chunks,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}] (Section: {chunk.chunk.section})\n{chunk.text}\n")
    return "\n---\n".join(context_parts)


def _build_prompt(query: str, context: str) -> str:
    """Build the final prompt for the LLM."""
    system_prompt = """You are a research assistant answering questions based on academic papers.
Use the provided context to answer the question accurately. Cite sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly. Do not make up facts."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    return f"{system_prompt}\n\n{user_prompt}"


async def _generate_ollama(client, prompt: str) -> tuple[str, int, int]:
    """Generate using Ollama."""
    import httpx
    
    response = await httpx.AsyncClient().post(
        f"{settings.ollama_llm_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        },
    )
    response.raise_for_status()
    data = response.json()
    
    return (
        data["response"],
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
    )


def _generate_hf(client, prompt: str) -> tuple[str, int, int]:
    """Generate using HuggingFace transformers."""
    import torch
    
    tokenizer = client["tokenizer"]
    model = client["model"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            do_sample=settings.llm_temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0, prompt_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return answer, prompt_length, len(generated_tokens)


async def _generate_vllm(client, prompt: str) -> tuple[str, int, int]:
    """Generate using vLLM (OpenAI-compatible API)."""
    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


async def _generate_openai(client, prompt: str) -> tuple[str, int, int]:
    """Generate using OpenAI API."""
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
    
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


async def _generate_anthropic(client, prompt: str) -> tuple[str, int, int]:
    """Generate using Anthropic API."""
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.llm_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    
    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
