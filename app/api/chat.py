"""Chat completions API router with retrieval injection."""
import json
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.models.retrieval import RetrievalQueryRequest, RetrievalQueryResponse
from app.utils.context_injector import (
    build_context_message,
    extract_query_from_messages,
    inject_context_message,
)

router = APIRouter(prefix="/v1", tags=["chat"])


async def query_retrieval(
    query: str,
    top_k: int = 5,
) -> RetrievalQueryResponse:
    """Query the retrieval API with the given text.
    
    Args:
        query: The search query
        top_k: Number of results to retrieve
        
    Returns:
        RetrievalQueryResponse with results
        
    Raises:
        HTTPException: If the retrieval API call fails
    """
    settings = get_settings()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{settings.retrieval_url}/query",
                json=RetrievalQueryRequest(query=query, top_k=top_k).model_dump(),
            )
            response.raise_for_status()
            return RetrievalQueryResponse(**response.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Retrieval API error: {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query retrieval API: {str(e)}",
            )


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(request: Request):
    """Create a chat completion with retrieval-augmented context injection.
    
    1. Extracts query from the last user message
    2. Queries the retrieval API
    3. Injects retrieval results as a 'context' message
    4. Forwards the modified request to the upstream API
    5. Returns streamed or regular response
    """
    settings = get_settings()
    
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")
    
    messages = body.get("messages", [])
    model = body.get("model", "")
    stream = body.get("stream", False)
    
    if not messages:
        raise HTTPException(status_code=400, detail="messages field is required")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")
    
    # Step 1: Extract query from last user message
    query = extract_query_from_messages(messages)
    
    if query:
        # Step 2: Query the retrieval API
        top_k = body.get("top_k", 5)
        try:
            retrieval_response = await query_retrieval(query, top_k)
            
            if retrieval_response.results:
                # Step 3: Build and inject context message
                context_msg = build_context_message(retrieval_response)
                modified_messages = inject_context_message(messages, context_msg)
                body["messages"] = modified_messages
        except HTTPException:
            # If retrieval fails, continue without context injection
            pass
    
    # Step 4: Forward to upstream API
    headers = dict(request.headers)
    headers.pop("host", None)  # Avoid host header issues
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            upstream_response = await client.post(
                f"{settings.base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            upstream_response.raise_for_status()
            
            if stream:
                # Return streaming response
                async def event_generator():
                    async for line in upstream_response.aiter_lines():
                        if line:
                            yield f"{line}\n"
                
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    status_code=upstream_response.status_code,
                )
            else:
                return upstream_response.json()
                
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Upstream API error: {e.response.text}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to call upstream API: {str(e)}",
            )