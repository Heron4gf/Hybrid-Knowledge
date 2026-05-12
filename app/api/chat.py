"""Chat completions API router with retrieval injection."""
import logging
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


async def query_retrieval(query: str, top_k: int = 5) -> RetrievalQueryResponse:
    """Query the KI-4-KMU retrieval service.

    Raises HTTPException on any failure — retrieval is mandatory in the pipeline.
    """
    settings = get_settings()
    retrieval_url = settings.retrieval_url.rstrip("/")

    logger.debug(f"Querying retrieval at {retrieval_url}/query with: {query!r}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{retrieval_url}/query",
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
                status_code=502,
                detail=f"Failed to reach retrieval service ({retrieval_url}): {str(e)}",
            )


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(request: Request):
    """Create a chat completion following the hybrid retrieval pipeline:

    1. Extract query from last user message
    2. Query the KI-4-KMU retrieval service (mandatory)
    3. Inject retrieved context as a system message
    4. Forward enriched messages to the upstream LLM
    5. Return the response (streaming or standard)
    """
    settings = get_settings()

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

    messages: List[Dict[str, Any]] = body.get("messages", [])
    model: str = body.get("model", "")
    stream: bool = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="messages field is required")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")

    # Step 1: Extract query from last user message
    query = extract_query_from_messages(messages)

    if query:
        # Step 2 & 3: Retrieve context and inject — mandatory per pipeline architecture
        top_k = body.get("top_k", 5)
        retrieval_response = await query_retrieval(query, top_k)

        if retrieval_response.results:
            context_msg = build_context_message(retrieval_response)
            body["messages"] = inject_context_message(messages, context_msg)
            logger.debug(f"Injected {len(retrieval_response.results)} retrieval results into context")
        else:
            logger.debug("Retrieval returned no results, proceeding without context injection")

    # Step 4: Forward to upstream LLM
    upstream_headers = {
        "Content-Type": "application/json",
        "Accept": request.headers.get("accept", "application/json"),
    }
    if settings.api_key:
        upstream_headers["Authorization"] = f"Bearer {settings.api_key}"

    base_url = settings.base_url.rstrip("/")
    logger.debug(f"Forwarding to upstream: {base_url}/chat/completions")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            upstream_response = await client.post(
                f"{base_url}/chat/completions",
                json=body,
                headers=upstream_headers,
            )
            upstream_response.raise_for_status()

            # Step 5: Return response
            if stream:
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
