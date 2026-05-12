"""Models listing API router."""
import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.config import get_settings

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict:
    """List available models by forwarding to upstream API.
    
    Returns:
        Model list response from upstream
    """
    settings = get_settings()
    
    headers = dict(request.headers)
    headers.pop("host", None)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{settings.base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
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