"""Pydantic models for the retrieval API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievalQueryRequest(BaseModel):
    """Request body for the retrieval query endpoint."""

    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class RetrievalResultMetadata(BaseModel):
    """Metadata associated with a retrieval result."""

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    section_id: str = Field(..., description="Identifier for the parent section")


class RetrievalQueryResult(BaseModel):
    """A single result from the retrieval query."""

    id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="The retrieved text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: RetrievalResultMetadata = Field(..., description="Result metadata")


class RetrievalQueryResponse(BaseModel):
    """Response from the retrieval query endpoint."""

    query: str = Field(..., description="The original query text")
    results: List[RetrievalQueryResult] = Field(default_factory=list, description="List of retrieved results")