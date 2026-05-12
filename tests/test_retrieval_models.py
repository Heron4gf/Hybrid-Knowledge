"""Unit tests for the retrieval models."""
import pytest
from pydantic import ValidationError

from app.models.retrieval import (
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RetrievalQueryResult,
    RetrievalResultMetadata,
)


class TestRetrievalQueryRequest:
    """Tests for RetrievalQueryRequest model."""

    def test_valid_request_with_defaults(self):
        """Test creating request with only required fields."""
        request = RetrievalQueryRequest(query="test query")
        assert request.query == "test query"
        assert request.top_k == 5  # default value

    def test_valid_request_with_custom_top_k(self):
        """Test creating request with custom top_k."""
        request = RetrievalQueryRequest(query="test query", top_k=10)
        assert request.top_k == 10

    def test_request_validation_error_on_invalid_top_k(self):
        """Test that invalid top_k raises validation error."""
        with pytest.raises(ValidationError):
            RetrievalQueryRequest(query="test", top_k=0)  # below minimum
        
        with pytest.raises(ValidationError):
            RetrievalQueryRequest(query="test", top_k=25)  # above maximum

    def test_request_empty_query(self):
        """Test that empty query is allowed (validation at API level)."""
        request = RetrievalQueryRequest(query="")
        assert request.query == ""


class TestRetrievalResultMetadata:
    """Tests for RetrievalResultMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = RetrievalResultMetadata(
            chunk_id="chunk123",
            section_id="section456",
        )
        assert metadata.chunk_id == "chunk123"
        assert metadata.section_id == "section456"


class TestRetrievalQueryResult:
    """Tests for RetrievalQueryResult model."""

    def test_valid_result(self):
        """Test creating valid retrieval result."""
        result = RetrievalQueryResult(
            id="doc1_chunk1",
            text="The engine has a compression ratio of 10:1",
            score=0.95,
            metadata=RetrievalResultMetadata(
                chunk_id="doc1_chunk1",
                section_id="section1",
            ),
        )
        assert result.id == "doc1_chunk1"
        assert result.text == "The engine has a compression ratio of 10:1"
        assert result.score == 0.95

    def test_result_score_boundaries(self):
        """Test score validation boundaries."""
        # Valid boundaries
        result_min = RetrievalQueryResult(
            id="test",
            text="test",
            score=0.0,
            metadata=RetrievalResultMetadata(chunk_id="c", section_id="s"),
        )
        assert result_min.score == 0.0

        result_max = RetrievalQueryResult(
            id="test",
            text="test",
            score=1.0,
            metadata=RetrievalResultMetadata(chunk_id="c", section_id="s"),
        )
        assert result_max.score == 1.0

    def test_result_invalid_score(self):
        """Test that score outside 0-1 range raises error."""
        with pytest.raises(ValidationError):
            RetrievalQueryResult(
                id="test",
                text="test",
                score=1.5,  # above 1.0
                metadata=RetrievalResultMetadata(chunk_id="c", section_id="s"),
            )


class TestRetrievalQueryResponse:
    """Tests for RetrievalQueryResponse model."""

    def test_valid_response_empty_results(self):
        """Test creating response with empty results."""
        response = RetrievalQueryResponse(
            query="test query",
            results=[],
        )
        assert response.query == "test query"
        assert response.results == []

    def test_valid_response_with_results(self):
        """Test creating response with multiple results."""
        response = RetrievalQueryResponse(
            query="engine specs",
            results=[
                RetrievalQueryResult(
                    id="doc1",
                    text="Result 1",
                    score=0.9,
                    metadata=RetrievalResultMetadata(chunk_id="c1", section_id="s1"),
                ),
                RetrievalQueryResult(
                    id="doc2",
                    text="Result 2",
                    score=0.8,
                    metadata=RetrievalResultMetadata(chunk_id="c2", section_id="s2"),
                ),
            ],
        )
        assert len(response.results) == 2
        assert response.results[0].id == "doc1"
        assert response.results[1].id == "doc2"