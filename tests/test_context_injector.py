"""Unit tests for the context injector utility."""
import json
import pytest

from app.models.retrieval import RetrievalQueryResponse, RetrievalQueryResult, RetrievalResultMetadata
from app.utils.context_injector import (
    build_context_message,
    extract_query_from_messages,
    inject_context_message,
)


class TestExtractQueryFromMessages:
    """Tests for extract_query_from_messages function."""

    def test_extract_query_from_simple_user_message(self):
        """Test extracting query from a simple user message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the meaning of life?"},
        ]
        result = extract_query_from_messages(messages)
        assert result == "What is the meaning of life?"

    def test_extract_query_from_last_user_message(self):
        """Test extracting query from the last user message when multiple exist."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ]
        result = extract_query_from_messages(messages)
        assert result == "Second question"

    def test_extract_query_from_empty_messages(self):
        """Test extracting query from empty messages list."""
        result = extract_query_from_messages([])
        assert result == ""

    def test_extract_query_no_user_message(self):
        """Test extracting query when no user message exists."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"},
        ]
        result = extract_query_from_messages(messages)
        assert result == ""

    def test_extract_query_from_content_list(self):
        """Test extracting query from message with content as list."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Query from structured content"}
            ]},
        ]
        result = extract_query_from_messages(messages)
        assert result == "Query from structured content"

    def test_extract_query_strips_whitespace(self):
        """Test that extracted query is stripped of whitespace."""
        messages = [
            {"role": "user", "content": "  Query with spaces  "},
        ]
        result = extract_query_from_messages(messages)
        assert result == "Query with spaces"


class TestBuildContextMessage:
    """Tests for build_context_message function."""

    def test_build_context_message_with_results(self):
        """Test building context message with retrieval results."""
        retrieval_response = RetrievalQueryResponse(
            query="engine specs",
            results=[
                RetrievalQueryResult(
                    id="doc1_chunk1",
                    text="Engine compression ratio is 10:1",
                    score=0.95,
                    metadata=RetrievalResultMetadata(
                        chunk_id="doc1_chunk1",
                        section_id="section1"
                    ),
                ),
            ],
        )
        
        result = build_context_message(retrieval_response)
        
        assert result["role"] == "context"
        content = json.loads(result["content"])
        assert content["query"] == "engine specs"
        assert len(content["results"]) == 1
        assert content["results"][0]["id"] == "doc1_chunk1"
        assert content["results"][0]["text"] == "Engine compression ratio is 10:1"

    def test_build_context_message_empty_results(self):
        """Test building context message with empty results."""
        retrieval_response = RetrievalQueryResponse(
            query="no results query",
            results=[],
        )
        
        result = build_context_message(retrieval_response)
        
        assert result["role"] == "context"
        content = json.loads(result["content"])
        assert content["query"] == "no results query"
        assert content["results"] == []


class TestInjectContextMessage:
    """Tests for inject_context_message function."""

    def test_inject_context_in_empty_messages(self):
        """Test injecting context into empty messages list."""
        context_msg = {"role": "context", "content": "{}"}
        result = inject_context_message([], context_msg)
        assert result == [context_msg]

    def test_inject_context_after_system_message(self):
        """Test injecting context after system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        context_msg = {"role": "context", "content": "{}"}
        result = inject_context_message(messages, context_msg)
        
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "context"
        assert result[2]["role"] == "user"

    def test_inject_context_after_developer_message(self):
        """Test injecting context after developer message."""
        messages = [
            {"role": "developer", "content": "Developer instructions"},
            {"role": "user", "content": "Hello"},
        ]
        context_msg = {"role": "context", "content": "{}"}
        result = inject_context_message(messages, context_msg)
        
        assert len(result) == 3
        assert result[0]["role"] == "developer"
        assert result[1]["role"] == "context"
        assert result[2]["role"] == "user"

    def test_inject_context_after_both_system_and_developer(self):
        """Test injecting context after both system and developer messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "developer", "content": "Developer instructions"},
            {"role": "user", "content": "Hello"},
        ]
        context_msg = {"role": "context", "content": "{}"}
        result = inject_context_message(messages, context_msg)
        
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "developer"
        assert result[2]["role"] == "context"
        assert result[3]["role"] == "user"

    def test_inject_context_when_no_user_message(self):
        """Test injecting context when no user message exists."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Assistant response"},
        ]
        context_msg = {"role": "context", "content": "{}"}
        result = inject_context_message(messages, context_msg)
        
        # Should append at the end since no user message found
        assert len(result) == 3
        assert result[-1]["role"] == "context"

    def test_inject_context_does_not_mutate_original(self):
        """Test that inject_context_message does not mutate the original list."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        context_msg = {"role": "context", "content": "{}"}
        original_length = len(messages)
        
        inject_context_message(messages, context_msg)
        
        assert len(messages) == original_length