"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "developer", "content": "You specialize in engine specifications."},
        {"role": "user", "content": "What is the compression ratio of the engine?"},
        {"role": "assistant", "content": "The compression ratio is 10:1."},
    ]


@pytest.fixture
def sample_retrieval_response():
    """Fixture providing sample retrieval response for testing."""
    return {
        "query": "engine compression ratio",
        "results": [
            {
                "id": "doc1_chunk1",
                "text": "The engine has a compression ratio of 10:1",
                "score": 0.95,
                "metadata": {
                    "chunk_id": "doc1_chunk1",
                    "section_id": "section1"
                }
            },
            {
                "id": "doc2_chunk3",
                "text": "Compression ratio affects fuel efficiency",
                "score": 0.85,
                "metadata": {
                    "chunk_id": "doc2_chunk3",
                    "section_id": "section2"
                }
            }
        ]
    }


@pytest.fixture
def mock_settings():
    """Fixture providing mock settings for testing."""
    class MockSettings:
        base_url = "https://openrouter.ai/api/v1"
        api_key = "test-key"
        retrieval_url = "http://localhost:8001/v1"
    
    return MockSettings()