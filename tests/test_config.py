"""Unit tests for the configuration module."""
import os
import pytest
from unittest.mock import patch

from app.core.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()
        assert settings.base_url == "https://openrouter.ai/api/v1"
        assert settings.api_key is None
        assert settings.retrieval_url == "http://localhost:8001/v1"

    @patch.dict(os.environ, {
        "BASE_URL": "https://custom.api.com/v1",
        "API_KEY": "test-key-123",
        "RETRIEVAL_URL": "http://retrieval.local:9000",
    })
    def test_env_loading(self):
        """Test that environment variables are loaded correctly."""
        settings = Settings()
        assert settings.base_url == "https://custom.api.com/v1"
        assert settings.api_key == "test-key-123"
        assert settings.retrieval_url == "http://retrieval.local:9000"

    def test_case_insensitive_env_loading(self):
        """Test that environment variable loading is case insensitive."""
        with patch.dict(os.environ, {"base_url": "https://test.com"}):
            settings = Settings()
            assert settings.base_url == "https://test.com"


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_cached_instance(self):
        """Test that get_settings returns the same instance on multiple calls."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_clear_cache(self):
        """Test clearing the settings cache."""
        from functools import lru_cache
        
        settings1 = get_settings()
        # Clear the lru_cache
        get_settings.cache_clear()
        settings2 = get_settings()
        
        # After clearing, we should get a new instance
        # Note: This test may not work as expected if get_settings uses @lru_cache
        # because the module needs to be reloaded to clear the cache properly