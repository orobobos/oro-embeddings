"""Shared test fixtures for oro-embeddings."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from oro_embeddings.config import clear_config_cache


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (require external services)")
    config.addinivalue_line("markers", "slow: Slow tests (>5s)")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all VKB_ and VALENCE_ environment variables."""
    env_prefixes = ("VKB_", "VALENCE_", "OPENAI_")
    for key in list(os.environ.keys()):
        if any(key.startswith(prefix) for prefix in env_prefixes):
            monkeypatch.delenv(key, raising=False)
    clear_config_cache()
    yield  # type: ignore[misc]
    clear_config_cache()


@pytest.fixture
def env_with_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    clear_config_cache()
    yield  # type: ignore[misc]
    clear_config_cache()


@pytest.fixture
def mock_openai() -> MagicMock:
    """Mock OpenAI client for embedding generation."""
    with patch("oro_embeddings.service.OpenAI") as mock_class:
        mock_client = MagicMock()
        mock_class.return_value = mock_client

        # Mock embedding response
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        yield mock_client
