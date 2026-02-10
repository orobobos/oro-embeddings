"""Tests for embedding provider configuration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from our_embeddings.config import clear_config_cache
from our_embeddings.service import (
    EmbeddingProvider,
    generate_embedding,
    generate_local_embedding,
    get_embedding_provider,
)


class TestEmbeddingProvider:
    def test_openai_provider(self):
        assert EmbeddingProvider.OPENAI == "openai"

    def test_local_provider(self):
        assert EmbeddingProvider.LOCAL == "local"


class TestGetEmbeddingProvider:
    def test_default_is_local(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VALENCE_EMBEDDING_PROVIDER", None)
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL
            finally:
                clear_config_cache()

    def test_openai_from_env(self):
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "openai"}):
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.OPENAI
            finally:
                clear_config_cache()

    def test_local_from_env(self):
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "local"}):
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL
            finally:
                clear_config_cache()

    def test_case_insensitive(self):
        test_cases = ["LOCAL", "Local", "LOCAL"]
        for value in test_cases:
            with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": value}):
                clear_config_cache()
                try:
                    provider = get_embedding_provider()
                    assert provider == EmbeddingProvider.LOCAL
                finally:
                    clear_config_cache()

    def test_unknown_defaults_to_local(self):
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "unknown-provider"}):
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL
            finally:
                clear_config_cache()


class TestLocalEmbedding:
    @pytest.fixture
    def mock_local_model(self):
        import numpy as np

        from our_embeddings.providers import local

        mock_model = MagicMock()
        normalized_vec = np.random.randn(384).astype(np.float32)
        normalized_vec = normalized_vec / np.linalg.norm(normalized_vec)
        mock_model.encode.return_value = normalized_vec
        mock_model.get_sentence_embedding_dimension.return_value = 384

        local._model = mock_model
        yield mock_model
        local.reset_model()

    def test_local_embedding_returns_384_dimensions(self, mock_local_model):
        result = generate_local_embedding("test text")
        assert len(result) == 384

    def test_local_embedding_calls_encode(self, mock_local_model):
        generate_local_embedding("test text")
        mock_local_model.encode.assert_called_once()


class TestGenerateEmbedding:
    @pytest.fixture
    def mock_openai(self):
        with patch("our_embeddings.service.get_openai_client") as mock:
            client = MagicMock()
            mock.return_value = client
            response = MagicMock()
            response.data = [MagicMock(embedding=[0.1] * 1536)]
            client.embeddings.create.return_value = response
            yield client

    @pytest.fixture
    def mock_local_model(self):
        import numpy as np

        from our_embeddings.providers import local

        mock_model = MagicMock()
        normalized_vec = np.random.randn(384).astype(np.float32)
        normalized_vec = normalized_vec / np.linalg.norm(normalized_vec)
        mock_model.encode.return_value = normalized_vec
        mock_model.get_sentence_embedding_dimension.return_value = 384

        local._model = mock_model
        yield mock_model
        local.reset_model()

    def test_uses_local_by_default(self, mock_local_model):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VALENCE_EMBEDDING_PROVIDER", None)
            result = generate_embedding("test text")
            assert len(result) == 384
            mock_local_model.encode.assert_called_once()

    def test_explicit_openai_provider(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            result = generate_embedding("test text", provider=EmbeddingProvider.OPENAI)
            assert len(result) == 1536

    def test_explicit_local_provider(self, mock_local_model):
        result = generate_embedding("test", provider=EmbeddingProvider.LOCAL)
        assert len(result) == 384

    def test_truncates_long_text(self, mock_openai):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            long_text = "a" * 10000
            generate_embedding(long_text, provider=EmbeddingProvider.OPENAI)
            call_args = mock_openai.embeddings.create.call_args
            sent_text = call_args.kwargs.get("input") or call_args[1].get("input")
            assert len(sent_text) == 8000
