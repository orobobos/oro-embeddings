"""Tests for local embedding provider (sentence-transformers/bge-small-en-v1.5)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oro_embeddings.config import clear_config_cache


class TestIsLocalPath:
    def test_absolute_path_detected(self, tmp_path):
        from oro_embeddings.providers import local

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        assert local._is_local_path(str(model_dir)) is True

    def test_relative_path_detected(self):
        from oro_embeddings.providers import local

        assert local._is_local_path("./models/bge") is True
        assert local._is_local_path("../models/bge") is True

    def test_home_path_detected(self):
        from oro_embeddings.providers import local

        assert local._is_local_path("~/models/bge") is True

    def test_huggingface_model_name_not_local(self):
        from oro_embeddings.providers import local

        assert local._is_local_path("BAAI/bge-small-en-v1.5") is False
        assert local._is_local_path("sentence-transformers/all-MiniLM-L6-v2") is False


class TestGetModel:
    def test_lazy_loading(self):
        from oro_embeddings.providers import local

        local.reset_model()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
            model = local.get_model()
            mock_st.assert_called_once()
            assert model is mock_model

    def test_reuses_cached_model(self):
        from oro_embeddings.providers import local

        mock_model = MagicMock()
        local._model = mock_model
        result = local.get_model()
        assert result is mock_model

    def test_respects_device_env(self):
        from oro_embeddings.providers import local

        local.reset_model()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_DEVICE": "cuda"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()
                    mock_st.assert_called_once()
                    call_kwargs = mock_st.call_args
                    assert call_kwargs[1]["device"] == "cuda"
            finally:
                clear_config_cache()

    def test_respects_model_path_env(self):
        from oro_embeddings.providers import local

        local.reset_model()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        custom_model = "custom/model-path"

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": custom_model}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()
                    mock_st.assert_called_once()
                    call_args = mock_st.call_args[0]
                    assert call_args[0] == custom_model
            finally:
                clear_config_cache()


class TestGenerateEmbedding:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        normalized_vec = np.random.randn(384).astype(np.float32)
        normalized_vec = normalized_vec / np.linalg.norm(normalized_vec)
        model.encode.return_value = normalized_vec
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    def test_returns_list_of_floats(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        result = local.generate_embedding("test text")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_returns_384_dimensions(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        result = local.generate_embedding("test text")
        assert len(result) == 384

    def test_l2_normalized(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        result = local.generate_embedding("test text")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.001

    def test_calls_encode_with_normalize(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        local.generate_embedding("test text")
        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True


class TestGenerateEmbeddingsBatch:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()

        def mock_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.random.randn(n, 384).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        model.encode.side_effect = mock_encode
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    def test_returns_list_of_embeddings(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        texts = ["text 1", "text 2", "text 3"]
        result = local.generate_embeddings_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_embedding_384_dimensions(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        result = local.generate_embeddings_batch(["text 1", "text 2"])
        for emb in result:
            assert len(emb) == 384

    def test_respects_batch_size(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        local.generate_embeddings_batch(["text"] * 10, batch_size=5)
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 5

    def test_shows_progress_for_large_batches(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        local.generate_embeddings_batch(["text"] * 150)
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is True

    def test_no_progress_for_small_batches(self, mock_model):
        from oro_embeddings.providers import local

        local._model = mock_model
        local.generate_embeddings_batch(["text"] * 50)
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is False


class TestServiceIntegration:
    def test_local_provider_default(self):
        from oro_embeddings.service import EmbeddingProvider, get_embedding_provider

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VALENCE_EMBEDDING_PROVIDER", None)
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL
            finally:
                clear_config_cache()

    def test_openai_provider_override(self):
        from oro_embeddings.service import EmbeddingProvider, get_embedding_provider

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "openai"}):
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.OPENAI
            finally:
                clear_config_cache()

    def test_generate_embedding_uses_local(self):
        from oro_embeddings import service
        from oro_embeddings.providers import local

        mock_embedding = [0.1] * 384
        with patch.object(local, "_model", MagicMock()):
            with patch("oro_embeddings.providers.local.generate_embedding", return_value=mock_embedding):
                with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "local"}):
                    result = service.generate_embedding("test", provider=service.EmbeddingProvider.LOCAL)
                    assert result == mock_embedding


class TestResetModel:
    def test_clears_cached_model(self):
        from oro_embeddings.providers import local

        local._model = MagicMock()
        local.reset_model()
        assert local._model is None


class TestOfflineSupport:
    def test_loads_from_local_path(self, tmp_path):
        from oro_embeddings.providers import local

        local.reset_model()
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": str(model_dir)}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()
                    mock_st.assert_called_once()
                    call_args = mock_st.call_args[0]
                    assert str(model_dir) in call_args[0]
            finally:
                clear_config_cache()

    def test_error_for_missing_local_path(self, tmp_path):
        from oro_embeddings.providers import local
        from oro_embeddings.providers.local import ModelLoadError

        local.reset_model()
        nonexistent_path = tmp_path / "does-not-exist"

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": str(nonexistent_path)}):
            clear_config_cache()
            try:
                with pytest.raises(ModelLoadError) as exc_info:
                    local.get_model()
                error_msg = str(exc_info.value)
                assert "not found" in error_msg.lower()
                assert "VALENCE_EMBEDDING_MODEL_PATH" in error_msg
            finally:
                clear_config_cache()

    def test_error_for_network_failure(self):
        from oro_embeddings.providers import local
        from oro_embeddings.providers.local import ModelLoadError

        local.reset_model()
        network_error = OSError("Connection error: could not resolve hostname")

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": "BAAI/bge-small-en-v1.5"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", side_effect=network_error):
                    with pytest.raises(ModelLoadError) as exc_info:
                        local.get_model()
                    error_msg = str(exc_info.value)
                    assert "network unavailable" in error_msg.lower() or "cannot download" in error_msg.lower()
            finally:
                clear_config_cache()

    def test_model_load_error_is_exception(self):
        from oro_embeddings.providers.local import ModelLoadError

        assert issubclass(ModelLoadError, Exception)
        error = ModelLoadError("test message")
        assert str(error) == "test message"

    def test_huggingface_name_not_treated_as_local(self):
        from oro_embeddings.providers import local

        local.reset_model()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": "BAAI/bge-small-en-v1.5"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()
                    call_args = mock_st.call_args[0]
                    assert call_args[0] == "BAAI/bge-small-en-v1.5"
            finally:
                clear_config_cache()
