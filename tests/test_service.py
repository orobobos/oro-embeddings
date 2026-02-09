"""Tests for oro_embeddings.service module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from oro_embeddings.exceptions import EmbeddingError

# ============================================================================
# get_openai_client Tests
# ============================================================================


class TestGetOpenaiClient:
    """Tests for get_openai_client function."""

    def test_lazy_initialization(self, env_with_openai_key):
        """Should lazily initialize OpenAI client."""
        from oro_embeddings import service

        service._openai_client = None

        with patch.object(service, "OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = service.get_openai_client()

            mock_openai.assert_called_once()
            assert client is mock_client

    def test_reuses_existing_client(self, env_with_openai_key):
        """Should reuse existing client."""
        from oro_embeddings import service

        mock_client = MagicMock()
        service._openai_client = mock_client

        result = service.get_openai_client()

        assert result is mock_client

    def test_raises_without_api_key(self, clean_env):
        """Should raise ValueError without API key."""
        from oro_embeddings import service

        service._openai_client = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            service.get_openai_client()


# ============================================================================
# generate_embedding Tests
# ============================================================================


class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    def test_success_with_openai(self, env_with_openai_key, mock_openai):
        """Should generate embedding with OpenAI provider."""
        from oro_embeddings import service
        from oro_embeddings.service import EmbeddingProvider

        service._openai_client = mock_openai

        result = service.generate_embedding("test text", provider=EmbeddingProvider.OPENAI)

        assert len(result) == 1536
        mock_openai.embeddings.create.assert_called_once()

    def test_truncates_long_text(self, env_with_openai_key, mock_openai):
        """Should truncate text longer than 8000 chars."""
        from oro_embeddings import service
        from oro_embeddings.service import EmbeddingProvider

        service._openai_client = mock_openai

        long_text = "x" * 10000
        service.generate_embedding(long_text, provider=EmbeddingProvider.OPENAI)

        call_args = mock_openai.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 8000

    def test_uses_specified_model(self, env_with_openai_key, mock_openai):
        """Should use specified model."""
        from oro_embeddings import service
        from oro_embeddings.service import EmbeddingProvider

        service._openai_client = mock_openai

        service.generate_embedding("test", model="text-embedding-3-large", provider=EmbeddingProvider.OPENAI)

        call_args = mock_openai.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-large"


# ============================================================================
# vector_to_pgvector Tests
# ============================================================================


class TestVectorToPgvector:
    """Tests for vector_to_pgvector function."""

    def test_format_conversion(self):
        """Should convert to pgvector format."""
        from oro_embeddings.service import vector_to_pgvector

        vector = [0.1, 0.2, 0.3]
        result = vector_to_pgvector(vector)

        assert result == "[0.1,0.2,0.3]"

    def test_empty_vector(self):
        """Should handle empty vector."""
        from oro_embeddings.service import vector_to_pgvector

        result = vector_to_pgvector([])
        assert result == "[]"

    def test_float_precision(self):
        """Should preserve float precision."""
        from oro_embeddings.service import vector_to_pgvector

        vector = [0.123456789]
        result = vector_to_pgvector(vector)

        assert "0.123456789" in result


# ============================================================================
# embed_content Tests
# ============================================================================


class TestEmbedContent:
    """Tests for embed_content function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_embed_belief(self, env_with_openai_key, mock_openai, mock_get_cursor, monkeypatch):
        """Should embed belief content."""
        from oro_embeddings import service

        monkeypatch.setenv("VALENCE_EMBEDDING_PROVIDER", "openai")
        service._openai_client = mock_openai

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("belief", str(uuid4()), "test content")

            assert result["content_type"] == "belief"
            assert result["dimensions"] == 1536

    def test_embed_exchange(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should embed exchange content."""
        from oro_embeddings import service

        service._openai_client = mock_openai

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("exchange", str(uuid4()), "test")

            assert result["content_type"] == "exchange"

    def test_embed_pattern(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should embed pattern content."""
        from oro_embeddings import service

        service._openai_client = mock_openai

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("pattern", str(uuid4()), "test")

            assert result["content_type"] == "pattern"


# ============================================================================
# search_similar Tests
# ============================================================================


class TestSearchSimilar:
    """Tests for search_similar function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_search_all_types(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should search all content types by default."""
        from oro_embeddings import service

        service._openai_client = mock_openai

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.search_similar("test query")

            assert isinstance(result, list)
            assert mock_get_cursor.execute.call_count == 3

    def test_filter_by_content_type(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should filter by content type."""
        from oro_embeddings import service

        service._openai_client = mock_openai

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            service.search_similar("test", content_type="belief")

            assert mock_get_cursor.execute.call_count == 1

    def test_returns_sorted_results(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should return results sorted by similarity."""
        from oro_embeddings import service

        service._openai_client = mock_openai

        mock_get_cursor.fetchall.side_effect = [
            [{"id": uuid4(), "content": "Test 1", "similarity": 0.7}],
            [{"id": uuid4(), "session_id": uuid4(), "content": "Test 2", "similarity": 0.9}],
            [],
        ]

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.search_similar("test")

            if len(result) >= 2:
                assert result[0]["similarity"] >= result[1]["similarity"]


# ============================================================================
# backfill_embeddings Tests
# ============================================================================


class TestBackfillEmbeddings:
    """Tests for backfill_embeddings function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_backfill_beliefs(self, env_with_openai_key, mock_get_cursor):
        """Should backfill belief embeddings."""
        from oro_embeddings import service

        belief_id = uuid4()
        mock_get_cursor.fetchall.return_value = [{"id": belief_id, "content": "Test belief"}]

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            with patch("oro_embeddings.service.embed_content") as mock_embed:
                result = service.backfill_embeddings("belief", batch_size=10)

                mock_embed.assert_called_once()
                assert result == 1

    def test_handles_errors(self, env_with_openai_key, mock_get_cursor):
        """Should continue on individual embedding errors."""
        from oro_embeddings import service

        mock_get_cursor.fetchall.return_value = [
            {"id": uuid4(), "content": "Test 1"},
            {"id": uuid4(), "content": "Test 2"},
        ]

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            with patch("oro_embeddings.service.embed_content") as mock_embed:
                mock_embed.side_effect = [
                    EmbeddingError("Failed"),
                    {"content_type": "belief"},
                ]

                result = service.backfill_embeddings("belief")

                assert result == 1

    def test_returns_zero_for_unknown_type(self, env_with_openai_key, mock_get_cursor):
        """Should return 0 for unknown content type."""
        from oro_embeddings import service

        with patch("oro_embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_get_type.return_value = mock_type

            result = service.backfill_embeddings("unknown_type")

            assert result == 0


# ============================================================================
# Async Function Tests
# ============================================================================


class TestAsyncFunctions:
    """Tests for async wrapper functions."""

    @pytest.mark.asyncio
    async def test_embed_content_async(self, env_with_openai_key):
        """Should wrap embed_content in async."""
        from oro_embeddings import service

        with patch("oro_embeddings.service.embed_content") as mock_embed:
            mock_embed.return_value = {"content_type": "belief"}

            result = await service.embed_content_async("belief", str(uuid4()), "test")

            assert result["content_type"] == "belief"

    @pytest.mark.asyncio
    async def test_search_similar_async(self, env_with_openai_key):
        """Should wrap search_similar in async."""
        from oro_embeddings import service

        with patch("oro_embeddings.service.search_similar") as mock_search:
            mock_search.return_value = []

            result = await service.search_similar_async("test query")

            assert result == []
