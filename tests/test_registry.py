"""Tests for oro_embeddings.registry module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEmbeddingType:
    """Tests for EmbeddingType dataclass."""

    def test_create(self):
        from oro_embeddings.registry import EmbeddingType

        et = EmbeddingType(
            id="openai_text3_small",
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            is_default=True,
            status="active",
        )
        assert et.id == "openai_text3_small"
        assert et.provider == "openai"
        assert et.dimensions == 1536
        assert et.is_default is True

    def test_default_values(self):
        from oro_embeddings.registry import EmbeddingType

        et = EmbeddingType(id="test", provider="openai", model="test-model", dimensions=1536)
        assert et.is_default is False
        assert et.status == "active"

    def test_to_dict(self):
        from oro_embeddings.registry import EmbeddingType

        et = EmbeddingType(id="test", provider="openai", model="test-model", dimensions=1536, is_default=True)
        d = et.to_dict()
        assert d["id"] == "test"
        assert d["dimensions"] == 1536
        assert d["is_default"] is True

    def test_from_row(self):
        from oro_embeddings.registry import EmbeddingType

        row = {
            "id": "openai_text3_small",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }
        et = EmbeddingType.from_row(row)
        assert et.id == "openai_text3_small"
        assert et.is_default is True


class TestGetEmbeddingType:
    @pytest.fixture
    def mock_cursor(self):
        return MagicMock()

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_by_id(self, mock_get_cursor):
        from oro_embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "openai_text3_small",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": False,
            "status": "active",
        }
        result = get_embedding_type("openai_text3_small")
        assert result is not None
        assert result.id == "openai_text3_small"

    def test_default(self, mock_get_cursor):
        from oro_embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "default_type",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }
        result = get_embedding_type()
        assert result is not None
        assert result.is_default is True

    def test_not_found(self, mock_get_cursor):
        from oro_embeddings.registry import get_embedding_type

        mock_get_cursor.fetchone.return_value = None
        result = get_embedding_type("nonexistent")
        assert result is None


class TestListEmbeddingTypes:
    @pytest.fixture
    def mock_cursor(self):
        return MagicMock()

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_list_all(self, mock_get_cursor):
        from oro_embeddings.registry import list_embedding_types

        mock_get_cursor.fetchall.return_value = [
            {
                "id": "type1",
                "provider": "openai",
                "model": "model1",
                "dimensions": 1536,
                "is_default": True,
                "status": "active",
            },
            {
                "id": "type2",
                "provider": "openai",
                "model": "model2",
                "dimensions": 3072,
                "is_default": False,
                "status": "active",
            },
        ]
        result = list_embedding_types()
        assert len(result) == 2

    def test_filter_by_status(self, mock_get_cursor):
        from oro_embeddings.registry import list_embedding_types

        mock_get_cursor.fetchall.return_value = []
        list_embedding_types(status="active")
        call_args = mock_get_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]


class TestRegisterEmbeddingType:
    @pytest.fixture
    def mock_cursor(self):
        return MagicMock()

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_create_new(self, mock_get_cursor):
        from oro_embeddings.registry import register_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "new_type",
            "provider": "openai",
            "model": "new-model",
            "dimensions": 1536,
            "is_default": False,
            "status": "active",
        }
        result = register_embedding_type("new_type", "openai", "new-model", 1536)
        assert result.id == "new_type"

    def test_set_as_default(self, mock_get_cursor):
        from oro_embeddings.registry import register_embedding_type

        mock_get_cursor.fetchone.return_value = {
            "id": "new_default",
            "provider": "openai",
            "model": "model",
            "dimensions": 1536,
            "is_default": True,
            "status": "active",
        }
        result = register_embedding_type("new_default", "openai", "model", 1536, is_default=True)
        assert result.is_default is True
        calls = mock_get_cursor.execute.call_args_list
        assert any("is_default = FALSE" in str(c) for c in calls)


class TestEnsureDefaultType:
    @pytest.fixture
    def mock_cursor(self):
        return MagicMock()

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("oro_embeddings.registry.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_returns_existing(self, mock_get_cursor):
        from oro_embeddings.registry import EmbeddingType, ensure_default_type

        with patch("oro_embeddings.registry.get_embedding_type") as mock_get:
            mock_get.return_value = EmbeddingType(
                id="existing_default",
                provider="openai",
                model="text-embedding-3-small",
                dimensions=1536,
                is_default=True,
            )
            result = ensure_default_type()
            assert result.id == "existing_default"

    def test_creates_default(self, mock_get_cursor):
        from oro_embeddings.registry import EmbeddingType, ensure_default_type

        with patch("oro_embeddings.registry.get_embedding_type") as mock_get:
            mock_get.return_value = None
            with patch("oro_embeddings.registry.register_embedding_type") as mock_register:
                mock_register.return_value = EmbeddingType(
                    id="openai_text3_small",
                    provider="openai",
                    model="text-embedding-3-small",
                    dimensions=1536,
                    is_default=True,
                )
                result = ensure_default_type()
                mock_register.assert_called_once()
                assert result.is_default is True


class TestKnownEmbeddings:
    def test_contains_text3_small(self):
        from oro_embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_text3_small" in KNOWN_EMBEDDINGS
        assert KNOWN_EMBEDDINGS["openai_text3_small"]["dimensions"] == 1536

    def test_contains_text3_large(self):
        from oro_embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_text3_large" in KNOWN_EMBEDDINGS
        assert KNOWN_EMBEDDINGS["openai_text3_large"]["dimensions"] == 3072

    def test_contains_ada(self):
        from oro_embeddings.registry import KNOWN_EMBEDDINGS

        assert "openai_ada_002" in KNOWN_EMBEDDINGS
