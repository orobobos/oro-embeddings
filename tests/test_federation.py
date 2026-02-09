"""Tests for federation embedding standard."""

import math
from unittest.mock import MagicMock, patch

import pytest


class TestFederationStandard:
    """Tests for get_federation_standard() and constants."""

    def test_federation_standard_constants(self):
        """Test that federation constants are defined correctly."""
        from oro_embeddings.federation import (
            FEDERATION_EMBEDDING_DIMS,
            FEDERATION_EMBEDDING_MODEL,
            FEDERATION_EMBEDDING_TYPE,
            FEDERATION_EMBEDDING_VERSION,
        )

        assert FEDERATION_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
        assert FEDERATION_EMBEDDING_DIMS == 384
        assert FEDERATION_EMBEDDING_TYPE == "bge_small_en_v15"
        assert FEDERATION_EMBEDDING_VERSION == "1.0"

    def test_get_federation_standard_returns_dict(self):
        """Test get_federation_standard returns complete specification."""
        from oro_embeddings.federation import get_federation_standard

        standard = get_federation_standard()

        assert isinstance(standard, dict)
        assert standard["model"] == "BAAI/bge-small-en-v1.5"
        assert standard["dimensions"] == 384
        assert standard["type"] == "bge_small_en_v15"
        assert standard["normalization"] == "L2"
        assert standard["version"] == "1.0"

    def test_get_federation_standard_immutable(self):
        """Test that returned standard doesn't affect internal state."""
        from oro_embeddings.federation import get_federation_standard

        standard1 = get_federation_standard()
        standard1["model"] = "modified"

        standard2 = get_federation_standard()
        assert standard2["model"] == "BAAI/bge-small-en-v1.5"


class TestFederationCompatibility:
    """Tests for is_federation_compatible()."""

    def test_compatible_embedding(self):
        from oro_embeddings.federation import is_federation_compatible

        assert is_federation_compatible("bge_small_en_v15", 384) is True

    def test_incompatible_wrong_type(self):
        from oro_embeddings.federation import is_federation_compatible

        assert is_federation_compatible("text_embedding_3_small", 384) is False
        assert is_federation_compatible("openai_ada", 384) is False
        assert is_federation_compatible("", 384) is False

    def test_incompatible_wrong_dimensions(self):
        from oro_embeddings.federation import is_federation_compatible

        assert is_federation_compatible("bge_small_en_v15", 1536) is False
        assert is_federation_compatible("bge_small_en_v15", 768) is False
        assert is_federation_compatible("bge_small_en_v15", 0) is False

    def test_incompatible_none_values(self):
        from oro_embeddings.federation import is_federation_compatible

        assert is_federation_compatible(None, 384) is False
        assert is_federation_compatible("bge_small_en_v15", None) is False
        assert is_federation_compatible(None, None) is False


class TestEmbeddingValidation:
    """Tests for validate_federation_embedding()."""

    def _make_normalized_embedding(self, dims: int = 384) -> list[float]:
        values = [1.0 / math.sqrt(dims)] * dims
        return values

    def test_valid_embedding(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = self._make_normalized_embedding(384)
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is True
        assert error is None

    def test_none_embedding(self):
        from oro_embeddings.federation import validate_federation_embedding

        is_valid, error = validate_federation_embedding(None)

        assert is_valid is False
        assert "None" in error

    def test_wrong_type(self):
        from oro_embeddings.federation import validate_federation_embedding

        is_valid, error = validate_federation_embedding("not a list")

        assert is_valid is False
        assert "list" in error.lower()

    def test_wrong_dimensions(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = self._make_normalized_embedding(1536)
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is False
        assert "384" in error
        assert "1536" in error

    def test_nan_values(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = self._make_normalized_embedding(384)
        embedding[0] = float("nan")
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is False
        assert "invalid" in error.lower() or "nan" in error.lower()

    def test_inf_values(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = self._make_normalized_embedding(384)
        embedding[0] = float("inf")
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is False
        assert "invalid" in error.lower() or "inf" in error.lower()

    def test_non_normalized_embedding(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = [0.5] * 384
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is False
        assert "normalized" in error.lower()

    def test_zero_vector_fails(self):
        from oro_embeddings.federation import validate_federation_embedding

        embedding = [0.0] * 384
        is_valid, error = validate_federation_embedding(embedding)

        assert is_valid is False
        assert "normalized" in error.lower()


class TestIncomingBeliefValidation:
    """Tests for validate_incoming_belief_embedding()."""

    def _make_normalized_embedding(self, dims: int = 384) -> list[float]:
        return [1.0 / math.sqrt(dims)] * dims

    def test_valid_belief_with_embedding(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {
            "content": "Test belief",
            "embedding": self._make_normalized_embedding(384),
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dims": 384,
            "embedding_type": "bge_small_en_v15",
        }

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is True
        assert error is None

    def test_belief_without_embedding_is_valid(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {"content": "Test belief without embedding"}

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is True
        assert error is None

    def test_belief_with_wrong_model_fails(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {
            "content": "Test belief",
            "embedding": self._make_normalized_embedding(384),
            "embedding_model": "text-embedding-3-small",
        }

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is False
        assert "model mismatch" in error.lower()

    def test_belief_with_wrong_dims_fails(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {
            "content": "Test belief",
            "embedding": self._make_normalized_embedding(384),
            "embedding_dims": 1536,
        }

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is False
        assert "dimensions mismatch" in error.lower()

    def test_belief_with_wrong_type_fails(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {
            "content": "Test belief",
            "embedding": self._make_normalized_embedding(384),
            "embedding_type": "openai_ada",
        }

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is False
        assert "type mismatch" in error.lower()

    def test_belief_with_invalid_embedding_fails(self):
        from oro_embeddings.federation import validate_incoming_belief_embedding

        belief_data = {
            "content": "Test belief",
            "embedding": [0.5] * 1536,
        }

        is_valid, error = validate_incoming_belief_embedding(belief_data)
        assert is_valid is False


class TestRegenerateEmbedding:
    """Tests for regenerate_embedding_if_needed()."""

    def test_regenerate_requires_content(self):
        from oro_embeddings.federation import regenerate_embedding_if_needed

        with pytest.raises(ValueError, match="no content"):
            regenerate_embedding_if_needed({})

    def test_regenerate_empty_content(self):
        from oro_embeddings.federation import regenerate_embedding_if_needed

        with pytest.raises(ValueError, match="no content"):
            regenerate_embedding_if_needed({"content": ""})

    @patch("oro_embeddings.providers.local.generate_embedding")
    def test_regenerate_calls_local_provider(self, mock_generate):
        from oro_embeddings.federation import regenerate_embedding_if_needed

        mock_embedding = [0.1] * 384
        mock_generate.return_value = mock_embedding

        result = regenerate_embedding_if_needed({"content": "Test content"})

        mock_generate.assert_called_once_with("Test content")
        assert result == mock_embedding


class TestPrepareBeliefForFederation:
    """Tests for prepare_belief_for_federation() async function."""

    @pytest.fixture
    def mock_db_row(self):
        from datetime import datetime

        return {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "content": "Test belief content",
            "confidence": {"overall": 0.85},
            "domain_path": ["tech", "databases"],
            "valid_from": None,
            "valid_until": None,
            "visibility": "federated",
            "share_level": "belief_only",
            "federation_id": None,
            "created_at": datetime.now(),
            "modified_at": datetime.now(),
            "embedding_type": None,
            "dimensions": None,
            "vector": None,
        }

    @pytest.mark.asyncio
    @patch("oro_db.get_cursor")
    @patch("oro_embeddings.providers.local.generate_embedding")
    async def test_prepare_generates_embedding(self, mock_generate, mock_cursor, mock_db_row):
        from oro_embeddings.federation import prepare_belief_for_federation

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = mock_db_row
        mock_cursor.return_value.__enter__.return_value = mock_cur

        mock_embedding = [0.05] * 384
        mock_generate.return_value = mock_embedding

        result = await prepare_belief_for_federation(
            "550e8400-e29b-41d4-a716-446655440000",
            node_did="did:vkb:web:test.example.com",
        )

        assert result["belief_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["content"] == "Test belief content"
        assert result["embedding"] == mock_embedding
        assert result["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert result["embedding_dims"] == 384
        assert result["embedding_type"] == "bge_small_en_v15"
        mock_generate.assert_called_once_with("Test belief content")

    @pytest.mark.asyncio
    @patch("oro_db.get_cursor")
    async def test_prepare_uses_existing_compatible_embedding(self, mock_cursor, mock_db_row):
        from oro_embeddings.federation import prepare_belief_for_federation

        mock_db_row["embedding_type"] = "bge_small_en_v15"
        mock_db_row["dimensions"] = 384
        mock_db_row["vector"] = "[" + ",".join(["0.05"] * 384) + "]"

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = mock_db_row
        mock_cursor.return_value.__enter__.return_value = mock_cur

        result = await prepare_belief_for_federation(
            "550e8400-e29b-41d4-a716-446655440000",
            node_did="did:vkb:web:test.example.com",
        )

        assert result["embedding"] == [0.05] * 384
        assert result["embedding_model"] == "BAAI/bge-small-en-v1.5"

    @pytest.mark.asyncio
    @patch("oro_db.get_cursor")
    async def test_prepare_raises_for_missing_belief(self, mock_cursor):
        from oro_embeddings.federation import prepare_belief_for_federation

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__.return_value = mock_cur

        with pytest.raises(ValueError, match="not found"):
            await prepare_belief_for_federation("nonexistent-id")

    @pytest.mark.asyncio
    @patch("oro_db.get_cursor")
    async def test_prepare_raises_for_private_belief(self, mock_cursor, mock_db_row):
        from oro_embeddings.federation import prepare_belief_for_federation

        mock_db_row["visibility"] = "private"

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = mock_db_row
        mock_cursor.return_value.__enter__.return_value = mock_cur

        with pytest.raises(ValueError, match="private visibility"):
            await prepare_belief_for_federation("550e8400-e29b-41d4-a716-446655440000")


class TestModuleExports:
    """Test that module exports are correctly exposed."""

    def test_federation_exports_in_init(self):
        from oro_embeddings import (
            FEDERATION_EMBEDDING_MODEL,
            get_federation_standard,
            is_federation_compatible,
            validate_federation_embedding,
        )

        assert FEDERATION_EMBEDDING_MODEL is not None
        assert callable(get_federation_standard)
        assert callable(is_federation_compatible)
        assert callable(validate_federation_embedding)
