# our-embeddings

Vector embedding generation and similarity search for the ourochronos ecosystem.

## Overview

our-embeddings provides a unified interface for generating and searching vector embeddings. It supports both local (sentence-transformers) and OpenAI providers, with a federation standard for cross-node embedding compatibility.

Default model: **BAAI/bge-small-en-v1.5** (384 dimensions, L2-normalized).

## Install

```bash
pip install our-embeddings
```

For local embeddings (default, no API key needed):
```bash
pip install our-embeddings[local]  # includes sentence-transformers
```

## Usage

### Generate Embeddings

```python
from our_embeddings.service import generate_embedding, vector_to_pgvector

# Generate a 384-dim embedding vector
vector = generate_embedding("PostgreSQL is excellent for JSONB queries")

# Convert to pgvector format for storage
pg_str = vector_to_pgvector(vector)
# → "[0.0231,0.0891,...]"
```

### Search Similar Content

```python
from our_embeddings import search_similar

results = search_similar(
    query="database performance",
    content_type="belief",
    limit=10,
    min_similarity=0.5,
)
# Returns list of dicts with id, content, similarity score
```

### Embed and Store

```python
from our_embeddings import embed_content

result = embed_content(
    content_type="belief",
    content_id="uuid-here",
    text="Valence uses dimensional confidence",
)
```

### Batch Operations

```python
from our_embeddings.local import generate_embeddings_batch

vectors = generate_embeddings_batch(
    ["text one", "text two", "text three"],
    batch_size=32,
)
```

### Backfill Missing Embeddings

```python
from our_embeddings import backfill_embeddings

count = backfill_embeddings(content_type="belief", batch_size=100)
```

## Configuration

### EmbeddingConfig

```python
from our_embeddings.config import EmbeddingConfig

config = EmbeddingConfig.from_env()
# Fields:
#   embedding_provider: str = "local"
#   embedding_model_path: str = "BAAI/bge-small-en-v1.5"
#   embedding_device: str = "cpu"
#   openai_api_key: str = ""
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VALENCE_EMBEDDING_PROVIDER` | `local` | `"local"` or `"openai"` |
| `VALENCE_EMBEDDING_MODEL_PATH` | `BAAI/bge-small-en-v1.5` | Model name or path |
| `VALENCE_EMBEDDING_DEVICE` | `cpu` | `"cpu"` or `"cuda"` |
| `OPENAI_API_KEY` | — | Required if provider is `openai` |

## Providers

### Local (default)

Uses sentence-transformers with BAAI/bge-small-en-v1.5:
- 384 dimensions, L2-normalized
- No API key required
- Model loaded lazily and cached as singleton
- Thread-safe initialization

### OpenAI

Uses OpenAI text-embedding-3-small:
- 1536 dimensions
- Requires `OPENAI_API_KEY`
- Text truncated to 8000 chars

## Embedding Type Registry

Register and manage multiple embedding types:

```python
from our_embeddings import register_embedding_type, list_embedding_types

register_embedding_type(
    type_id="local_bge_small",
    provider="local",
    model="BAAI/bge-small-en-v1.5",
    dimensions=384,
    is_default=True,
)

types = list_embedding_types(status="active")
```

## Federation Standard

Cross-node embedding compatibility for federated knowledge sharing:

```python
from our_embeddings import get_federation_standard, validate_federation_embedding

standard = get_federation_standard()
# → {"model": "BAAI/bge-small-en-v1.5", "dimensions": 384,
#    "type": "bge_small_en_v15", "normalization": "L2", "version": "1.0"}

valid, error = validate_federation_embedding([0.1, 0.2, ...])
```

Federation functions for belief exchange:
- `prepare_belief_for_federation(belief_id)` — Package belief with embedding
- `validate_incoming_belief_embedding(data)` — Validate received embeddings
- `regenerate_embedding_if_needed(data)` — Re-embed if format differs

## State Ownership

Owns the `embedding_types` and `embedding_coverage` tables in the valence schema. Reads/writes the `embedding` column on `beliefs`, `vkb_exchanges`, and `vkb_patterns` tables.

## Development

```bash
make dev       # Install with dev dependencies
make lint      # Run linters
make test      # Run tests
make test-cov  # Tests with coverage
make format    # Auto-format
```

## Part of Valence

This brick is part of the [Valence](https://github.com/ourochronos/valence) knowledge substrate. See [our-infra](https://github.com/ourochronos/our-infra) for ourochronos conventions.

## License

MIT
