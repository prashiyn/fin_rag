# OpenAPI Specs

This folder contains the generated OpenAPI spec for the FinSage RAG API so other services can integrate without importing the server code.

## Files

- `openapi.json`: canonical OpenAPI spec (JSON)
- `openapi.yaml`: same spec in YAML form

Both files are generated from `src/server.py` (FastAPI) and should be regenerated whenever API routes or request/response models change.

## Regenerate

From the project root:

```bash
PYTHONPATH=src uv run python script/generate_openapi.py
```

This overwrites:
- `openapi/openapi.json`
- `openapi/openapi.yaml`

## Notes for integrators

- **Auth**: Most endpoints require `Authorization: Bearer <token>` (see `bearer_token` in config or env `BEARER_TOKEN`).
- **Collection-scoped**: The API is **per-collection**. Endpoints that query RAG require `collection_name` (either in JSON body or query param, as defined in the spec).
- **Ingestion**: `POST /load-data` creates/updates a collection by ingesting chunk payloads.

