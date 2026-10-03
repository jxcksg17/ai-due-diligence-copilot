# Operations

## Runtime boundary

The React frontend, API, migrations, and PostgreSQL run in separate containers. Ollama runs on the host and is reached through `host.docker.internal`; model weights are not copied into either image. Hugging Face models are downloaded into a named cache volume on first use. Local AI requests default to one at a time because the verified 16 GB Apple Silicon machine cannot safely keep BGE, MiniLM, Qwen, and DeBERTa resident concurrently.

## Start and migrate

Create `.env` from `.env.example` and set a non-default `POSTGRES_PASSWORD`. Ensure Ollama is running and `qwen3:8b-q4_K_M` is available, then run:

```bash
docker compose build
docker compose run --rm migrate
docker compose up -d
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

The research terminal is available at `http://localhost:5173`. `VITE_API_BASE_URL` is compiled into the static frontend image and defaults to the browser-accessible API at `http://localhost:8000`. Local Vite development uses `npm run dev` from `frontend/`; the API explicitly allows only configured origins through `CORS_ALLOWED_ORIGINS`.

`migrate` is deliberately separate from API startup. A failed migration prevents the API service from starting instead of hiding schema changes inside an entrypoint.

## Health and failure semantics

- `/health/live` proves the process can serve HTTP and never loads a model.
- `/health/ready` checks PostgreSQL, Alembic head, Ollama, and the configured Qwen tag without loading local transformer models.
- Invalid input returns 422; ambiguous scope 409; missing filings 404; unavailable dependencies 503; local model timeout 504; bounded local-AI capacity 429.
- Every response carries `X-Request-ID`. JSON logs include method, path, status, and duration, never prompts, filing text, or response bodies.

## Resource assumptions

Run one Uvicorn worker and one in-flight AI request on a 16 GB machine. Retrieval models are released before Qwen generation, Qwen is unloaded before CPU NLI verification, and no background queue is introduced in M12. Scale-out requires measuring memory per replica and running Ollama as a separately managed inference service.
