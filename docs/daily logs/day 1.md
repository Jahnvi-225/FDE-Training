# Day 1 - Development Log

**Date:** November 23–28, 2026
**Project:** FDE Training — Multi-Provider LLM API
**Engineer:** Jahnvi

---

## Summary

Day 1 covered the full initial setup of a clean, production-ready FastAPI backend in `app/` that integrates three major LLM providers (OpenAI, Anthropic, and Groq) with a hardcoded model variable, timeout guards, global error handling, structured per-request logging, and an automated unit test suite.

---

## Work Done

### 1. Created `docs/` Folder with LLM Provider Comparison

**File:** [`docs/llm_providers_comparison.md`](../llm_providers_comparison.md)

- Documented key differences between **OpenAI**, **Anthropic**, and **Groq** in a **table-first layout** for easy reading.

#### Key Comparison Points Documented:
| Feature | OpenAI | Anthropic | Groq |
| :--- | :--- | :--- | :--- |
| **Flagship Models** | `gpt-4o`, `gpt-6-luna`, `o1`, `o3-mini` | `claude-3-5-sonnet`, `claude-3.5-haiku` | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` |
| **Speed (Throughput)** | 60–110 tok/s | 70–120 tok/s | **300–800+ tok/s (LPU)** |
| **Context Window** | 128k tokens | **200k tokens** | 8k–128k (hardware bound) |
| **Coding Strength** | Strong (`o1`, `o3-mini`) | **Best-in-class** (SWE-bench) | Model-dependent |
| **Self-Hosting** | No | No | No (GroqCloud only) |
| **Audio API** | Yes (Realtime Voice) | No | Via Whisper LPU |

---

### 2. Initialized FastAPI Project with `uv`

**Files Created:**
- `pyproject.toml` — project manifest, dependencies, `uv_build` configured for `app/`, and pytest configuration
- `uv.lock` — pinned dependency lockfile
- `.python-version` — Python 3.15
- `.venv/` — project virtual environment

**Dependencies Installed:**
```
fastapi, uvicorn, pydantic, openai, anthropic, groq, python-dotenv
pytest, pytest-asyncio (dev)
```

---

### 3. Added `.gitignore`

**File:** [`.gitignore`](../../.gitignore)

Configured to ignore `__pycache__/`, `.venv/`, `dist/`, `build/`, `.env` (keeping `.env.example`), `logs/`, `.vscode/`, `.idea/`, and OS files.

---

### 4. Implemented FastAPI Health Endpoint & Multi-Provider Chat Service

**Core Architecture (`app/`):**

| File | Responsibility |
| :--- | :--- |
| `app/__init__.py` | Package entrypoint (`uv run fde-training`) |
| `app/config.py` | Central `HARDCODED_MODEL` variable, timeout, and API key loading |
| `app/exceptions.py` | Custom domain exceptions with HTTP status codes |
| `app/schemas.py` | Pydantic models: `ChatRequest`, `ChatResponse`, `TokenUsage`, `ErrorResponse` |
| `app/services/llm_service.py` | Unified provider router → OpenAI, Anthropic, or Groq |
| `app/main.py` | FastAPI application with endpoints and global exception handlers |

#### Hardcoded Model Variable
In `app/config.py`:
```python
HARDCODED_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-6-luna")
```

The service automatically routes based on model name prefix:
- `gpt-`, `o1`, `o3` → **OpenAI**
- `claude-` → **Anthropic**
- `llama-`, `mixtral-`, `gemma` → **Groq**

#### Timeout Protection:
- All provider calls wrapped with `asyncio.wait_for()`. Timeout triggers `HTTP 504`.

---

### 5. API Endpoints Implemented

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | Multi-provider LLM chat completion |
| `GET` | `/health` | Service health check |
| `GET` | `/` | Root index with endpoint listing |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

### 6. Global Error Handling

Registered two `@app.exception_handler` handlers on the FastAPI app:

1. **`LLMServiceException` handler** — catches all custom domain exceptions, returns structured JSON with `error`, `error_type`, `status_code`, and `timestamp`.
2. **`Exception` handler** — fallback for any unexpected `RuntimeError` or uncaught exception, returns `HTTP 500` with safe generic message.

| Error Type | HTTP Code | Trigger |
| :--- | :--- | :--- |
| `unsupported_model` | 400 | Model prefix doesn't match any known provider |
| `authentication_error` | 401 | Missing or rejected API key |
| `provider_api_error` | 502 | Downstream provider returned an error |
| `llm_timeout` | 504 | Provider call exceeded `REQUEST_TIMEOUT_SECONDS` |
| `internal_server_error` | 500 | Uncaught system exception |

---

### 7. Latency & Token Usage Logging

**Updated:** `app/services/llm_service.py`, `app/main.py`

- Logs request initiation, latency (in seconds and ms), token breakdown (prompt, completion, total), and failures/timeouts to both **console** and **`logs/app.log`**.

---

### 8. Unit Test Suite

**Directory:** [`tests/`](../../tests/)

| File | Covers |
| :--- | :--- |
| `conftest.py` | ASGI `async_client` fixture |
| `test_api_endpoints.py` | `/`, `/health`, `/chat`, and all 5 global error handler scenarios |
| `test_llm_service.py` | Provider routing, mocked OpenAI/Anthropic/Groq calls, timeout |
| `test_exceptions.py` | Status codes, error type tags, exception messages |
| `test_schemas.py` | Pydantic field validation (token limits, temperature bounds) |

**Run command:**
```powershell
uv run pytest -v
```
