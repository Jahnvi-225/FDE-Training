# FDE Training: Multi-Provider LLM Service

A production-ready FastAPI service managed by `uv` that integrates **OpenAI**, **Anthropic**, and **Groq** with intelligent model routing, timeouts, global error handling, and structured logging.

---

## Table of Contents
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Quick Start](#quick-start)
- [Configuration & Hardcoded Models](#configuration--hardcoded-models)
- [API Endpoints Reference](#api-endpoints-reference)
- [Viewing Generated Logs](#viewing-generated-logs)
- [Global Error Handling](#global-error-handling)

---

## Features

- **Multi-Provider Unified Gateway:** Connects to OpenAI, Anthropic, and Groq through a single endpoint (`POST /chat`).
- **Hardcoded Model Variable with Smart Routing:** Define your target model in one place (e.g., `gpt-6-luna`, `llama-3.3-70b-versatile`, or `claude-3-5-sonnet-20241022`). The service automatically routes to the right provider.
- **Configurable Timeouts:** `asyncio.wait_for` timeout protection (`HTTP 504`) prevents hanging requests.
- **Granular Latency & Token Logging:** Measures and records latency (in milliseconds) and token counts (prompt, completion, total) for every call to both console and `logs/app.log`.
- **Global Error Handling:** All domain exceptions return uniform, clean JSON responses with UTC timestamps.

---

## Project Architecture

```
FDE-Training/
├── .env                     # Private API keys and local settings
├── .env.example             # Template for required environment variables
├── .gitignore               # Excludes .venv, __pycache__, and logs/
├── pyproject.toml           # Project dependencies managed by uv
├── uv.lock                  # Pinned dependency lockfile
├── README.md                # Project documentation and log viewer
├── docs/
│   ├── llm_providers_comparison.md   # Comparison of OpenAI, Anthropic & Groq
│   └── daily logs/
│       └── day 1.md                  # Development log
├── logs/
│   └── app.log              # Persistent log file capturing all events
├── tests/                   # Automated pytest suite
└── app/
    ├── __init__.py          # Package entrypoint (uv run fde-training)
    ├── config.py            # Central HARDCODED_MODEL, timeout, and API key settings
    ├── exceptions.py        # Custom domain exceptions
    ├── schemas.py           # Pydantic request/response models
    ├── main.py              # FastAPI application & global error handlers
    └── services/
        ├── __init__.py
        └── llm_service.py   # Provider routing, execution, and metric logging
```

---

## Quick Start

### 1. Install Dependencies
Dependencies are managed using `uv`:
```powershell
uv sync
```

### 2. Configure Environment Keys
Ensure your `.env` contains your API keys:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Active default model
DEFAULT_MODEL=gpt-6-luna

# Guardrails
REQUEST_TIMEOUT_SECONDS=30.0
```

### 3. Start the Server
```powershell
uv run uvicorn app.main:app --reload --port 8000
```
Or using the registered CLI shortcut:
```powershell
uv run fde-training
```

### 4. Run Unit Tests
```powershell
uv run pytest -v
```

---

## Configuration & Hardcoded Models

In **`app/config.py`**, you can change the target model:

```python
HARDCODED_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-6-luna")
```

The router identifies the provider from the model name:
| Model Prefix | Provider | Example Supported Models |
| :--- | :--- | :--- |
| `gpt-`, `o1`, `o3` | **OpenAI** | `gpt-6-luna`, `gpt-4o`, `gpt-4o-mini`, `o1-mini` |
| `claude-` | **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` |
| `llama-`, `mixtral-`, `gemma` | **Groq** | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768` |

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/chat` | Send a chat prompt with optional model overrides, max tokens, and temperature. |
| `GET` | `/health` | Service liveness health check (`healthy` status). |
| `GET` | `/` | Root index with endpoint listing. |
| `GET` | `/docs` | Interactive Swagger UI for testing all endpoints. |

---

## Viewing Generated Logs

All logs are formatted with ISO timestamps, log level severity, logger name, and structured key-value pairs.

### Log Locations
1. **Standard Console Output (stdout):** Rendered live in the terminal running uvicorn.
2. **Persistent Log File:** Written to **`logs/app.log`** with UTF-8 encoding.

### Live Log Streaming Commands

#### In Windows PowerShell (Live Tail):
```powershell
Get-Content -Path logs\app.log -Wait -Tail 30
```

#### In Bash / macOS / Linux:
```bash
tail -f logs/app.log
```

### Sample Log Scenarios

#### 1. Successful LLM Call (Latency & Token Breakdown)
```text
2026-11-24 02:01:02 | INFO    | app.llm | [LLM Request Initiated] Provider: Openai | Model: gpt-6-luna | MaxTokens: 1024
2026-11-24 02:01:05 | INFO    | app.llm | [LLM Metrics] Provider: Openai | Model: gpt-6-luna | Latency: 2.904s (2904.0ms) | Tokens: [Prompt: 12, Completion: 23, Total: 35]
2026-11-24 02:01:05 | INFO    | uvicorn.access | 127.0.0.1 - "POST /chat HTTP/1.1" 200 OK
```

#### 2. Provider Call Timeout (HTTP 504)
```text
2026-11-24 02:08:15 | ERROR   | app.llm | [LLM Timeout] Provider: Anthropic | Model: claude-3-5-sonnet-20241022 | Latency: 30.002s (30002.0ms) | Timeout limit: 30.0s
2026-11-24 02:08:15 | WARNING | app | Handled LLM exception on /chat: [llm_timeout] Call to anthropic (model: 'claude-3-5-sonnet-20241022') exceeded timeout of 30.0s.
```

#### 3. Authentication Failure (HTTP 401)
```text
2026-11-24 02:11:02 | WARNING | app | Handled LLM exception on /chat: [authentication_error] OpenAI Authentication Failed: Incorrect API key provided.
```

---

## Global Error Handling

All unhandled exceptions and domain errors return uniform JSON payloads:

```json
{
  "error": "Detailed description of the issue",
  "error_type": "authentication_error | llm_timeout | unsupported_model | provider_api_error | internal_server_error",
  "status_code": 400,
  "timestamp": "2026-11-24T02:01:05.123456Z"
}
```
