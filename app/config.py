import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ==============================================================================
# HARDCODED MODEL CONFIGURATION
# You can change this variable to any supported OpenAI, Anthropic, or Groq model.
# Examples:
#   OpenAI:    "gpt-4o", "gpt-4o-mini", "gpt-6-luna", "o1-mini", "o3-mini"
#   Anthropic: "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"
#   Groq:      "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"
# ==============================================================================
HARDCODED_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-6-luna")

# Timeout Control
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0"))

# Provider API Keys (strip whitespace/leading spaces if present)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
