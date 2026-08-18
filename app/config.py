"""
Central configuration. All values are overridable via environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "ATS Backend"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- Auth (HR-facing routes only, see core/security.py) ---
    ATS_API_KEY: str = "changeme"

    # --- LLM (structuring + scoring) ---
    # Gemini API - free tier, get a key at https://aistudio.google.com/apikey
    # - no billing/credit card required. IMPORTANT: new API keys are
    # currently restricted to the 3.x model family (older 2.5 models return
    # "no longer available to new users"). gemini-3.1-flash-lite has the
    # best free daily quota among currently-allowed models; gemini-3.5-flash
    # works too but has a much lower daily cap. Always double-check current
    # limits at https://ai.google.dev/gemini-api/docs/rate-limits since
    # Google adjusts these often.
    #
    # Up to 3 Gemini keys can be set (e.g. 3 separate free-tier accounts).
    # GEMINI_API_KEY_2 / GEMINI_API_KEY_3 are optional - leave blank to use
    # just one key. When a key hits its rate limit / daily quota, the next
    # configured key is tried automatically, in order, before falling back
    # to Grok (if configured).
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_MAX_RETRIES: int = 2

    # Optional fallback if ALL configured Gemini keys are exhausted (e.g.
    # every key hit its daily quota) - only used when at least one
    # GROK_API_KEY* is set. Get a key (paid, no free tier) at
    # https://console.x.ai. Leave all three blank to disable the fallback
    # entirely - behavior is then identical to Gemini-only. Same rotation
    # rule applies: GROK_API_KEY_2 / GROK_API_KEY_3 are tried in order if the
    # one before it gets rate-limited.
    GROK_API_KEY: str = ""
    GROK_API_KEY_2: str = ""
    GROK_API_KEY_3: str = ""
    GROK_MODEL: str = "grok-4.3"

    @property
    def gemini_keys(self) -> list[str]:
        """Non-empty Gemini keys, in priority order."""
        return [k for k in (self.GEMINI_API_KEY, self.GEMINI_API_KEY_2, self.GEMINI_API_KEY_3) if k and k.strip()]

    @property
    def grok_keys(self) -> list[str]:
        """Non-empty Grok keys, in priority order."""
        return [k for k in (self.GROK_API_KEY, self.GROK_API_KEY_2, self.GROK_API_KEY_3) if k and k.strip()]

    # --- Embeddings ---
    # Local, no external key required by default. Swap for a Voyage AI / other
    # provider client in services/embedding_service.py if you prefer a hosted model.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- ChromaDB (the ONLY database) ---
    # mode = "embedded": PersistentClient writes straight to CHROMA_PERSIST_DIR (default,
    #        fine for a single backend instance / internship scale).
    # mode = "server":   HttpClient connects to a standalone ChromaDB server, so
    #        multiple FastAPI replicas can share one index (see docker-compose.yml).
    CHROMA_MODE: str = "embedded"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    JOBS_COLLECTION: str = "job_descriptions"
    RESUMES_COLLECTION: str = "candidate_resumes"

    # --- File storage ---
    STORAGE_DIR: str = "./storage/resumes"
    PENDING_SUBDIR: str = "pending"
    PROCESSED_SUBDIR: str = "processed"
    MAX_FILE_SIZE_MB: int = 5
    ALLOWED_EXTENSIONS: tuple = (".pdf", ".docx")

    # --- Search ---
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 200

    # --- Search assistant answer ---
    # Cap on how many scored candidates go into the natural-language answer's
    # LLM context in one call (highest-scored first), so a very large
    # candidate pool never blows the context window. Does NOT affect how
    # many candidates get scored/returned in "results" - only the answer.
    SEARCH_ANSWER_MAX_CANDIDATES: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
