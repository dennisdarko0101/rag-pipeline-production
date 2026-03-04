from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"

    # Logging
    log_level: str = "INFO"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # LLM
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_fallback_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_requests: int = 60
    rate_limit_window: int = 60
    cors_origins: str = "*"

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()
