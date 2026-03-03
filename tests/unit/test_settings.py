from src.config.settings import Settings


def test_default_settings():
    s = Settings(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
    )
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.embedding_model == "text-embedding-3-small"
    assert s.llm_model == "claude-3-5-sonnet-20241022"
    assert s.log_level == "INFO"
    assert s.api_port == 8000


def test_chroma_persist_path():
    s = Settings(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        chroma_persist_dir="/tmp/test-chroma",
    )
    assert str(s.chroma_persist_path) == "/tmp/test-chroma"
