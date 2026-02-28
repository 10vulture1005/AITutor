from typing import List, Optional, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Eduverse Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]


    JWT_SECRET: str = "your-256-bit-secret-change-this-in-production"
    FERNET_KEY: str = "your-fernet-key-change-this-in-production"  
    SECRET_KEY: str = "change-me-to-a-real-secret-key" 
    
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/eduverse"
    
    UPLOAD_DIR: str = "./uploads"
    
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: Optional[str] = "http://localhost:8000/auth/callback"
    
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    # ── Model settings (different models to distribute TPM load) ──
    # Agent: best tool calling + structured output support
    AGENT_MODEL: str = "openai/gpt-oss-120b"
    # JSON tasks (flashcards, etc.): lightweight + great JSON output
    JSON_MODEL: str = "openai/gpt-oss-20b"
    # Web search: Groq's built-in web search
    WEB_SEARCH_MODEL: str = "groq/compound-mini"

    # RAG settings
    RAG_LLM_TEMPERATURE: float = 0.3
    RAG_RETRIEVER_K: int = 5
    RAG_RETRIEVER_FETCH_K: int = 30
    RAG_RERANK_TOP_N: int = 5

    GROQ_API_KEY: Optional[str] = None

    # Supabase (optional — for Storage API)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    @property
    def PG_SYNC_URL(self) -> str:
        """SQLAlchemy psycopg3 URL — used by PGVector (via SQLAlchemy engine).
        
        Format: postgresql+psycopg://...
        """
        url = self.DATABASE_URL.replace("+asyncpg", "+psycopg")
        # Append sslmode for Supabase TLS
        sep = "&" if "?" in url else "?"
        return url + sep + "sslmode=require"

    @property
    def PG_CONNINFO(self) -> str:
        """Plain psycopg3 connection string — used by components that call
        psycopg.connect() directly.
        
        Used by: PostgresChatMessageHistory.
        Format: postgresql://... (no +psycopg dialect prefix)
        """
        url = self.DATABASE_URL.replace("+asyncpg", "")
        sep = "&" if "?" in url else "?"
        return url + sep + "sslmode=require&options=-c%20statement_timeout%3D180000"

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

settings = Settings()
