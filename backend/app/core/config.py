"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Settings
    api_title: str = "WOCU River Bank Erosion API"
    api_version: str = "1.0.0"
    api_description: str = (
        "API for predicting river bank erosion using machine learning"
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = Field(default=True, description="Enable debug mode")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = Field(default=True, description="Enable auto-reload in development")

    # Paths
    data_dir: str = Field(default="data", description="Directory for data storage")
    model_dir: str = Field(default="models", description="Directory for ML models")

    # WFS Service Configuration
    wfs_url: str = Field(
        default="https://service.pdok.nl/rws/ahn/wfs/v1_0",
        description="WFS service URL for fetching geospatial data",
    )
    wfs_timeout: int = Field(default=30, description="WFS request timeout in seconds")

    # Model Configuration
    model_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for ML model inference",
    )
    model_batch_size: int = Field(default=32, description="Batch size for predictions")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
