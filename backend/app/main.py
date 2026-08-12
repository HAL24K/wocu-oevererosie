"""
FastAPI application entry point.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # TODO: Load ML model here
    # logger.info("Loading ML models...")

    yield

    # Shutdown
    logger.info("Shutting down application")
    # TODO: Cleanup resources (close connections, unload models, etc.)


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> JSONResponse:
        """
        Health check endpoint.

        Returns:
            JSONResponse: Health status
        """
        return JSONResponse(
            content={
                "status": "healthy",
                "version": settings.api_version,
                "environment": settings.environment,
            }
        )

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> JSONResponse:
        """
        Root endpoint with API information.

        Returns:
            JSONResponse: API information
        """
        return JSONResponse(
            content={
                "message": "WOCU River Bank Erosion Prediction API",
                "version": settings.api_version,
                "docs": "/docs",
                "health": "/health",
            }
        )

    # TODO: Register routers here
    # from app.api.routes import predictions, data
    # app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
    # app.include_router(data.router, prefix="/api/v1/data", tags=["data"])

    return app


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        reload_excludes=[".venv/*", "*.pyc", "__pycache__/*"],
        log_level=settings.log_level.lower(),
    )
