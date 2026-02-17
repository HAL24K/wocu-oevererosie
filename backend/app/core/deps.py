"""
Dependency injection for FastAPI routes.
Provides reusable dependencies for database connections, settings, etc.
"""

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings

# Type alias for settings dependency
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_current_settings() -> Settings:
    """
    Dependency to get application settings.
    
    Returns:
        Settings: Application settings instance
    """
    return get_settings()


# Example dependency for future use (e.g., database session)
# async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
#     """
#     Dependency to get database session.
#     
#     Yields:
#         AsyncSession: Database session
#     """
#     async with async_session_maker() as session:
#         yield session




