"""
Application entry point for the FastAPI prediction service.
"""

from src.api.main import app

# Export the app instance for use with ASGI servers
__all__ = ["app"]