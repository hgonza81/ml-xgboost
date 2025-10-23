"""
Authentication middleware for API key-based authentication.

This module implements:
- API key-based authentication system
- Authentication middleware for request validation
- CORS policy enforcement with whitelisted domains
"""

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Security scheme for API key authentication
security = HTTPBearer()

# Configuration for authentication
class AuthConfig:
    """Configuration for authentication settings."""
    
    def __init__(self):
        # API key from environment variable or default for development
        self.api_key = os.getenv("API_KEY", "dev-api-key-12345")
        
        # CORS settings
        self.cors_origins = self._parse_cors_origins()
        self.cors_methods = ["GET", "POST"]
        self.cors_headers = ["*"]
        
        logger.info(f"Auth configured with CORS origins: {self.cors_origins}")
    
    def _parse_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
        if cors_env:
            origins = [origin.strip() for origin in cors_env.split(",")]
            return origins
        return ["http://localhost:3000", "http://localhost:8080"]  # Default for development

# Global auth configuration
auth_config = AuthConfig()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify the API key from the Authorization header.
    
    Args:
        credentials: HTTP authorization credentials containing the API key
        
    Returns:
        str: The verified API key
        
    Raises:
        HTTPException: If the API key is invalid or missing
    """
    
    if not credentials:
        logger.warning("Missing authorization credentials")
        raise HTTPException(
            status_code=401,
            detail="Missing authorization credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if credentials.credentials != auth_config.api_key:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    logger.debug("API key validated successfully")
    return credentials.credentials


async def get_current_api_key(api_key: str = Depends(verify_api_key)) -> str:
    """
    Dependency to get the current validated API key.
    
    Args:
        api_key: The verified API key from verify_api_key
        
    Returns:
        str: The validated API key
    """
    return api_key


def setup_cors(app):
    """
    Set up CORS middleware with whitelisted domains.
    
    Args:
        app: FastAPI application instance
    """
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=auth_config.cors_origins,
        allow_credentials=True,
        allow_methods=auth_config.cors_methods,
        allow_headers=auth_config.cors_headers,
    )
    
    logger.info(f"CORS middleware configured with origins: {auth_config.cors_origins}")


class AuthenticationMiddleware:
    """
    Custom authentication middleware for request validation and logging.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """
        ASGI middleware for authentication logging and monitoring.
        
        This middleware logs authentication attempts and can be extended
        for rate limiting and suspicious activity detection.
        """
        
        if scope["type"] == "http":
            # Log request details for monitoring
            method = scope.get("method", "")
            path = scope.get("path", "")
            
            # Skip logging for health checks and docs
            if path not in ["/health/ready", "/health/live", "/docs", "/redoc", "/openapi.json"]:
                logger.info(f"Request: {method} {path}")
        
        await self.app(scope, receive, send)


def add_auth_middleware(app):
    """
    Add authentication middleware to the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    # Add custom authentication middleware
    app.add_middleware(AuthenticationMiddleware)
    
    logger.info("Authentication middleware added")


# Optional: Rate limiting configuration (for future enhancement)
class RateLimitConfig:
    """Configuration for rate limiting (placeholder for future implementation)."""
    
    def __init__(self):
        self.requests_per_minute = int(os.getenv("RATE_LIMIT_RPM", "60"))
        self.requests_per_hour = int(os.getenv("RATE_LIMIT_RPH", "1000"))
        
    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited (placeholder)."""
        # TODO: Implement actual rate limiting logic
        return False