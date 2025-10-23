"""
Security configuration and middleware setup.
"""

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from typing import Optional

from app.services.auth_service import AuthService, AuthenticationError
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Security scheme for API key authentication
security = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
    auth_service: AuthService = Depends(lambda: AuthService())
) -> str:
    """
    Verify the API key from the Authorization header.
    
    Args:
        credentials: HTTP authorization credentials containing the API key
        auth_service: Authentication service instance
        
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
    
    try:
        api_key = await auth_service.verify_api_key(credentials.credentials)
        auth_service.log_authentication_attempt(success=True)
        return api_key
    except AuthenticationError as e:
        auth_service.log_authentication_attempt(success=False)
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_api_key(api_key: str = Depends(verify_api_key)) -> str:
    """
    Dependency to get the current validated API key.
    
    Args:
        api_key: The verified API key from verify_api_key
        
    Returns:
        str: The validated API key
    """
    return api_key


def setup_cors(app, auth_service: AuthService):
    """
    Set up CORS middleware with configured origins.
    
    Args:
        app: FastAPI application instance
        auth_service: Authentication service for CORS configuration
    """
    
    cors_origins = auth_service.get_cors_origins()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    logger.info(f"CORS middleware configured with origins: {cors_origins}")


def setup_security_middleware(app):
    """
    Set up security middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    settings = get_settings()
    
    # Add trusted host middleware for production
    if settings.api.https_only:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual allowed hosts in production
        )
        logger.info("Trusted host middleware configured")
    
    # Add custom security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        if settings.api.https_only:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    logger.info("Security middleware configured")


class AuthenticationMiddleware:
    """
    Custom authentication middleware for request validation and logging.
    """
    
    def __init__(self, app):
        self.app = app
        self.auth_service = AuthService()
    
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