"""
Authentication service for API key validation and security.
"""

import logging
import os
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""
    pass


class AuthService:
    """
    Service class for authentication and authorization logic.
    """
    
    def __init__(self):
        """Initialize authentication service with configuration."""
        self.api_key = os.getenv("API_KEY", "dev-api-key-12345")
        self.cors_origins = self._parse_cors_origins()
        logger.info(f"Auth service initialized with CORS origins: {self.cors_origins}")
    
    def _parse_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
        if cors_env:
            origins = [origin.strip() for origin in cors_env.split(",")]
            return origins
        return ["http://localhost:3000", "http://localhost:8080"]  # Default for development
    
    async def verify_api_key(self, credentials: str) -> str:
        """
        Verify the API key.
        
        Args:
            credentials: API key to verify
            
        Returns:
            str: The verified API key
            
        Raises:
            AuthenticationError: If the API key is invalid
        """
        if not credentials:
            logger.warning("Missing API key credentials")
            raise AuthenticationError("Missing API key credentials")
        
        if credentials != self.api_key:
            logger.warning(f"Invalid API key attempt: {credentials[:10]}...")
            raise AuthenticationError("Invalid API key")
        
        logger.debug("API key validated successfully")
        return credentials
    
    def get_cors_origins(self) -> List[str]:
        """
        Get configured CORS origins.
        
        Returns:
            List of allowed CORS origins
        """
        return self.cors_origins
    
    def log_authentication_attempt(self, success: bool, client_info: Optional[str] = None):
        """
        Log authentication attempt for monitoring.
        
        Args:
            success: Whether authentication was successful
            client_info: Optional client information for logging
        """
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.utcnow().isoformat()
        
        if client_info:
            logger.info(f"Authentication {status} at {timestamp} for client: {client_info}")
        else:
            logger.info(f"Authentication {status} at {timestamp}")
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """
        Check if client is rate limited (placeholder for future implementation).
        
        Args:
            client_ip: Client IP address
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        # TODO: Implement actual rate limiting logic
        return False