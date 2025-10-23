"""
Logging utilities and configuration.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
import time


class StructuredLogger:
    """Structured logging utility for consistent log formatting."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_request(self, request: Request, request_id: str, start_time: float):
        """Log incoming request details."""
        log_data = {
            "event": "request_received",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat(),
            "start_time": start_time
        }
        self.logger.info(json.dumps(log_data))
    
    def log_response(self, request: Request, response: Response, request_id: str, 
                    start_time: float, end_time: float):
        """Log response details."""
        duration = end_time - start_time
        log_data = {
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, request: Request, error: Exception, request_id: str):
        """Log error details."""
        log_data = {
            "event": "request_error",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.error(json.dumps(log_data))


class ErrorLogger:
    """Specialized logger for error tracking and analysis."""
    
    @staticmethod
    def log_validation_error(request_id: str, path: str, errors: list):
        """Log validation errors with detailed context."""
        logger = logging.getLogger("validation_errors")
        log_data = {
            "event": "validation_error",
            "request_id": request_id,
            "path": path,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.warning(json.dumps(log_data))
    
    @staticmethod
    def log_internal_error(request_id: str, path: str, error: Exception):
        """Log internal server errors."""
        logger = logging.getLogger("internal_errors")
        log_data = {
            "event": "internal_error",
            "request_id": request_id,
            "path": path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.error(json.dumps(log_data), exc_info=True)


class LoggingMiddleware:
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app):
        self.app = app
        self.logger = StructuredLogger("request_logger")
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware for request/response logging."""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to headers for tracking
        scope["headers"] = list(scope.get("headers", []))
        scope["headers"].append((b"x-request-id", request_id.encode()))
        
        # Log incoming request
        self.logger.log_request(request, request_id, start_time)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Create response object for logging
                response = Response()
                response.status_code = message["status"]
                
                end_time = time.time()
                self.logger.log_response(request, response, request_id, start_time, end_time)
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.log_error(request, e, request_id)
            raise


def add_logging_middleware(app):
    """
    Add logging middleware to the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(LoggingMiddleware)
    logging.getLogger(__name__).info("Logging middleware added")


def check_logging_health() -> Dict[str, str]:
    """
    Check logging system health.
    
    Returns:
        Dictionary with logging health status
    """
    try:
        # Test logging functionality
        test_logger = logging.getLogger("health_check")
        test_logger.info("Logging health check")
        
        return {
            "logging": "healthy",
            "structured_logging": "enabled"
        }
    except Exception as e:
        return {
            "logging": "unhealthy",
            "logging_error": str(e)
        }