"""
Comprehensive logging configuration for the ML API Platform.

This module implements:
- Structured logging with JSON format
- Request/response logging middleware
- Error logging with stack traces
- Performance monitoring and metrics
"""

import logging
import logging.config
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import os

# Configure structlog for structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get structured logger
logger = structlog.get_logger(__name__)


class LoggingConfig:
    """Configuration for logging settings."""
    
    def __init__(self):
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json")  # json or text
        self.log_requests = os.getenv("LOG_REQUESTS", "true").lower() == "true"
        self.log_responses = os.getenv("LOG_RESPONSES", "false").lower() == "true"
        
        # Performance monitoring settings
        self.slow_request_threshold = float(os.getenv("SLOW_REQUEST_THRESHOLD", "1.0"))  # seconds
        
        logger.info("Logging configuration initialized", 
                   level=self.log_level, 
                   format=self.log_format,
                   log_requests=self.log_requests)

# Global logging configuration
logging_config = LoggingConfig()


def setup_logging():
    """Set up comprehensive logging configuration."""
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, logging_config.log_level),
        format='%(message)s' if logging_config.log_format == 'json' else 
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # Reduce noise
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    logger.info("Logging system initialized")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request/response logging with performance monitoring.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Log requests, responses, and performance metrics.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in the chain
            
        Returns:
            Response object with added logging
        """
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request information
        method = request.method
        url = str(request.url)
        path = request.url.path
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Skip logging for health checks and docs to reduce noise
        skip_paths = ["/health/ready", "/health/live", "/docs", "/redoc", "/openapi.json"]
        should_log = path not in skip_paths
        
        if should_log and logging_config.log_requests:
            logger.info("Request started",
                       request_id=request_id,
                       method=method,
                       path=path,
                       client_ip=client_ip,
                       user_agent=user_agent)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response information
            if should_log:
                log_data = {
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 3),
                    "client_ip": client_ip
                }
                
                # Log slow requests as warnings
                if process_time > logging_config.slow_request_threshold:
                    logger.warning("Slow request detected", **log_data)
                elif logging_config.log_responses:
                    logger.info("Request completed", **log_data)
            
            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log errors with full context
            process_time = time.time() - start_time
            
            logger.error("Request failed",
                        request_id=request_id,
                        method=method,
                        path=path,
                        error=str(e),
                        process_time=round(process_time, 3),
                        client_ip=client_ip,
                        exc_info=True)
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers."""
        
        # Check for forwarded headers (common in load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


class ErrorLogger:
    """Centralized error logging with context and stack traces."""
    
    @staticmethod
    def log_validation_error(request_id: str, path: str, errors: list):
        """Log validation errors with context."""
        logger.warning("Validation error",
                      request_id=request_id,
                      path=path,
                      validation_errors=errors)
    
    @staticmethod
    def log_authentication_error(request_id: str, path: str, reason: str):
        """Log authentication failures."""
        logger.warning("Authentication failed",
                      request_id=request_id,
                      path=path,
                      reason=reason)
    
    @staticmethod
    def log_internal_error(request_id: str, path: str, error: Exception):
        """Log internal server errors with full stack trace."""
        logger.error("Internal server error",
                    request_id=request_id,
                    path=path,
                    error=str(error),
                    exc_info=True)
    
    @staticmethod
    def log_prediction_error(request_id: str, samples_count: int, error: Exception):
        """Log prediction-specific errors."""
        logger.error("Prediction processing failed",
                    request_id=request_id,
                    samples_count=samples_count,
                    error=str(error),
                    exc_info=True)


class MetricsCollector:
    """
    Simple metrics collection for monitoring.
    
    In production, this would integrate with CloudWatch or other monitoring systems.
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.prediction_count = 0
        self.total_processing_time = 0.0
        self.start_time = datetime.utcnow()
    
    def record_request(self, processing_time: float, status_code: int):
        """Record request metrics."""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if status_code >= 400:
            self.error_count += 1
    
    def record_prediction(self, samples_count: int):
        """Record prediction metrics."""
        self.prediction_count += samples_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_count / max(self.request_count, 1), 3),
            "total_predictions": self.prediction_count,
            "average_processing_time": round(avg_processing_time, 3),
            "requests_per_second": round(self.request_count / max(uptime, 1), 3)
        }

# Global metrics collector
metrics = MetricsCollector()


def add_logging_middleware(app):
    """
    Add comprehensive logging middleware to the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    # Set up logging configuration
    setup_logging()
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Logging middleware configured and added")


# Health check function for logging system
def check_logging_health() -> Dict[str, str]:
    """Check if logging system is functioning properly."""
    try:
        # Test log message
        logger.info("Logging health check")
        return {"logging": "healthy"}
    except Exception as e:
        return {"logging": f"unhealthy: {str(e)}"}