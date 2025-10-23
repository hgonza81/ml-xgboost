"""
FastAPI main application with modular architecture.

This module implements the restructured FastAPI application with:
- Automatic router registration from api/routes/
- Modular organization with clear separation of concerns
- Centralized middleware and exception handling
- Relative imports throughout the application
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from typing import Dict
import logging
import uuid
from datetime import datetime

# Import routers
from app.api.routes import predictions, health, models

# Import schemas
from app.schemas.error import ErrorResponse, ErrorDetail

# Import core modules
from app.core.config import get_settings
from app.core.security import setup_cors, setup_security_middleware, add_auth_middleware
from app.services.auth_service import AuthService

# Import utilities
from app.utils.logging import add_logging_middleware, ErrorLogger
from app.utils.metrics import metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Get settings
    settings = get_settings()
    
    # Create FastAPI application instance
    app = FastAPI(
        title="ML API Platform",
        description="Machine Learning API Platform for classification predictions with modular architecture",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Initialize services
    auth_service = AuthService()
    
    # Set up CORS middleware (must be first)
    setup_cors(app, auth_service)
    
    # Add logging middleware (should be added before auth middleware)
    add_logging_middleware(app)
    
    # Add security middleware
    setup_security_middleware(app)
    
    # Add authentication middleware
    add_auth_middleware(app)
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register all routers
    register_routers(app)
    
    # Add root endpoint
    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint providing basic API information."""
        return {
            "message": "ML API Platform - Modular Architecture",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "architecture": "modular"
        }
    
    logger.info("FastAPI application created with modular architecture")
    return app


def register_routers(app: FastAPI):
    """
    Register all API routers with the application.
    
    Args:
        app: FastAPI application instance
    """
    # Register prediction routes
    app.include_router(predictions.router, prefix="/api/v1")
    
    # Register health check routes
    app.include_router(health.router, prefix="/api/v1")
    
    # Register model information routes
    app.include_router(models.router, prefix="/api/v1")
    
    logger.info("All API routers registered successfully")


def register_exception_handlers(app: FastAPI):
    """
    Register global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors with detailed error information."""
        
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Convert Pydantic errors to our ErrorDetail format
        error_details = []
        for error in exc.errors():
            error_details.append(ErrorDetail(
                field=".".join(str(loc) for loc in error["loc"]),
                message=error["msg"],
                type=error["type"],
                input=error.get("input")
            ))
        
        # Log validation error with context
        ErrorLogger.log_validation_error(request_id, request.url.path, exc.errors())
        
        # Record metrics
        metrics.record_error("validation_error")
        
        error_response = ErrorResponse.create_validation_error(
            message="Request validation failed",
            details=error_details
        )
        error_response.request_id = request_id
        
        return JSONResponse(
            status_code=400,
            content=error_response.dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions with proper error responses."""
        
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Log internal error with full context
        ErrorLogger.log_internal_error(request_id, request.url.path, exc)
        
        # Record metrics
        metrics.record_error("internal_error")
        
        error_response = ErrorResponse.create_internal_error(
            message="An internal server error occurred",
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )
    
    logger.info("Exception handlers registered")


# Create the application instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        log_level=settings.api.log_level.lower()
    )