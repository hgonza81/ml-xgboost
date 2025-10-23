"""
FastAPI main application module.

This module implements the core FastAPI prediction service with:
- Main FastAPI app with basic routing structure
- /predict endpoint that accepts JSON tabular data with validation
- API key-based authentication system
- CORS policy enforcement with whitelisted domains
- OpenAPI documentation endpoints (/docs, /redoc)
- Proper error handling for invalid data with HTTP 400 responses
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from typing import Dict, Any, List
import logging
import uuid
from datetime import datetime
from pydantic import ValidationError
import pandas as pd
import numpy as np

from .models import (
    PredictionRequest, 
    PredictionResponse, 
    ErrorResponse, 
    ErrorDetail,
    HealthResponse,
    ModelMetadata
)
from .auth import (
    get_current_api_key,
    setup_cors,
    add_auth_middleware
)
from .logging_config import (
    add_logging_middleware,
    metrics,
    ErrorLogger,
    check_logging_health
)
from .model_loader import (
    get_model_loader,
    ModelLoadError,
    MLflowModelLoader
)
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="ML API Platform",
    description="Machine Learning API Platform for classification predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Set up CORS middleware
setup_cors(app)

# Add logging middleware (should be added before auth middleware)
add_logging_middleware(app)

# Add authentication middleware
add_auth_middleware(app)


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
    
    error_response = ErrorResponse.create_internal_error(
        message="An internal server error occurred",
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing basic API information."""
    return {
        "message": "ML API Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(get_current_api_key)
) -> PredictionResponse:
    """
    Prediction endpoint that accepts JSON tabular data with validation.
    
    This endpoint requires API key authentication, validates input data 
    structure and types, loads the XGBoost model from MLflow registry,
    and returns classification predictions with model metadata.
    
    Args:
        request: Validated prediction request containing tabular data
        api_key: Validated API key from authentication middleware
        
    Returns:
        PredictionResponse with predictions and metadata
        
    Raises:
        HTTPException: For authentication (401), validation (400), model errors (503), or internal errors (500)
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Log the authenticated request
        logger.info(f"Processing authenticated prediction request {request_id} with {len(request.data)} samples")
        
        # Record prediction metrics
        metrics.record_prediction(len(request.data))
        
        # Get application settings
        settings = get_settings()
        model_name = settings.api.model_name
        model_stage = settings.api.model_stage
        
        # Get model loader instance
        model_loader = get_model_loader()
        
        # Convert input data to pandas DataFrame
        try:
            input_df = pd.DataFrame(request.data)
            logger.info(f"Converted {len(input_df)} samples to DataFrame with columns: {list(input_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to convert input data to DataFrame: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data format for model input: {str(e)}"
            )
        
        # Load model and make predictions
        try:
            model, metadata = model_loader.load_model(model_name, model_stage)
            logger.info(f"Loaded model '{metadata.name}' version '{metadata.version}' from stage '{metadata.stage}'")
            
            # Make predictions using the loaded model
            predictions = model.predict(input_df)
            
            # Convert predictions to list and ensure they are JSON serializable
            if isinstance(predictions, np.ndarray):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)
            
            # Ensure predictions are numeric (int or float)
            processed_predictions = []
            for pred in predictions_list:
                if isinstance(pred, (int, float, np.integer, np.floating)):
                    processed_predictions.append(float(pred) if isinstance(pred, (np.floating, float)) else int(pred))
                else:
                    # Handle string predictions or other types
                    try:
                        processed_predictions.append(float(pred))
                    except (ValueError, TypeError):
                        processed_predictions.append(str(pred))
            
            logger.info(f"Generated {len(processed_predictions)} predictions using model version {metadata.version}")
            
        except ModelLoadError as e:
            logger.error(f"Model loading failed for request {request_id}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Prediction generation failed for request {request_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction processing failed: {str(e)}"
            )
        
        # Create response with model metadata
        model_metadata = ModelMetadata(
            name=metadata.name,
            version=metadata.version,
            stage=metadata.stage,
            run_id=metadata.run_id
        )
        
        response = PredictionResponse(
            predictions=processed_predictions,
            model_version=f"{metadata.name}-v{metadata.version}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            samples_processed=len(request.data),
            model_metadata=model_metadata
        )
        
        logger.info(f"Successfully processed request {request_id} with {len(request.data)} samples using model {metadata.name} v{metadata.version}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the prediction request"
        )


@app.get("/health/ready", response_model=HealthResponse)
async def health_ready() -> HealthResponse:
    """
    Readiness probe endpoint for container orchestration.
    
    Returns:
        HealthResponse indicating if the service is ready to accept requests
    """
    # Check component health
    checks = {
        "api": "ready",
        "auth": "ready"
    }
    checks.update(check_logging_health())
    
    # Check MLflow model loader health
    try:
        model_loader = get_model_loader()
        mlflow_health = model_loader.health_check()
        checks.update(mlflow_health)
    except Exception as e:
        logger.warning(f"MLflow health check failed: {e}")
        checks["mlflow_connection"] = "unhealthy"
        checks["mlflow_error"] = str(e)
    
    # Determine overall status
    overall_status = "ready" if all(
        status in ["ready", "healthy"] for status in checks.values() 
        if not status.startswith("error") and status != "unhealthy"
    ) else "not_ready"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        checks=checks
    )


@app.get("/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    """
    Liveness probe endpoint for container orchestration.
    
    Returns:
        HealthResponse indicating if the service is alive and functioning
    """
    return HealthResponse(
        status="alive",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        checks={
            "api": "alive"
        }
    )


@app.get("/model/info")
async def get_model_info(api_key: str = Depends(get_current_api_key)) -> Dict[str, Any]:
    """
    Model information endpoint providing details about the currently configured model.
    
    Requires authentication and returns model metadata from MLflow registry.
    
    Args:
        api_key: Validated API key from authentication middleware
        
    Returns:
        Dictionary containing model information and metadata
    """
    try:
        settings = get_settings()
        model_name = settings.api.model_name
        model_stage = settings.api.model_stage
        
        model_loader = get_model_loader()
        model_info = model_loader.get_model_info(model_name, model_stage)
        
        logger.info(f"Model info requested for {model_name} in stage {model_stage}")
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "ml-api-platform",
            "configured_model": {
                "name": model_name,
                "stage": model_stage
            },
            "model_info": model_info
        }
        
    except ModelLoadError as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Model information unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(get_current_api_key)) -> Dict[str, Any]:
    """
    Metrics endpoint for monitoring and observability.
    
    Requires authentication and returns current system metrics.
    
    Args:
        api_key: Validated API key from authentication middleware
        
    Returns:
        Dictionary containing current metrics and performance data
    """
    current_metrics = metrics.get_metrics()
    
    logger.info(f"Metrics requested: {current_metrics}")
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ml-api-platform",
        "version": "1.0.0",
        "metrics": current_metrics
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)