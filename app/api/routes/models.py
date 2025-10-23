"""
Model information API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
import logging
from datetime import datetime
from typing import Dict, Any

from app.services.model_service import ModelService, ModelLoadError
from app.core.security import get_current_api_key
from app.core.config import get_settings
from app.utils.metrics import metrics

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/model",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)


def get_model_service() -> ModelService:
    """Dependency to get model service instance."""
    # Import here to avoid circular imports
    from src.api.model_loader import get_model_loader
    model_loader = get_model_loader()
    return ModelService(model_loader=model_loader)


@router.get("/info")
async def get_model_info(
    model_service: ModelService = Depends(get_model_service),
    api_key: str = Depends(get_current_api_key)
) -> Dict[str, Any]:
    """
    Model information endpoint providing details about the currently configured model.
    
    Requires authentication and returns model metadata from MLflow registry.
    
    Args:
        model_service: Model service for retrieving model information
        api_key: Validated API key from authentication middleware
        
    Returns:
        Dictionary containing model information and metadata
    """
    try:
        settings = get_settings()
        model_name = settings.api.model_name
        model_stage = settings.api.model_stage
        
        model_info = await model_service.get_model_info(model_name, model_stage)
        
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
        metrics.record_error("model_info_error")
        raise HTTPException(
            status_code=503,
            detail=f"Model information unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting model info: {e}")
        metrics.record_error("internal_error")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@router.get("/metrics")
async def get_model_metrics(
    api_key: str = Depends(get_current_api_key)
) -> Dict[str, Any]:
    """
    Model metrics endpoint for monitoring and observability.
    
    Requires authentication and returns current model usage metrics.
    
    Args:
        api_key: Validated API key from authentication middleware
        
    Returns:
        Dictionary containing current model metrics and performance data
    """
    current_metrics = metrics.get_metrics()
    
    logger.info(f"Model metrics requested: {current_metrics}")
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ml-api-platform",
        "version": "1.0.0",
        "metrics": current_metrics
    }