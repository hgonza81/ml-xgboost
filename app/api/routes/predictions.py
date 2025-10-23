"""
Prediction API routes.
"""

from fastapi import APIRouter, HTTPException, Depends
import logging
import uuid
from datetime import datetime

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.services.model_service import ModelService, ModelLoadError
from app.core.security import get_current_api_key
from app.core.config import get_settings
from app.utils.metrics import metrics

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/predict",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)


def get_model_service() -> ModelService:
    """Dependency to get model service instance."""
    # Import here to avoid circular imports
    from src.api.model_loader import get_model_loader
    model_loader = get_model_loader()
    return ModelService(model_loader=model_loader)


def get_prediction_service(model_service: ModelService = Depends(get_model_service)) -> PredictionService:
    """Dependency to get prediction service instance."""
    return PredictionService(model_service=model_service)


@router.post("", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    api_key: str = Depends(get_current_api_key)
) -> PredictionResponse:
    """
    Prediction endpoint that accepts JSON tabular data with validation.
    
    This endpoint requires API key authentication, validates input data 
    structure and types, loads the XGBoost model from MLflow registry,
    and returns classification predictions with model metadata.
    
    Args:
        request: Validated prediction request containing tabular data
        prediction_service: Prediction service for business logic
        api_key: Validated API key from authentication middleware
        
    Returns:
        PredictionResponse with predictions and metadata
        
    Raises:
        HTTPException: For authentication (401), validation (400), model errors (503), or internal errors (500)
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Log the authenticated request
        logger.info(f"Processing authenticated prediction request {request_id} with {len(request.data)} samples")
        
        # Record prediction metrics
        metrics.record_prediction(len(request.data))
        
        # Get application settings
        settings = get_settings()
        model_name = settings.api.model_name
        model_stage = settings.api.model_stage
        
        # Process prediction through service layer
        response = await prediction_service.predict(request, model_name, model_stage)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Record successful prediction metrics
        metrics.record_prediction(
            sample_count=len(request.data),
            duration_ms=duration_ms,
            model_version=response.model_version
        )
        
        logger.info(f"Successfully processed request {request_id} with {len(request.data)} samples using model {response.model_version}")
        return response
        
    except ModelLoadError as e:
        logger.error(f"Model loading failed for request {request_id}: {e}")
        metrics.record_error("model_load_error")
        raise HTTPException(
            status_code=503,
            detail=f"Model service unavailable: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Data validation failed for request {request_id}: {e}")
        metrics.record_error("data_validation_error")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error processing request {request_id}: {str(e)}", exc_info=True)
        metrics.record_error("internal_error")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the prediction request"
        )