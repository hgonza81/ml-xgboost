"""
Health check API routes.
"""

from fastapi import APIRouter, Depends
import logging
from datetime import datetime

from app.schemas.health import HealthResponse
from app.services.model_service import ModelService
from app.utils.logging import check_logging_health
from app.utils.metrics import metrics

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)


def get_model_service() -> ModelService:
    """Dependency to get model service instance."""
    # Import here to avoid circular imports
    from src.api.model_loader import get_model_loader
    model_loader = get_model_loader()
    return ModelService(model_loader=model_loader)


@router.get("/ready", response_model=HealthResponse)
async def health_ready(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
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
    
    # Check model service health
    try:
        model_health = model_service.health_check()
        checks.update(model_health)
    except Exception as e:
        logger.warning(f"Model service health check failed: {e}")
        checks["model_service"] = "unhealthy"
        checks["model_service_error"] = str(e)
    
    # Get metrics health
    health_metrics = metrics.get_health_metrics()
    checks["metrics"] = "healthy" if health_metrics["is_healthy"] else "unhealthy"
    checks["error_rate"] = str(health_metrics["error_rate"])
    
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


@router.get("/live", response_model=HealthResponse)
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