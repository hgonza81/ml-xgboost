"""
Health check related Pydantic schemas.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional


class HealthResponse(BaseModel):
    """
    Response model for health check endpoints.
    """
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO timestamp of health check")
    version: str = Field(..., description="Application version")
    checks: Optional[Dict[str, str]] = Field(None, description="Individual component health checks")