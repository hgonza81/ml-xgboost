"""
Pydantic models for request/response validation.

This module defines the data models used for:
- Request validation for the /predict endpoint
- Response validation and structure
- Error handling with HTTP 400 responses
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureRow(BaseModel):
    """
    Represents a single row of tabular data with dynamic features.
    
    This model accepts any combination of string, integer, and float values
    to accommodate various tabular datasets.
    """
    
    class Config:
        extra = "allow"  # Allow additional fields not explicitly defined
        
    def __init__(self, **data):
        # Validate that all values are of acceptable types
        for key, value in data.items():
            if not isinstance(value, (str, int, float, type(None))):
                raise ValueError(f"Feature '{key}' has invalid type {type(value)}. Only str, int, float, and None are allowed.")
        super().__init__(**data)


class PredictionRequest(BaseModel):
    """
    Request model for the /predict endpoint.
    
    Validates that the request contains a 'data' field with a list of feature rows.
    """
    
    data: List[Dict[str, Union[str, int, float, None]]] = Field(
        ...,
        description="List of feature dictionaries for prediction",
        min_items=1,
        max_items=1000  # Reasonable limit to prevent abuse
    )
    
    @validator('data')
    def validate_data_structure(cls, v):
        """Validate that each row in data has consistent structure."""
        if not v:
            raise ValueError("Data list cannot be empty")
        
        # Check that all rows have at least one feature
        for i, row in enumerate(v):
            if not row:
                raise ValueError(f"Row {i} cannot be empty")
            
            # Validate feature values
            for feature_name, feature_value in row.items():
                if not isinstance(feature_name, str):
                    raise ValueError(f"Feature names must be strings, got {type(feature_name)} for row {i}")
                
                if feature_value is not None and not isinstance(feature_value, (str, int, float)):
                    raise ValueError(
                        f"Feature '{feature_name}' in row {i} has invalid type {type(feature_value)}. "
                        "Only str, int, float, and None are allowed."
                    )
        
        return v


class ModelMetadata(BaseModel):
    """
    Model metadata included in prediction responses.
    """
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage (production, staging, etc.)")
    run_id: Optional[str] = Field(None, description="MLflow run ID")


class PredictionResponse(BaseModel):
    """
    Response model for successful predictions.
    """
    
    predictions: List[Union[int, float]] = Field(
        ...,
        description="List of prediction results"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for predictions"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp when predictions were made"
    )
    samples_processed: int = Field(
        ...,
        description="Number of samples that were processed"
    )
    model_metadata: Optional[ModelMetadata] = Field(
        None,
        description="Additional model metadata from MLflow registry"
    )


class ErrorDetail(BaseModel):
    """
    Detailed error information for validation failures.
    """
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    input: Optional[Any] = Field(None, description="Input value that caused the error")


class ErrorResponse(BaseModel):
    """
    Standardized error response model.
    """
    
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(..., description="ISO timestamp when error occurred")
    request_id: Optional[str] = Field(None, description="Unique request identifier for tracking")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed validation errors")
    
    @classmethod
    def create_validation_error(cls, message: str, details: List[ErrorDetail] = None) -> "ErrorResponse":
        """Create a validation error response."""
        return cls(
            error="validation_error",
            message=message,
            timestamp=datetime.utcnow().isoformat() + "Z",
            details=details or []
        )
    
    @classmethod
    def create_internal_error(cls, message: str, request_id: str = None) -> "ErrorResponse":
        """Create an internal server error response."""
        return cls(
            error="internal_error",
            message=message,
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id
        )


class HealthResponse(BaseModel):
    """
    Response model for health check endpoints.
    """
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO timestamp of health check")
    version: str = Field(..., description="Application version")
    checks: Optional[Dict[str, str]] = Field(None, description="Individual component health checks")