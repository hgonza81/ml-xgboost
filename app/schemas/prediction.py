"""
Prediction-related Pydantic schemas for request/response validation.
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