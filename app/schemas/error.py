"""
Error response Pydantic schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


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