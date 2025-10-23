"""
ML API Platform - Modular FastAPI Application

This package contains the restructured FastAPI application with clear separation of concerns:
- api/routes/: API route modules with APIRouter
- schemas/: Pydantic models for input/output validation  
- models/: Database/ORM models
- services/: Business logic layer
- core/: Core configurations and setup
- utils/: Helper functions and utilities
"""

__version__ = "1.0.0"