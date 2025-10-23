"""
AWS Lambda handler for FastAPI application.
Optimized for Lambda container deployment.
"""

import os
from mangum import Mangum
from src.api.main import app

# Set production environment
os.environ.setdefault("ENVIRONMENT", "production")

# Create Lambda handler
handler = Mangum(app, lifespan="off")