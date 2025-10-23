# ML API Platform - Modular Architecture

This directory contains the restructured FastAPI application with a clean, modular architecture.

## Directory Structure

```
app/
├── main.py                  # Entry point with FastAPI app and router registration
├── api/
│   └── routes/              # API route modules with APIRouter
│       ├── predictions.py   # Prediction endpoints (/api/v1/predict)
│       ├── health.py        # Health check endpoints (/api/v1/health/*)
│       ├── models.py        # Model information endpoints (/api/v1/model/*)
│       └── __init__.py
├── schemas/                 # Pydantic models for input/output validation
│   ├── prediction.py        # Prediction request/response schemas
│   ├── health.py           # Health check schemas
│   ├── error.py            # Error response schemas
│   └── __init__.py
├── models/                  # Database/ORM models (future use)
│   ├── user.py             # User model for future authentication
│   ├── prediction_log.py   # Prediction logging model
│   └── __init__.py
├── services/                # Business logic layer
│   ├── prediction_service.py # Prediction business logic
│   ├── model_service.py     # Model loading and management
│   ├── auth_service.py      # Authentication logic
│   └── __init__.py
├── core/                    # Core configurations and setup
│   ├── config.py           # Application configuration with pydantic-settings
│   ├── database.py         # Database setup (future use)
│   ├── security.py         # Security and authentication setup
│   └── __init__.py
└── utils/                   # Helper functions and utilities
    ├── logging.py          # Structured logging utilities
    ├── metrics.py          # Metrics collection
    └── __init__.py
```

## Key Features

### Modular Design
- **Clear separation of concerns**: Each module has a single responsibility
- **APIRouter usage**: Each route module uses FastAPI's APIRouter with proper prefixes and tags
- **Service layer**: Business logic is encapsulated in service classes
- **Dependency injection**: Services are injected as dependencies for better testability

### Configuration Management
- **Environment-based config**: Uses pydantic-settings for type-safe configuration
- **Multiple config sections**: API, MLflow, Security, and Monitoring configurations
- **Environment variable support**: All settings can be overridden via environment variables

### Security
- **API key authentication**: Secure token-based authentication
- **CORS configuration**: Configurable cross-origin resource sharing
- **Security headers**: Automatic security headers for production
- **Input validation**: Comprehensive request validation with detailed error responses

### Observability
- **Structured logging**: JSON-formatted logs with request tracking
- **Metrics collection**: Performance and usage metrics
- **Health checks**: Kubernetes-style readiness and liveness probes
- **Error tracking**: Detailed error logging and categorization

## Running the Application

### Development
```bash
# From project root
python run_server.py

# Or directly
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

All endpoints are prefixed with `/api/v1`:

- **POST /api/v1/predict** - Make predictions (requires authentication)
- **GET /api/v1/health/ready** - Readiness probe
- **GET /api/v1/health/live** - Liveness probe  
- **GET /api/v1/model/info** - Model information (requires authentication)
- **GET /api/v1/model/metrics** - System metrics (requires authentication)

## Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Configuration

The application uses environment variables for configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MODEL_NAME=wine_classifier
MODEL_STAGE=production
API_KEY=your-secure-api-key
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
HTTPS_ONLY=false

# MLflow Configuration  
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_ARTIFACT_ROOT=./mlruns
MLFLOW_S3_BUCKET=your-s3-bucket

# Security Configuration
SECRETS_MANAGER_ENABLED=false
PARAMETER_STORE_ENABLED=false

# Monitoring Configuration
CLOUDWATCH_ENABLED=false
SNS_TOPIC_ARN=arn:aws:sns:region:account:topic
```

## Migration from Old Structure

The old monolithic structure in `src/api/` has been replaced with this modular architecture. Key changes:

1. **Routes**: Moved from `src/api/main.py` to individual route modules in `app/api/routes/`
2. **Schemas**: Extracted from `src/api/models.py` to domain-specific files in `app/schemas/`
3. **Services**: Business logic extracted to `app/services/`
4. **Configuration**: Centralized in `app/core/config.py` with pydantic-settings
5. **Entry point**: Changed from `src.api.main:app` to `app.main:app`

## Testing

```bash
# Test basic import
python -c "from app.main import app; print('✓ App imported successfully')"

# Test health endpoint
python -c "
from app.main import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.get('/api/v1/health/live')
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')
"
```