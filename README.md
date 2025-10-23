# ML API Platform

A complete machine learning platform with FastAPI prediction service, MLflow experiment tracking, and containerized deployment.

## Project Structure

```
ml-api-platform/
├── src/                          # Source code
│   ├── api/                      # FastAPI prediction service
│   ├── mlflow_components/        # MLflow tracking and registry
│   ├── training/                 # Model training pipeline
│   └── __init__.py
├── config/                       # Configuration management
│   ├── settings.py              # Pydantic settings with validation
│   ├── development.env          # Development environment config
│   ├── production.env           # Production environment config
│   └── __init__.py
├── deployment/                   # Deployment configurations
│   ├── lambda_handler.py        # AWS Lambda handler
│   └── __init__.py
├── tests/                        # Test suite
│   └── __init__.py
├── data/                         # Data storage (created at runtime)
├── Dockerfile.api               # FastAPI service container
├── Dockerfile.mlflow            # MLflow server container
├── Dockerfile.lambda            # AWS Lambda container
├── docker-compose.yml           # Development orchestration
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── Makefile                    # Development commands
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Quick Start

### Development Environment

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ml-api-platform
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start development environment**:
   ```bash
   make dev
   # or
   docker-compose up -d
   ```

3. **Access services**:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000

### Local Development (without Docker)

1. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Set environment**:
   ```bash
   export ENVIRONMENT=development
   source config/development.env
   ```

## Configuration

The platform uses environment-specific configuration with validation:

- **Development**: Uses SQLite for MLflow backend, local file storage
- **Production**: Uses S3 for artifacts, AWS services for secrets and monitoring

Configuration is managed through:
- `config/settings.py`: Pydantic models with validation
- `config/development.env`: Development environment variables
- `config/production.env`: Production environment template
- `.env`: Local overrides (not committed)

## Available Commands

```bash
make help           # Show all available commands
make install-dev    # Install development dependencies
make test           # Run test suite
make lint           # Run code quality checks
make format         # Format code
make build          # Build Docker images
make up             # Start development environment
make down           # Stop development environment
make logs           # Show service logs
make train          # Run training pipeline
```

## Environment Variables

Key environment variables (see `.env.example` for complete list):

- `ENVIRONMENT`: `development` or `production`
- `API_KEY`: Authentication key for API access
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MODEL_NAME`: Model name in MLflow registry
- `MODEL_STAGE`: Model stage to load (`staging`, `production`)

## Next Steps

This is the foundational project structure. The next tasks will implement:

1. FastAPI prediction service with authentication
2. MLflow integration for model loading
3. Model training pipeline with XGBoost
4. CI/CD pipeline with GitHub Actions
5. Production deployment to AWS Lambda

## Development Workflow

1. Make changes to source code
2. Run tests: `make test`
3. Check code quality: `make lint`
4. Format code: `make format`
5. Build and test locally: `make up`
6. Commit changes and push

## Architecture

The platform follows a microservices architecture:

- **API Service**: FastAPI application serving predictions
- **MLflow Service**: Experiment tracking and model registry
- **Training Service**: Model training and hyperparameter optimization
- **Deployment**: Container-based deployment to AWS Lambda

All services are containerized and orchestrated with Docker Compose for development, with production deployment to AWS Lambda using container images.