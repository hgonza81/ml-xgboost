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

## Technology Stack

### 🐍 Base Language
- **Python 3.11+** – Main programming language of the project

---

### 🚀 Web Framework & API
- **FastAPI (>=0.104.0)** – Modern, high-performance web framework for APIs  
- **Uvicorn (>=0.24.0)** – ASGI server for running FastAPI  
- **Pydantic (>=2.5.0)** – Data validation and serialization  
- **python-multipart (>=0.0.6)** – Multipart form data handling  

---

### 🤖 Machine Learning & MLOps
- **MLflow (>=2.8.0)** – MLOps platform for experiment tracking and model registry  
- **XGBoost (>=2.0.0)** – Core machine learning algorithm  
- **scikit-learn (>=1.3.0)** – Complementary ML library  
- **Pandas (>=2.1.0)** – Data manipulation  
- **NumPy (>=1.24.0)** – Numerical computing  
- **SciPy (>=1.11.0)** – Scientific algorithms  
- **Joblib (>=1.3.0)** – Model serialization  

---

### ☁️ Cloud Services (AWS)
- **Boto3 (>=1.34.0)** – AWS SDK for Python  
- **Botocore (>=1.34.0)** – Core AWS SDK components  

---

### 🌐 HTTP & Networking
- **HTTPx (>=0.25.0)** – Asynchronous HTTP client  
- **Requests (>=2.31.0)** – Traditional synchronous HTTP client  

---

### 📊 Logging & Monitoring
- **Structlog (>=23.2.0)** – Structured logging  
- **python-json-logger (>=2.0.7)** – JSON logging formatter  

---

### ⚙️ Configuration
- **pydantic-settings (>=2.1.0)** – Configuration management  
- **python-dotenv (>=1.0.0)** – Environment variable management  

---

### 🔍 Hyperparameter Optimization
- **Optuna (>=3.4.0)** – Hyperparameter optimization framework  
- **optuna-integration[mlflow] (>=3.4.0)** – MLflow integration for Optuna  

---

### 🐳 Containerization & Orchestration
- **Docker** – Containerization for services  
- **Docker Compose** – Local orchestration  
- **AWS Lambda** – Serverless deployment (via `Dockerfile.lambda`)  

---

### 🧪 Testing & Code Quality (Dev Dependencies)
- **pytest (>=7.4.0)** – Testing framework  
- **pytest-asyncio (>=0.21.0)** – Async testing support  
- **pytest-cov (>=4.1.0)** – Code coverage reporting  
- **black (>=23.11.0)** – Code formatter  
- **isort (>=5.12.0)** – Import sorter  
- **flake8 (>=6.1.0)** – Linter  
- **mypy (>=1.7.0)** – Static type checker  
- **pre-commit (>=3.6.0)** – Git hooks automation  

---

### 📈 Production Monitoring (Optional)
- **prometheus-client (>=0.19.0)** – Prometheus metrics exporter  
- **sentry-sdk[fastapi] (>=1.38.0)** – Error tracking and tracing  

---

### 🗄️ Databases (Production)
- **psycopg2-binary (>=2.9.9)** – PostgreSQL adapter  
- **pymysql (>=1.1.0)** – MySQL adapter  
- **SQLite** – Default lightweight database for MLflow  

---

### 🔧 Development Tools
- **jupyter (>=1.0.0)** – Interactive notebooks for analysis  
- **ipython (>=8.17.0)** – Enhanced interactive Python shell  
- **Make** – Task automation (Makefile)  

---

### 🏗️ System Architecture
- **Microservices** – Separate API, MLflow, and Training services  
- **RESTful API** – Standard REST endpoints  
- **API Key Authentication** – Security mechanism  
- **CORS** – Cross-Origin Resource Sharing policy  
- **Health Checks** – Service health monitoring  
- **Multi-stage Docker builds** – Optimized container images  

---

### 📁 Configuration Structure
- **Environment-based config** – Development vs. Production  
- **Pydantic Settings** – Validated configuration system  
- **Docker environment files** – Environment-specific variables  

---

### 🔄 Workflow
- **MLflow Model Registry** – Centralized model versioning  
- **Hyperparameter Optimization** – Tuning with Optuna  
- **Model Promotion Pipeline** – Controlled model rollout  
- **Experiment Tracking** – Complete experiment traceability  