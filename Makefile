# ML API Platform Makefile

.PHONY: help install install-dev clean test lint format type-check build up down logs shell train

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  clean        Clean up build artifacts and cache"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  build        Build Docker images"
	@echo "  up           Start development environment"
	@echo "  down         Stop development environment"
	@echo "  logs         Show logs from services"
	@echo "  shell        Open shell in API container"
	@echo "  train        Run training pipeline"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

# Code quality
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

# Docker operations
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash

# Training
train:
	docker-compose --profile training up training

# Development shortcuts
dev: up
	@echo "Development environment started. API available at http://localhost:8000"
	@echo "MLflow UI available at http://localhost:5000"

stop: down

restart: down up