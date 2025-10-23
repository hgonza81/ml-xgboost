"""
Configuration management system with environment-specific settings.
Supports development and production environments with validation.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class APIConfig(BaseModel):
    """FastAPI service configuration"""
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, description="API port")
    model_name: str = Field(default="xgboost_classifier", description="Model name in registry")
    model_stage: str = Field(default="production", description="Model stage to load")
    api_key: str = Field(description="API key for authentication")
    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: List[str] = Field(default=[], description="Whitelisted CORS origins")
    https_only: bool = Field(default=True, description="Enforce HTTPS in production")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()


class MLflowConfig(BaseModel):
    """MLflow tracking and registry configuration"""
    tracking_uri: str = Field(description="MLflow tracking server URI")
    artifact_root: str = Field(description="Artifact storage root path")
    backend_store_uri: str = Field(description="Backend store URI")
    default_artifact_root: str = Field(description="Default artifact root")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for production artifacts")
    aws_region: str = Field(default="us-east-1", description="AWS region")


class SecurityConfig(BaseModel):
    """Security and secrets management configuration"""
    secrets_manager_enabled: bool = Field(default=False, description="Enable AWS Secrets Manager")
    parameter_store_enabled: bool = Field(default=False, description="Enable AWS Parameter Store")
    iam_role_arn: Optional[str] = Field(default=None, description="IAM role ARN")


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    cloudwatch_enabled: bool = Field(default=False, description="Enable CloudWatch metrics")
    sns_topic_arn: Optional[str] = Field(default=None, description="SNS topic for alerts")
    pagerduty_integration_key: Optional[str] = Field(default=None, description="PagerDuty integration key")
    log_retention_days: int = Field(default=30, description="Log retention period in days")


class Settings(BaseModel):
    """Main application settings"""
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    api: APIConfig
    mlflow: MLflowConfig
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        env_nested_delimiter = '__'
        case_sensitive = False


def get_development_settings() -> Settings:
    """Get development environment settings"""
    return Settings(
        environment=Environment.DEVELOPMENT,
        api=APIConfig(
            api_key=os.getenv("API_KEY", "dev-api-key-12345"),
            cors_origins=["http://localhost:3000", "http://localhost:8080"],
            https_only=False
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5001",
            artifact_root="./mlruns",
            backend_store_uri="sqlite:///mlflow.db",
            default_artifact_root="./mlruns"
        )
    )


def get_production_settings() -> Settings:
    """Get production environment settings"""
    return Settings(
        environment=Environment.PRODUCTION,
        api=APIConfig(
            api_key=os.getenv("API_KEY", ""),
            cors_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else [],
            https_only=True
        ),
        mlflow=MLflowConfig(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", ""),
            backend_store_uri=os.getenv("MLFLOW_BACKEND_STORE_URI", ""),
            default_artifact_root=os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", ""),
            s3_bucket=os.getenv("MLFLOW_S3_BUCKET"),
            aws_region=os.getenv("AWS_REGION", "us-east-1")
        ),
        security=SecurityConfig(
            secrets_manager_enabled=os.getenv("SECRETS_MANAGER_ENABLED", "false").lower() == "true",
            parameter_store_enabled=os.getenv("PARAMETER_STORE_ENABLED", "false").lower() == "true",
            iam_role_arn=os.getenv("IAM_ROLE_ARN")
        ),
        monitoring=MonitoringConfig(
            cloudwatch_enabled=os.getenv("CLOUDWATCH_ENABLED", "false").lower() == "true",
            sns_topic_arn=os.getenv("SNS_TOPIC_ARN"),
            pagerduty_integration_key=os.getenv("PAGERDUTY_INTEGRATION_KEY"),
            log_retention_days=int(os.getenv("LOG_RETENTION_DAYS", "30"))
        )
    )


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return get_production_settings()
    else:
        return get_development_settings()


# Global settings instance
settings = get_settings()