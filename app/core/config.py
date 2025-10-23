"""
Core configuration management for the ML API Platform.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    model_name: str = Field(default="wine_classifier", env="MODEL_NAME")
    model_stage: str = Field(default="production", env="MODEL_STAGE")
    api_key: str = Field(default="dev-api-key-12345", env="API_KEY")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8080", env="CORS_ORIGINS")
    https_only: bool = Field(default=False, env="HTTPS_ONLY")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        protected_namespaces = ('settings_',)


class MLflowConfig(BaseSettings):
    """MLflow configuration settings."""
    
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    artifact_root: str = Field(default="./mlruns", env="MLFLOW_ARTIFACT_ROOT")
    backend_store_uri: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_BACKEND_STORE_URI")
    default_artifact_root: str = Field(default="./mlruns", env="MLFLOW_DEFAULT_ARTIFACT_ROOT")
    s3_bucket: Optional[str] = Field(default=None, env="MLFLOW_S3_BUCKET")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    
    secrets_manager_enabled: bool = Field(default=False, env="SECRETS_MANAGER_ENABLED")
    parameter_store_enabled: bool = Field(default=False, env="PARAMETER_STORE_ENABLED")
    iam_role_arn: Optional[str] = Field(default=None, env="IAM_ROLE_ARN")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """Monitoring configuration settings."""
    
    cloudwatch_enabled: bool = Field(default=False, env="CLOUDWATCH_ENABLED")
    sns_topic_arn: Optional[str] = Field(default=None, env="SNS_TOPIC_ARN")
    pagerduty_integration_key: Optional[str] = Field(default=None, env="PAGERDUTY_INTEGRATION_KEY")
    log_retention_days: int = Field(default=30, env="LOG_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings:
    """Main settings class that combines all configuration sections."""
    
    def __init__(self):
        self.api = APIConfig()
        self.mlflow = MLflowConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        
        # Configure logging level
        logging.basicConfig(level=getattr(logging, self.api.log_level.upper()))
        logger.info(f"Settings initialized with log level: {self.api.log_level}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern).
    
    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.info("Global settings instance created")
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings (useful for testing or configuration changes).
    
    Returns:
        Settings: New settings instance
    """
    global _settings
    _settings = Settings()
    logger.info("Settings reloaded")
    return _settings