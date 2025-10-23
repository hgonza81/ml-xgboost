"""
MLflow tracking server configuration and management.
Provides tracking server setup with SQLite backend for development
and artifact storage configuration.
"""

import os
import logging
import mlflow
from typing import Optional, Dict, Any
from pathlib import Path
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class MLflowTrackingServer:
    """
    MLflow tracking server configuration and management.
    Handles server setup, artifact storage, and experiment tracking.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize MLflow tracking server with configuration."""
        self.settings = settings or get_settings()
        self.tracking_uri = self.settings.mlflow.tracking_uri
        self.artifact_root = self.settings.mlflow.artifact_root
        self.backend_store_uri = self.settings.mlflow.backend_store_uri
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize MLflow tracking server configuration."""
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create artifact directories if they don't exist
            self._ensure_artifact_directories()
            
            # Set environment variables for MLflow
            self._set_environment_variables()
            
            # Verify connection
            self._verify_connection()
            
            self._initialized = True
            logger.info(f"MLflow tracking server initialized with URI: {self.tracking_uri}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow tracking server: {e}")
            raise
    
    def _ensure_artifact_directories(self) -> None:
        """Ensure artifact storage directories exist."""
        if self.artifact_root.startswith('./') or self.artifact_root.startswith('/'):
            # Local file system path
            artifact_path = Path(self.artifact_root)
            artifact_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created artifact directory: {artifact_path}")
    
    def _set_environment_variables(self) -> None:
        """Set MLflow environment variables."""
        env_vars = {
            'MLFLOW_TRACKING_URI': self.tracking_uri,
            'MLFLOW_BACKEND_STORE_URI': self.backend_store_uri,
            'MLFLOW_DEFAULT_ARTIFACT_ROOT': self.settings.mlflow.default_artifact_root,
        }
        
        # Add S3 configuration for production
        if self.settings.mlflow.s3_bucket:
            env_vars.update({
                'MLFLOW_S3_BUCKET': self.settings.mlflow.s3_bucket,
                'AWS_DEFAULT_REGION': self.settings.mlflow.aws_region,
            })
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"Set environment variable: {key}")
    
    def _verify_connection(self) -> None:
        """Verify connection to MLflow tracking server."""
        try:
            # Try to get or create a test experiment
            experiment_name = "connection_test"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created test experiment: {experiment_id}")
            else:
                logger.info(f"Connected to existing experiment: {experiment.experiment_id}")
                
        except Exception as e:
            logger.error(f"Failed to verify MLflow connection: {e}")
            raise
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get MLflow server configuration information."""
        return {
            'tracking_uri': self.tracking_uri,
            'artifact_root': self.artifact_root,
            'backend_store_uri': self.backend_store_uri,
            'initialized': self._initialized,
            'environment': self.settings.environment.value,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on MLflow tracking server."""
        try:
            # Try to list experiments to verify server is responsive
            experiments = mlflow.search_experiments()
            
            return {
                'status': 'healthy',
                'tracking_uri': self.tracking_uri,
                'experiments_count': len(experiments),
                'initialized': self._initialized,
            }
            
        except Exception as e:
            logger.error(f"MLflow health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'tracking_uri': self.tracking_uri,
                'initialized': self._initialized,
            }


# Global tracking server instance
_tracking_server: Optional[MLflowTrackingServer] = None


def get_tracking_server() -> MLflowTrackingServer:
    """Get or create global MLflow tracking server instance."""
    global _tracking_server
    
    if _tracking_server is None:
        _tracking_server = MLflowTrackingServer()
        _tracking_server.initialize()
    
    return _tracking_server


def initialize_tracking_server(settings: Optional[Settings] = None) -> MLflowTrackingServer:
    """Initialize MLflow tracking server with optional custom settings."""
    global _tracking_server
    
    _tracking_server = MLflowTrackingServer(settings)
    _tracking_server.initialize()
    
    return _tracking_server