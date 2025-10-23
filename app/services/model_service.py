"""
Model service for loading and managing ML models from MLflow.
"""

import logging
from typing import Tuple, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information from MLflow registry."""
    name: str
    version: str
    stage: str
    run_id: str


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelService:
    """
    Service class for model loading and management.
    """
    
    def __init__(self, model_loader=None):
        """
        Initialize model service.
        
        Args:
            model_loader: MLflow model loader instance (injected dependency)
        """
        self.model_loader = model_loader
        self._model_cache = {}
    
    async def load_model(self, model_name: str, model_stage: str) -> Tuple[Any, ModelInfo]:
        """
        Load model from MLflow registry.
        
        Args:
            model_name: Name of the model to load
            model_stage: Stage of the model (production, staging, etc.)
            
        Returns:
            Tuple of (model, model_info)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        cache_key = f"{model_name}:{model_stage}"
        
        # Check cache first
        if cache_key in self._model_cache:
            logger.debug(f"Using cached model for {cache_key}")
            return self._model_cache[cache_key]
        
        try:
            if not self.model_loader:
                raise ModelLoadError("Model loader not initialized")
            
            model, metadata = self.model_loader.load_model(model_name, model_stage)
            
            model_info = ModelInfo(
                name=metadata.name,
                version=metadata.version,
                stage=metadata.stage,
                run_id=metadata.run_id
            )
            
            # Cache the loaded model
            self._model_cache[cache_key] = (model, model_info)
            
            logger.info(f"Successfully loaded model {model_name} v{model_info.version} from stage {model_stage}")
            return model, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from stage {model_stage}: {e}")
            raise ModelLoadError(f"Model loading failed: {str(e)}")
    
    async def get_model_info(self, model_name: str, model_stage: str) -> Dict[str, Any]:
        """
        Get model information without loading the model.
        
        Args:
            model_name: Name of the model
            model_stage: Stage of the model
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ModelLoadError: If model info retrieval fails
        """
        try:
            if not self.model_loader:
                raise ModelLoadError("Model loader not initialized")
            
            model_info = self.model_loader.get_model_info(model_name, model_stage)
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            raise ModelLoadError(f"Model info retrieval failed: {str(e)}")
    
    def health_check(self) -> Dict[str, str]:
        """
        Perform health check on model service.
        
        Returns:
            Dictionary with health check results
        """
        checks = {}
        
        try:
            if self.model_loader:
                mlflow_health = self.model_loader.health_check()
                checks.update(mlflow_health)
            else:
                checks["model_loader"] = "not_initialized"
        except Exception as e:
            checks["model_service"] = "unhealthy"
            checks["model_service_error"] = str(e)
        
        return checks