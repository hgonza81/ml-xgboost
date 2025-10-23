"""
MLflow model registry integration for loading and managing ML models.

This module provides:
- Model loader that connects to MLflow registry
- Model version management and loading logic
- Error handling for model loading failures with fallback
- Model caching for performance optimization
"""

import logging
import os
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import threading
from functools import lru_cache

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException, RestException
import pandas as pd
import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading failures."""
    pass


class ModelMetadata:
    """Container for model metadata and information."""
    
    def __init__(self, name: str, version: str, stage: str, run_id: str, 
                 model_uri: str, loaded_at: datetime, metrics: Dict[str, Any] = None):
        self.name = name
        self.version = version
        self.stage = stage
        self.run_id = run_id
        self.model_uri = model_uri
        self.loaded_at = loaded_at
        self.metrics = metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "run_id": self.run_id,
            "model_uri": self.model_uri,
            "loaded_at": self.loaded_at.isoformat(),
            "metrics": self.metrics
        }


class MLflowModelLoader:
    """
    MLflow model registry integration with caching and error handling.
    
    Provides model loading from MLflow registry with automatic fallback
    to previous versions on failure and performance optimization through caching.
    """
    
    def __init__(self, tracking_uri: str = None, cache_ttl_minutes: int = 30):
        """
        Initialize the model loader.
        
        Args:
            tracking_uri: MLflow tracking server URI
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.settings = get_settings()
        self.tracking_uri = tracking_uri or self.settings.mlflow.tracking_uri
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Model cache with thread safety
        self._model_cache: Dict[str, Tuple[Any, ModelMetadata, datetime]] = {}
        self._cache_lock = threading.RLock()
        
        # Fallback model versions
        self._fallback_versions: Dict[str, str] = {}
        
        logger.info(f"Initialized MLflow model loader with tracking URI: {self.tracking_uri}")
    
    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if cached model is still valid based on TTL."""
        return datetime.utcnow() - cached_at < self.cache_ttl
    
    def _get_model_versions(self, model_name: str, stage: str = None) -> list:
        """
        Get available model versions from registry.
        
        Args:
            model_name: Name of the model in registry
            stage: Model stage filter (e.g., 'production', 'staging')
            
        Returns:
            List of model versions sorted by version number (descending)
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                # Get all versions and sort by version number
                all_versions = self.client.search_model_versions(f"name='{model_name}'")
                versions = sorted(all_versions, key=lambda x: int(x.version), reverse=True)
            
            logger.info(f"Found {len(versions)} versions for model '{model_name}' with stage '{stage}'")
            return versions
            
        except RestException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.error(f"Model '{model_name}' not found in registry")
                raise ModelLoadError(f"Model '{model_name}' not found in MLflow registry")
            else:
                logger.error(f"Failed to get model versions: {e}")
                raise ModelLoadError(f"Failed to access MLflow registry: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting model versions: {e}")
            raise ModelLoadError(f"Unexpected error accessing model registry: {e}")
    
    def _load_model_from_uri(self, model_uri: str, model_name: str, version: str) -> Tuple[Any, ModelMetadata]:
        """
        Load model from MLflow URI and create metadata.
        
        Args:
            model_uri: MLflow model URI
            model_name: Model name
            version: Model version
            
        Returns:
            Tuple of (loaded_model, model_metadata)
        """
        try:
            logger.info(f"Loading model from URI: {model_uri}")
            
            # Load the model using MLflow
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model version details for metadata
            model_version = self.client.get_model_version(model_name, version)
            
            # Get run details for additional metadata
            run_info = None
            metrics = {}
            if model_version.run_id:
                try:
                    run = self.client.get_run(model_version.run_id)
                    run_info = run.info
                    metrics = run.data.metrics
                except Exception as e:
                    logger.warning(f"Could not get run details for model: {e}")
            
            # Create metadata
            metadata = ModelMetadata(
                name=model_name,
                version=version,
                stage=model_version.current_stage,
                run_id=model_version.run_id or "unknown",
                model_uri=model_uri,
                loaded_at=datetime.utcnow(),
                metrics=metrics
            )
            
            logger.info(f"Successfully loaded model '{model_name}' version '{version}' from stage '{metadata.stage}'")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from URI '{model_uri}': {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def load_model(self, model_name: str, stage: str = "production", 
                   use_cache: bool = True, enable_fallback: bool = True) -> Tuple[Any, ModelMetadata]:
        """
        Load model from MLflow registry with caching and fallback support.
        
        Args:
            model_name: Name of the model in MLflow registry
            stage: Model stage to load ('production', 'staging', etc.)
            use_cache: Whether to use cached model if available
            enable_fallback: Whether to fallback to previous versions on failure
            
        Returns:
            Tuple of (loaded_model, model_metadata)
            
        Raises:
            ModelLoadError: If model loading fails and no fallback is available
        """
        cache_key = f"{model_name}:{stage}"
        
        # Check cache first if enabled
        if use_cache:
            with self._cache_lock:
                if cache_key in self._model_cache:
                    model, metadata, cached_at = self._model_cache[cache_key]
                    if self._is_cache_valid(cached_at):
                        logger.info(f"Using cached model '{model_name}' version '{metadata.version}'")
                        return model, metadata
                    else:
                        logger.info(f"Cache expired for model '{model_name}', reloading")
        
        # Get available model versions
        try:
            versions = self._get_model_versions(model_name, stage)
            if not versions:
                raise ModelLoadError(f"No models found for '{model_name}' in stage '{stage}'")
            
            # Try to load the latest version first
            latest_version = versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            
            try:
                model, metadata = self._load_model_from_uri(model_uri, model_name, latest_version.version)
                
                # Cache the successfully loaded model
                if use_cache:
                    with self._cache_lock:
                        self._model_cache[cache_key] = (model, metadata, datetime.utcnow())
                        # Update fallback version
                        self._fallback_versions[model_name] = latest_version.version
                
                return model, metadata
                
            except ModelLoadError as e:
                if not enable_fallback or len(versions) <= 1:
                    raise e
                
                logger.warning(f"Failed to load latest version, trying fallback versions: {e}")
                
                # Try fallback versions
                for fallback_version in versions[1:]:
                    try:
                        fallback_uri = f"models:/{model_name}/{fallback_version.version}"
                        logger.info(f"Attempting fallback to version {fallback_version.version}")
                        
                        model, metadata = self._load_model_from_uri(fallback_uri, model_name, fallback_version.version)
                        
                        # Cache the fallback model
                        if use_cache:
                            with self._cache_lock:
                                self._model_cache[cache_key] = (model, metadata, datetime.utcnow())
                        
                        logger.warning(f"Successfully loaded fallback model version {fallback_version.version}")
                        return model, metadata
                        
                    except ModelLoadError as fallback_error:
                        logger.warning(f"Fallback version {fallback_version.version} also failed: {fallback_error}")
                        continue
                
                # If all versions failed
                raise ModelLoadError(f"All available versions of model '{model_name}' failed to load")
                
        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model '{model_name}': {e}")
            raise ModelLoadError(f"Unexpected error loading model: {e}")
    
    def predict(self, model_name: str, input_data: pd.DataFrame, 
                stage: str = "production") -> np.ndarray:
        """
        Make predictions using a loaded model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data as pandas DataFrame
            stage: Model stage to use
            
        Returns:
            Numpy array of predictions
            
        Raises:
            ModelLoadError: If model loading or prediction fails
        """
        try:
            # Load the model
            model, metadata = self.load_model(model_name, stage)
            
            # Make predictions
            logger.info(f"Making predictions with model '{model_name}' version '{metadata.version}'")
            predictions = model.predict(input_data)
            
            logger.info(f"Successfully generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {e}")
            raise ModelLoadError(f"Prediction failed: {e}")
    
    def get_model_info(self, model_name: str, stage: str = "production") -> Dict[str, Any]:
        """
        Get model information without loading the model.
        
        Args:
            model_name: Name of the model
            stage: Model stage
            
        Returns:
            Dictionary containing model information
        """
        try:
            versions = self._get_model_versions(model_name, stage)
            if not versions:
                raise ModelLoadError(f"No models found for '{model_name}' in stage '{stage}'")
            
            latest_version = versions[0]
            
            # Get run details if available
            metrics = {}
            if latest_version.run_id:
                try:
                    run = self.client.get_run(latest_version.run_id)
                    metrics = run.data.metrics
                except Exception as e:
                    logger.warning(f"Could not get run metrics: {e}")
            
            return {
                "name": model_name,
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "created_at": latest_version.creation_timestamp,
                "last_updated": latest_version.last_updated_timestamp,
                "metrics": metrics,
                "description": latest_version.description
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for '{model_name}': {e}")
            raise ModelLoadError(f"Failed to get model info: {e}")
    
    def clear_cache(self, model_name: str = None):
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to clear from cache, or None to clear all
        """
        with self._cache_lock:
            if model_name:
                # Clear specific model from cache
                keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(f"{model_name}:")]
                for key in keys_to_remove:
                    del self._model_cache[key]
                logger.info(f"Cleared cache for model '{model_name}'")
            else:
                # Clear all cache
                self._model_cache.clear()
                logger.info("Cleared all model cache")
    
    def health_check(self) -> Dict[str, str]:
        """
        Perform health check on MLflow connection.
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Test connection to MLflow tracking server
            experiments = self.client.search_experiments(max_results=1)
            
            return {
                "mlflow_connection": "healthy",
                "tracking_uri": self.tracking_uri,
                "cache_size": str(len(self._model_cache))
            }
            
        except Exception as e:
            logger.error(f"MLflow health check failed: {e}")
            return {
                "mlflow_connection": "unhealthy",
                "tracking_uri": self.tracking_uri,
                "error": str(e)
            }


# Global model loader instance
_model_loader: Optional[MLflowModelLoader] = None
_loader_lock = threading.Lock()


def get_model_loader() -> MLflowModelLoader:
    """
    Get or create the global model loader instance.
    
    Returns:
        MLflowModelLoader instance
    """
    global _model_loader
    
    if _model_loader is None:
        with _loader_lock:
            if _model_loader is None:
                _model_loader = MLflowModelLoader()
    
    return _model_loader


@lru_cache(maxsize=1)
def get_cached_model_loader() -> MLflowModelLoader:
    """
    Get cached model loader instance (for dependency injection).
    
    Returns:
        MLflowModelLoader instance
    """
    return get_model_loader()