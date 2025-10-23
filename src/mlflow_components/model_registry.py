"""
MLflow model registry operations and management.
Handles model registration, versioning, stage management, and metadata storage.
"""

import logging
import mlflow
from typing import Optional, Dict, Any, List
from datetime import datetime
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.exceptions import MlflowException
from .tracking_server import get_tracking_server

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages MLflow model registry operations.
    Provides functionality for model registration, versioning, and stage management.
    """
    
    def __init__(self):
        """Initialize model registry."""
        self.tracking_server = get_tracking_server()
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model_uri: str, name: str, 
                      tags: Optional[Dict[str, str]] = None,
                      description: Optional[str] = None) -> ModelVersion:
        """
        Register a model in the MLflow model registry.
        
        Args:
            model_uri: URI of the model artifact
            name: Name for the registered model
            tags: Optional model tags
            description: Optional model description
            
        Returns:
            ModelVersion object
        """
        try:
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=name,
                tags=tags or {}
            )
            
            # Add description if provided
            if description:
                self.client.update_model_version(
                    name=name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model '{name}' version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model '{name}': {e}")
            raise
    
    def create_registered_model(self, name: str, tags: Optional[Dict[str, str]] = None,
                               description: Optional[str] = None) -> RegisteredModel:
        """
        Create a new registered model.
        
        Args:
            name: Model name
            tags: Optional model tags
            description: Optional model description
            
        Returns:
            RegisteredModel object
        """
        try:
            # Check if model already exists
            try:
                existing_model = self.client.get_registered_model(name)
                logger.info(f"Registered model '{name}' already exists")
                return existing_model
            except MlflowException:
                # Model doesn't exist, create it
                pass
            
            # Create the registered model
            registered_model = self.client.create_registered_model(
                name=name,
                tags=tags or {},
                description=description
            )
            
            logger.info(f"Created registered model '{name}'")
            return registered_model
            
        except Exception as e:
            logger.error(f"Failed to create registered model '{name}': {e}")
            raise
    
    def get_registered_model(self, name: str) -> Optional[RegisteredModel]:
        """Get registered model by name."""
        try:
            return self.client.get_registered_model(name)
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                return None
            logger.error(f"Failed to get registered model '{name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get registered model '{name}': {e}")
            return None
    
    def list_registered_models(self) -> List[RegisteredModel]:
        """List all registered models."""
        try:
            return self.client.search_registered_models()
        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []
    
    def get_model_version(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        try:
            return self.client.get_model_version(name, version)
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                return None
            logger.error(f"Failed to get model version '{name}' v{version}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get model version '{name}' v{version}: {e}")
            return None
    
    def get_latest_model_version(self, name: str, stage: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get latest model version, optionally filtered by stage.
        
        Args:
            name: Model name
            stage: Optional stage filter ("staging", "production", "archived")
            
        Returns:
            Latest ModelVersion or None
        """
        try:
            versions = self.client.get_latest_versions(name, stages=[stage] if stage else None)
            return versions[0] if versions else None
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                return None
            logger.error(f"Failed to get latest model version for '{name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get latest model version for '{name}': {e}")
            return None
    
    def transition_model_version_stage(self, name: str, version: str, stage: str,
                                     archive_existing_versions: bool = False) -> ModelVersion:
        """
        Transition model version to a new stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage ("staging", "production", "archived")
            archive_existing_versions: Whether to archive existing versions in target stage
            
        Returns:
            Updated ModelVersion
        """
        try:
            model_version = self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Transitioned model '{name}' v{version} to stage '{stage}'")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to transition model '{name}' v{version} to stage '{stage}': {e}")
            raise
    
    def update_model_version(self, name: str, version: str, 
                           description: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """
        Update model version metadata.
        
        Args:
            name: Model name
            version: Model version
            description: Optional new description
            tags: Optional tags to add/update
            
        Returns:
            Updated ModelVersion
        """
        try:
            # Update description if provided
            if description is not None:
                self.client.update_model_version(
                    name=name,
                    version=version,
                    description=description
                )
            
            # Update tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(name, version, key, value)
            
            # Get updated model version
            model_version = self.client.get_model_version(name, version)
            logger.info(f"Updated model '{name}' v{version} metadata")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to update model '{name}' v{version}: {e}")
            raise
    
    def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version."""
        try:
            self.client.delete_model_version(name, version)
            logger.info(f"Deleted model '{name}' v{version}")
        except Exception as e:
            logger.error(f"Failed to delete model '{name}' v{version}: {e}")
            raise
    
    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model and all its versions."""
        try:
            self.client.delete_registered_model(name)
            logger.info(f"Deleted registered model '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete registered model '{name}': {e}")
            raise
    
    def get_model_metadata(self, name: str, version: Optional[str] = None,
                          stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive model metadata.
        
        Args:
            name: Model name
            version: Specific version (optional)
            stage: Stage filter for latest version (optional)
            
        Returns:
            Model metadata dictionary
        """
        try:
            # Get registered model
            registered_model = self.get_registered_model(name)
            if not registered_model:
                return {'error': f'Model {name} not found'}
            
            # Get model version
            if version:
                model_version = self.get_model_version(name, version)
            else:
                model_version = self.get_latest_model_version(name, stage)
            
            if not model_version:
                return {
                    'error': f'No model version found for {name}' + 
                            (f' with stage {stage}' if stage else '')
                }
            
            return {
                'name': registered_model.name,
                'description': registered_model.description,
                'creation_timestamp': registered_model.creation_timestamp,
                'last_updated_timestamp': registered_model.last_updated_timestamp,
                'tags': registered_model.tags,
                'version': {
                    'version': model_version.version,
                    'creation_timestamp': model_version.creation_timestamp,
                    'last_updated_timestamp': model_version.last_updated_timestamp,
                    'description': model_version.description,
                    'user_id': model_version.user_id,
                    'current_stage': model_version.current_stage,
                    'source': model_version.source,
                    'run_id': model_version.run_id,
                    'status': model_version.status,
                    'status_message': model_version.status_message,
                    'tags': model_version.tags,
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model metadata for '{name}': {e}")
            return {'error': str(e)}
    
    def search_model_versions(self, filter_string: Optional[str] = None,
                            max_results: int = 100) -> List[ModelVersion]:
        """
        Search model versions with optional filter.
        
        Args:
            filter_string: Optional filter string
            max_results: Maximum number of results
            
        Returns:
            List of ModelVersion objects
        """
        try:
            return self.client.search_model_versions(
                filter_string=filter_string,
                max_results=max_results
            )
        except Exception as e:
            logger.error(f"Failed to search model versions: {e}")
            return []
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get model registry statistics."""
        try:
            registered_models = self.list_registered_models()
            
            total_models = len(registered_models)
            total_versions = 0
            stage_counts = {'staging': 0, 'production': 0, 'archived': 0, 'none': 0}
            
            for model in registered_models:
                try:
                    versions = self.client.search_model_versions(f"name='{model.name}'")
                    total_versions += len(versions)
                    
                    for version in versions:
                        stage = version.current_stage.lower()
                        if stage in stage_counts:
                            stage_counts[stage] += 1
                        else:
                            stage_counts['none'] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to get versions for model '{model.name}': {e}")
            
            return {
                'total_registered_models': total_models,
                'total_model_versions': total_versions,
                'versions_by_stage': stage_counts,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {'error': str(e)}


# Global model registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry instance."""
    global _model_registry
    
    if _model_registry is None:
        _model_registry = ModelRegistry()
    
    return _model_registry