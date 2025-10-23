"""
Model registry management for MLflow with model promotion workflows.
Handles model registration, versioning, and stage transitions.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from .config import TrainingConfig

logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """
    Manages model registration and promotion in MLflow Model Registry.
    Provides functionality for model versioning, stage management, and metadata handling.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize model registry manager.
        
        Args:
            config: Training configuration containing MLflow settings
        """
        self.config = config
        self.client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
        self.model_name = config.mlflow.model_name
        
    def register_model(self, 
                      run_id: str,
                      model_path: str = "model",
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """
        Register a model from an MLflow run to the model registry.
        
        Args:
            run_id: MLflow run ID containing the model
            model_path: Path to the model artifact within the run
            description: Optional description for the model version
            tags: Optional tags for the model version
            
        Returns:
            ModelVersion object representing the registered model
        """
        try:
            # Ensure model registry exists
            self._ensure_registered_model_exists()
            
            # Create model URI
            model_uri = f"runs:/{run_id}/{model_path}"
            
            logger.info(f"Registering model from run {run_id} to registry as {self.model_name}")
            
            # Register model version
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=model_version.version,
                    description=description
                )
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            # Add default tags
            default_tags = {
                "registered_at": datetime.now().isoformat(),
                "run_id": run_id,
                "model_path": model_path
            }
            
            for key, value in default_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
            
            logger.info(f"Model registered successfully as version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def promote_model(self, 
                     version: str,
                     stage: str,
                     archive_existing: bool = True,
                     description: Optional[str] = None) -> ModelVersion:
        """
        Promote a model version to a specific stage.
        
        Args:
            version: Model version to promote
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Whether to archive existing models in the target stage
            description: Optional description for the transition
            
        Returns:
            Updated ModelVersion object
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")
        
        try:
            logger.info(f"Promoting model {self.model_name} version {version} to {stage}")
            
            # Archive existing models in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                self._archive_existing_models_in_stage(stage)
            
            # Transition model to new stage
            model_version = self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            
            # Add promotion description as a tag
            if description:
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key="promotion_description",
                    value=description
                )
            
            # Add promotion metadata
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key=f"promoted_to_{stage.lower()}_at",
                value=datetime.now().isoformat()
            )
            
            logger.info(f"Model version {version} successfully promoted to {stage}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def get_model_version(self, version: str) -> ModelVersion:
        """
        Get details of a specific model version.
        
        Args:
            version: Model version to retrieve
            
        Returns:
            ModelVersion object
        """
        try:
            return self.client.get_model_version(name=self.model_name, version=version)
        except Exception as e:
            logger.error(f"Failed to get model version {version}: {e}")
            raise
    
    def get_latest_model_version(self, stage: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get the latest model version, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter ("Staging", "Production", etc.)
            
        Returns:
            Latest ModelVersion object or None if no models found
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(name=self.model_name, stages=[stage])
                return versions[0] if versions else None
            else:
                # Get all versions and return the latest
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
                if not versions:
                    return None
                
                # Sort by version number (assuming numeric versions)
                try:
                    versions.sort(key=lambda v: int(v.version), reverse=True)
                except ValueError:
                    # If versions are not numeric, sort by creation time
                    versions.sort(key=lambda v: v.creation_timestamp, reverse=True)
                
                return versions[0]
                
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None
    
    def list_model_versions(self, stage: Optional[str] = None) -> List[ModelVersion]:
        """
        List all model versions, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of ModelVersion objects
        """
        try:
            if stage:
                return self.client.get_latest_versions(name=self.model_name, stages=[stage])
            else:
                return self.client.search_model_versions(f"name='{self.model_name}'")
                
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def delete_model_version(self, version: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            version: Model version to delete
        """
        try:
            logger.info(f"Deleting model version {version}")
            self.client.delete_model_version(name=self.model_name, version=version)
            logger.info(f"Model version {version} deleted successfully")
            
        except Exception as e:
            logger.error(f"Failed to delete model version {version}: {e}")
            raise
    
    def add_model_version_metadata(self, 
                                  version: str,
                                  metrics: Optional[Dict[str, float]] = None,
                                  tags: Optional[Dict[str, str]] = None,
                                  description: Optional[str] = None) -> None:
        """
        Add metadata to a model version.
        
        Args:
            version: Model version to update
            metrics: Optional metrics to add as tags
            tags: Optional tags to add
            description: Optional description to update
        """
        try:
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=version,
                    description=description
                )
            
            # Add metrics as tags
            if metrics:
                for metric_name, metric_value in metrics.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=version,
                        key=f"metric_{metric_name}",
                        value=str(metric_value)
                    )
            
            # Add custom tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=version,
                        key=key,
                        value=value
                    )
            
            logger.info(f"Metadata added to model version {version}")
            
        except Exception as e:
            logger.error(f"Failed to add metadata to model version {version}: {e}")
            raise
    
    def get_model_version_metadata(self, version: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a model version.
        
        Args:
            version: Model version to query
            
        Returns:
            Dictionary containing model version metadata
        """
        try:
            model_version = self.get_model_version(version)
            
            # Extract metrics from tags
            metrics = {}
            other_tags = {}
            
            for key, value in model_version.tags.items():
                if key.startswith("metric_"):
                    metric_name = key.replace("metric_", "")
                    try:
                        metrics[metric_name] = float(value)
                    except ValueError:
                        other_tags[key] = value
                else:
                    other_tags[key] = value
            
            metadata = {
                "version": model_version.version,
                "stage": model_version.current_stage,
                "description": model_version.description,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "run_id": model_version.run_id,
                "source": model_version.source,
                "status": model_version.status,
                "metrics": metrics,
                "tags": other_tags
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata for model version {version}: {e}")
            raise
    
    def _ensure_registered_model_exists(self) -> None:
        """Ensure the registered model exists in the registry"""
        try:
            self.client.get_registered_model(self.model_name)
            logger.debug(f"Registered model {self.model_name} already exists")
        except MlflowException:
            # Model doesn't exist, create it
            logger.info(f"Creating registered model {self.model_name}")
            self.client.create_registered_model(
                name=self.model_name,
                description=f"XGBoost model for {self.model_name}"
            )
    
    def _archive_existing_models_in_stage(self, stage: str) -> None:
        """Archive existing models in the specified stage"""
        try:
            existing_versions = self.client.get_latest_versions(
                name=self.model_name, 
                stages=[stage]
            )
            
            for version in existing_versions:
                logger.info(f"Archiving existing model version {version.version} from {stage}")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=version.version,
                    stage="Archived"
                )
                
        except Exception as e:
            logger.warning(f"Failed to archive existing models in {stage}: {e}")


class ModelPromotionWorkflow:
    """
    Automated model promotion workflow with validation and approval processes.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize promotion workflow"""
        self.config = config
        self.registry_manager = ModelRegistryManager(config)
        
    def register_and_promote_model(self,
                                  run_id: str,
                                  metrics: Dict[str, float],
                                  model_path: str = "model",
                                  auto_promote_to_staging: bool = None,
                                  auto_promote_to_production: bool = None) -> ModelVersion:
        """
        Register model and optionally promote based on configuration and metrics.
        
        Args:
            run_id: MLflow run ID
            metrics: Model performance metrics
            model_path: Path to model artifact
            auto_promote_to_staging: Override config for staging promotion
            auto_promote_to_production: Override config for production promotion
            
        Returns:
            Registered ModelVersion
        """
        # Use config defaults if not specified
        if auto_promote_to_staging is None:
            auto_promote_to_staging = self.config.mlflow.promote_to_staging
        if auto_promote_to_production is None:
            auto_promote_to_production = self.config.mlflow.promote_to_production
        
        # Register model
        model_version = self.registry_manager.register_model(
            run_id=run_id,
            model_path=model_path,
            description=f"Model trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Add metrics metadata
        self.registry_manager.add_model_version_metadata(
            version=model_version.version,
            metrics=metrics,
            tags={
                "training_dataset": self.config.data.dataset_name,
                "algorithm": "xgboost",
                "framework": "xgboost",
                "auto_registered": "true"
            }
        )
        
        # Auto-promote to staging if configured
        if auto_promote_to_staging:
            logger.info("Auto-promoting model to Staging")
            model_version = self.registry_manager.promote_model(
                version=model_version.version,
                stage="Staging",
                description="Auto-promoted to Staging after training"
            )
        
        # Auto-promote to production if configured (and meets criteria)
        if auto_promote_to_production:
            if self._should_promote_to_production(metrics):
                logger.info("Auto-promoting model to Production")
                model_version = self.registry_manager.promote_model(
                    version=model_version.version,
                    stage="Production",
                    description="Auto-promoted to Production based on performance criteria"
                )
            else:
                logger.info("Model does not meet criteria for auto-promotion to Production")
        
        return model_version
    
    def _should_promote_to_production(self, metrics: Dict[str, float]) -> bool:
        """
        Determine if model should be auto-promoted to production based on metrics.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            True if model should be promoted to production
        """
        # Define promotion criteria (can be made configurable)
        criteria = {
            "test_accuracy": 0.85,  # Minimum accuracy for classification
            "test_f1": 0.80,        # Minimum F1 score for classification
            "test_r2": 0.80,        # Minimum R² for regression
            "cv_mean": 0.80         # Minimum cross-validation score
        }
        
        # Check if any criterion is met
        for metric_name, threshold in criteria.items():
            if metric_name in metrics and metrics[metric_name] >= threshold:
                logger.info(f"Model meets production criteria: {metric_name} = {metrics[metric_name]:.4f} >= {threshold}")
                return True
        
        logger.info("Model does not meet any production promotion criteria")
        return False


def register_model_from_run(config: TrainingConfig,
                           run_id: str,
                           metrics: Dict[str, float],
                           model_path: str = "model") -> ModelVersion:
    """
    Convenience function to register and promote a model from an MLflow run.
    
    Args:
        config: Training configuration
        run_id: MLflow run ID
        metrics: Model performance metrics
        model_path: Path to model artifact
        
    Returns:
        Registered ModelVersion
    """
    workflow = ModelPromotionWorkflow(config)
    return workflow.register_and_promote_model(run_id, metrics, model_path)