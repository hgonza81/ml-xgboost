"""
MLflow experiment management and logging functionality.
Handles experiment creation, tracking, and artifact storage.
"""

import logging
import mlflow
from typing import Optional, Dict, Any, List
from datetime import datetime
from mlflow.entities import Experiment
from .tracking_server import get_tracking_server

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages MLflow experiments and logging operations.
    Provides functionality for experiment creation, tracking, and artifact management.
    """
    
    def __init__(self):
        """Initialize experiment manager."""
        self.tracking_server = get_tracking_server()
    
    def create_experiment(self, name: str, artifact_location: Optional[str] = None, 
                         tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new MLflow experiment.
        
        Args:
            name: Experiment name
            artifact_location: Optional custom artifact location
            tags: Optional experiment tags
            
        Returns:
            Experiment ID
        """
        try:
            # Check if experiment already exists
            existing_experiment = mlflow.get_experiment_by_name(name)
            if existing_experiment:
                logger.info(f"Experiment '{name}' already exists with ID: {existing_experiment.experiment_id}")
                return existing_experiment.experiment_id
            
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=name,
                artifact_location=artifact_location,
                tags=tags or {}
            )
            
            logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment '{name}': {e}")
            raise
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        try:
            return mlflow.get_experiment_by_name(name)
        except Exception as e:
            logger.error(f"Failed to get experiment '{name}': {e}")
            return None
    
    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        try:
            return mlflow.search_experiments()
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name
            tags: Optional run tags
            
        Returns:
            Active MLflow run
        """
        try:
            # Get or create experiment
            experiment = self.get_experiment(experiment_name)
            if not experiment:
                experiment_id = self.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            # Start run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags or {}
            )
            
            logger.info(f"Started run '{run_name or run.info.run_id}' in experiment '{experiment_name}'")
            return run
            
        except Exception as e:
            logger.error(f"Failed to start run in experiment '{experiment_name}': {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to current run."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to current run."""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact '{local_path}': {e}")
            raise
    
    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """Log model to current run."""
        try:
            # Determine model type and use appropriate logging function
            if hasattr(model, 'get_booster'):  # XGBoost
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, artifact_path, **kwargs)
            elif hasattr(model, 'predict'):  # Scikit-learn
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            else:
                # Generic Python model
                import mlflow.pyfunc
                mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)
            
            logger.info(f"Logged model to artifact path: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End current MLflow run."""
        try:
            mlflow.end_run(status=status)
            logger.debug(f"Ended run with status: {status}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
            raise
    
    def get_experiment_stats(self, experiment_name: str) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        try:
            experiment = self.get_experiment(experiment_name)
            if not experiment:
                return {'error': f'Experiment {experiment_name} not found'}
            
            # Get runs for the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            return {
                'experiment_id': experiment.experiment_id,
                'experiment_name': experiment.name,
                'total_runs': len(runs),
                'artifact_location': experiment.artifact_location,
                'lifecycle_stage': experiment.lifecycle_stage,
                'creation_time': experiment.creation_time,
                'last_update_time': experiment.last_update_time,
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment stats for '{experiment_name}': {e}")
            return {'error': str(e)}


# Global experiment manager instance
_experiment_manager: Optional[ExperimentManager] = None


def get_experiment_manager() -> ExperimentManager:
    """Get or create global experiment manager instance."""
    global _experiment_manager
    
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    
    return _experiment_manager