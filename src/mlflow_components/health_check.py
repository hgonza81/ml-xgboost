"""
MLflow system health check functionality.
Provides comprehensive health monitoring for tracking server and model registry.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .tracking_server import get_tracking_server
from .model_registry import get_model_registry
from .experiment_manager import get_experiment_manager

logger = logging.getLogger(__name__)


class MLflowHealthCheck:
    """
    Comprehensive health check for MLflow system components.
    Monitors tracking server, model registry, and experiment management.
    """
    
    def __init__(self):
        """Initialize MLflow health check."""
        self.tracking_server = None
        self.model_registry = None
        self.experiment_manager = None
    
    def _initialize_components(self) -> None:
        """Lazy initialization of MLflow components."""
        try:
            if self.tracking_server is None:
                self.tracking_server = get_tracking_server()
            if self.model_registry is None:
                self.model_registry = get_model_registry()
            if self.experiment_manager is None:
                self.experiment_manager = get_experiment_manager()
        except Exception as e:
            logger.error(f"Failed to initialize MLflow components: {e}")
            raise
    
    def check_tracking_server(self) -> Dict[str, Any]:
        """Check tracking server health."""
        try:
            self._initialize_components()
            return self.tracking_server.health_check()
        except Exception as e:
            logger.error(f"Tracking server health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'component': 'tracking_server'
            }
    
    def check_model_registry(self) -> Dict[str, Any]:
        """Check model registry health."""
        try:
            self._initialize_components()
            
            # Try to list registered models
            models = self.model_registry.list_registered_models()
            
            return {
                'status': 'healthy',
                'registered_models_count': len(models),
                'component': 'model_registry'
            }
            
        except Exception as e:
            logger.error(f"Model registry health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'component': 'model_registry'
            }
    
    def check_experiment_manager(self) -> Dict[str, Any]:
        """Check experiment manager health."""
        try:
            self._initialize_components()
            
            # Try to list experiments
            experiments = self.experiment_manager.list_experiments()
            
            return {
                'status': 'healthy',
                'experiments_count': len(experiments),
                'component': 'experiment_manager'
            }
            
        except Exception as e:
            logger.error(f"Experiment manager health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'component': 'experiment_manager'
            }
    
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all MLflow components."""
        timestamp = datetime.utcnow().isoformat()
        
        # Check all components
        tracking_health = self.check_tracking_server()
        registry_health = self.check_model_registry()
        experiment_health = self.check_experiment_manager()
        
        # Determine overall status
        all_healthy = all(
            check['status'] == 'healthy' 
            for check in [tracking_health, registry_health, experiment_health]
        )
        
        overall_status = 'healthy' if all_healthy else 'unhealthy'
        
        # Collect any errors
        errors = []
        for check in [tracking_health, registry_health, experiment_health]:
            if check['status'] == 'unhealthy' and 'error' in check:
                errors.append(f"{check['component']}: {check['error']}")
        
        result = {
            'overall_status': overall_status,
            'timestamp': timestamp,
            'components': {
                'tracking_server': tracking_health,
                'model_registry': registry_health,
                'experiment_manager': experiment_health,
            }
        }
        
        if errors:
            result['errors'] = errors
        
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive MLflow system information."""
        try:
            self._initialize_components()
            
            # Get tracking server info
            server_info = self.tracking_server.get_server_info()
            
            # Get registry stats
            registry_stats = self.model_registry.get_registry_stats()
            
            # Get experiment stats for a few recent experiments
            experiments = self.experiment_manager.list_experiments()
            experiment_stats = []
            
            for exp in experiments[:5]:  # Get stats for first 5 experiments
                stats = self.experiment_manager.get_experiment_stats(exp.name)
                if 'error' not in stats:
                    experiment_stats.append(stats)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'tracking_server': server_info,
                'model_registry': registry_stats,
                'recent_experiments': experiment_stats,
                'total_experiments': len(experiments),
            }
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }


# Global health check instance
_health_check: Optional[MLflowHealthCheck] = None


def get_mlflow_health_check() -> MLflowHealthCheck:
    """Get or create global MLflow health check instance."""
    global _health_check
    
    if _health_check is None:
        _health_check = MLflowHealthCheck()
    
    return _health_check