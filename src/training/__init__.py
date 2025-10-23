# Model training pipeline components

from .data_loader import DataLoader, DataPreprocessor, create_train_test_split
from .data_validation import DataValidator, ValidationResult, ValidationSeverity
from .config import TrainingConfig, get_training_config
from .trainer import ModelTrainer, train_model
from .model_registry_manager import ModelRegistryManager, ModelPromotionWorkflow, register_model_from_run

# Hyperparameter optimization (optional import)
try:
    from .hyperparameter_optimizer import OPTUNA_AVAILABLE
    if OPTUNA_AVAILABLE:
        from .hyperparameter_optimizer import HyperparameterOptimizer, optimize_hyperparameters, OptimizedModelTrainer
        HPO_AVAILABLE = True
    else:
        HyperparameterOptimizer = None
        optimize_hyperparameters = None
        OptimizedModelTrainer = None
        HPO_AVAILABLE = False
except ImportError:
    HyperparameterOptimizer = None
    optimize_hyperparameters = None
    OptimizedModelTrainer = None
    HPO_AVAILABLE = False

__all__ = [
    'DataLoader',
    'DataPreprocessor', 
    'create_train_test_split',
    'DataValidator',
    'ValidationResult',
    'ValidationSeverity',
    'TrainingConfig',
    'get_training_config',
    'ModelTrainer',
    'train_model',
    'ModelRegistryManager',
    'ModelPromotionWorkflow',
    'register_model_from_run',
    'HyperparameterOptimizer',
    'optimize_hyperparameters',
    'OptimizedModelTrainer',
    'HPO_AVAILABLE'
]