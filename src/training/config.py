"""
Training configuration and settings for the ML pipeline.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import os


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration"""
    dataset_name: str = Field(default="wine", description="Name of the dataset to use")
    custom_dataset_path: Optional[str] = Field(default=None, description="Path to custom dataset")
    target_column: Optional[str] = Field(default=None, description="Target column name for custom datasets")
    test_size: float = Field(default=0.2, description="Test set proportion")
    validation_size: float = Field(default=0.2, description="Validation set proportion")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    stratify: bool = Field(default=True, description="Whether to stratify train/test split")
    
    # Preprocessing options
    numeric_strategy: str = Field(default="median", description="Missing value strategy for numeric features")
    categorical_strategy: str = Field(default="most_frequent", description="Missing value strategy for categorical features")
    scaling_method: str = Field(default="standard", description="Feature scaling method")
    encoding_method: str = Field(default="onehot", description="Categorical encoding method")


class XGBoostConfig(BaseModel):
    """XGBoost model configuration"""
    objective: str = Field(default="multi:softprob", description="XGBoost objective function")
    eval_metric: str = Field(default="mlogloss", description="Evaluation metric")
    n_estimators: int = Field(default=100, description="Number of boosting rounds")
    max_depth: int = Field(default=6, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, description="Learning rate")
    subsample: float = Field(default=1.0, description="Subsample ratio")
    colsample_bytree: float = Field(default=1.0, description="Column subsample ratio")
    reg_alpha: float = Field(default=0.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, description="L2 regularization")
    random_state: int = Field(default=42, description="Random seed")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    verbosity: int = Field(default=1, description="Verbosity level")
    
    # Early stopping
    early_stopping_rounds: Optional[int] = Field(default=10, description="Early stopping rounds")
    
    def to_xgboost_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dictionary"""
        params = self.dict()
        # Remove non-XGBoost parameters
        params.pop('early_stopping_rounds', None)
        return params


class HyperparameterConfig(BaseModel):
    """Hyperparameter optimization configuration"""
    enabled: bool = Field(default=False, description="Enable hyperparameter optimization")
    n_trials: int = Field(default=50, description="Number of optimization trials")
    timeout: Optional[int] = Field(default=3600, description="Optimization timeout in seconds")
    
    # Parameter search spaces
    n_estimators_range: List[int] = Field(default=[50, 200], description="N estimators search range")
    max_depth_range: List[int] = Field(default=[3, 10], description="Max depth search range")
    learning_rate_range: List[float] = Field(default=[0.01, 0.3], description="Learning rate search range")
    subsample_range: List[float] = Field(default=[0.6, 1.0], description="Subsample search range")
    colsample_bytree_range: List[float] = Field(default=[0.6, 1.0], description="Column subsample search range")
    reg_alpha_range: List[float] = Field(default=[0.0, 1.0], description="L1 regularization search range")
    reg_lambda_range: List[float] = Field(default=[0.0, 2.0], description="L2 regularization search range")


class MLflowConfig(BaseModel):
    """MLflow tracking configuration"""
    experiment_name: str = Field(default="xgboost_classification", description="MLflow experiment name")
    run_name: Optional[str] = Field(default=None, description="MLflow run name")
    tracking_uri: str = Field(default="http://localhost:5001", description="MLflow tracking URI")
    artifact_location: Optional[str] = Field(default=None, description="Artifact storage location")
    
    # Model registration
    model_name: str = Field(default="xgboost_classifier", description="Model name in registry")
    register_model: bool = Field(default=True, description="Whether to register model")
    promote_to_staging: bool = Field(default=False, description="Promote to staging after training")
    promote_to_production: bool = Field(default=False, description="Promote to production after training")


class TrainingConfig(BaseModel):
    """Main training configuration"""
    data: DataConfig = Field(default_factory=DataConfig)
    model: XGBoostConfig = Field(default_factory=XGBoostConfig)
    hyperparameter_optimization: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    # Training settings
    cross_validation_folds: int = Field(default=5, description="Number of CV folds")
    save_model_artifacts: bool = Field(default=True, description="Save model artifacts")
    save_predictions: bool = Field(default=True, description="Save test predictions")
    
    class Config:
        env_nested_delimiter = '__'


def get_training_config() -> TrainingConfig:
    """Get training configuration with environment overrides"""
    config = TrainingConfig()
    
    # Override with environment variables if present
    if os.getenv("DATASET_NAME"):
        config.data.dataset_name = os.getenv("DATASET_NAME")
    
    if os.getenv("MLFLOW_TRACKING_URI"):
        config.mlflow.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if os.getenv("EXPERIMENT_NAME"):
        config.mlflow.experiment_name = os.getenv("EXPERIMENT_NAME")
    
    if os.getenv("MODEL_NAME"):
        config.mlflow.model_name = os.getenv("MODEL_NAME")
    
    # Hyperparameter optimization settings
    if os.getenv("ENABLE_HPO", "").lower() == "true":
        config.hyperparameter_optimization.enabled = True
    
    if os.getenv("HPO_N_TRIALS"):
        config.hyperparameter_optimization.n_trials = int(os.getenv("HPO_N_TRIALS"))
    
    return config


# Default configuration instance
training_config = get_training_config()