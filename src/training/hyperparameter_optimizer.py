"""
Hyperparameter optimization for XGBoost models using Optuna with MLflow integration.
Provides automated parameter tuning with experiment tracking.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional, Callable, Tuple
import logging
from datetime import datetime
import json

# Optuna imports
try:
    import optuna
    from optuna.integration.mlflow import MLflowCallback
    OPTUNA_AVAILABLE = True
    OptunaTrial = optuna.Trial
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    MLflowCallback = None
    # Create a dummy class for type annotations
    class OptunaTrial:
        pass

# MLflow imports
import mlflow
import mlflow.xgboost

# Sklearn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from .config import TrainingConfig, HyperparameterConfig
from .data_loader import DataPreprocessor
from .data_validation import DataValidator

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for XGBoost models using Optuna.
    Integrates with MLflow for experiment tracking and supports both classification and regression.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            config: Training configuration containing hyperparameter settings
            
        Raises:
            ImportError: If Optuna is not available
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install it with: pip install optuna"
            )
        
        self.config = config
        self.hpo_config = config.hyperparameter_optimization
        self.study = None
        self.best_params = None
        self.best_score = None
        self.task_type = None
        
    def optimize(self, 
                 X_train: np.ndarray, 
                 y_train: pd.Series,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[pd.Series] = None,
                 task_type: str = "classification") -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, uses CV if not provided)
            y_val: Validation target (optional, uses CV if not provided)
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            Dictionary containing optimization results
        """
        self.task_type = task_type
        
        logger.info(f"Starting hyperparameter optimization for {task_type}")
        logger.info(f"Number of trials: {self.hpo_config.n_trials}")
        logger.info(f"Timeout: {self.hpo_config.timeout} seconds")
        
        # Create study
        study_name = f"{self.config.mlflow.experiment_name}_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        direction = "maximize" if task_type == "classification" else "minimize"
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.config.data.random_state)
        )
        
        # Set up MLflow callback
        mlflow_callback = MLflowCallback(
            tracking_uri=self.config.mlflow.tracking_uri,
            metric_name="objective_value"
        )
        
        # Create objective function
        objective_func = self._create_objective_function(X_train, y_train, X_val, y_val)
        
        # Run optimization
        try:
            self.study.optimize(
                objective_func,
                n_trials=self.hpo_config.n_trials,
                timeout=self.hpo_config.timeout,
                callbacks=[mlflow_callback],
                show_progress_bar=True
            )
            
            # Store results
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            logger.info(f"Optimization completed!")
            logger.info(f"Best score: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            # Prepare results
            results = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "n_trials": len(self.study.trials),
                "study": self.study,
                "optimization_history": self._get_optimization_history(),
                "parameter_importance": self._get_parameter_importance()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise
    
    def _create_objective_function(self, 
                                   X_train: np.ndarray, 
                                   y_train: pd.Series,
                                   X_val: Optional[np.ndarray] = None,
                                   y_val: Optional[pd.Series] = None) -> Callable:
        """Create objective function for Optuna optimization"""
        
        def objective(trial: OptunaTrial) -> float:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)
            
            # Create model
            if self.task_type == "classification":
                model = xgb.XGBClassifier(**params)
                scoring = "accuracy"
            else:
                model = xgb.XGBRegressor(**params)
                scoring = "neg_mean_squared_error"
            
            # Evaluate model
            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                if self.task_type == "classification":
                    predictions = model.predict(X_val)
                    score = accuracy_score(y_val, predictions)
                else:
                    predictions = model.predict(X_val)
                    score = -mean_squared_error(y_val, predictions)  # Negative for minimization
            else:
                # Use cross-validation
                if self.task_type == "classification":
                    cv = StratifiedKFold(
                        n_splits=self.config.cross_validation_folds,
                        shuffle=True,
                        random_state=self.config.data.random_state
                    )
                else:
                    cv = self.config.cross_validation_folds
                
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                score = scores.mean()
            
            # Log trial results to MLflow (only if not already in a run)
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("objective_value", score)
                    mlflow.log_metric("trial_number", trial.number)
            except Exception as e:
                # If nested run fails, just log the score without MLflow
                logger.debug(f"Could not log trial to MLflow: {e}")
            
            return score
        
        return objective
    
    def _sample_hyperparameters(self, trial: OptunaTrial) -> Dict[str, Any]:
        """Sample hyperparameters for a trial"""
        
        # Base parameters
        params = {
            "objective": self.config.model.objective,
            "eval_metric": self.config.model.eval_metric,
            "random_state": self.config.data.random_state,
            "n_jobs": -1,
            "verbosity": 0
        }
        
        # Sample hyperparameters within configured ranges
        params.update({
            "n_estimators": trial.suggest_int(
                "n_estimators",
                self.hpo_config.n_estimators_range[0],
                self.hpo_config.n_estimators_range[1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                self.hpo_config.max_depth_range[0],
                self.hpo_config.max_depth_range[1]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.hpo_config.learning_rate_range[0],
                self.hpo_config.learning_rate_range[1],
                log=True
            ),
            "subsample": trial.suggest_float(
                "subsample",
                self.hpo_config.subsample_range[0],
                self.hpo_config.subsample_range[1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                self.hpo_config.colsample_bytree_range[0],
                self.hpo_config.colsample_bytree_range[1]
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                self.hpo_config.reg_alpha_range[0],
                self.hpo_config.reg_alpha_range[1]
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                self.hpo_config.reg_lambda_range[0],
                self.hpo_config.reg_lambda_range[1]
            )
        })
        
        return params
    
    def _get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history for analysis"""
        if self.study is None:
            return {}
        
        trials_df = self.study.trials_dataframe()
        
        history = {
            "trial_numbers": trials_df["number"].tolist(),
            "objective_values": trials_df["value"].tolist(),
            "best_values": [self.study.trials[i].value for i in range(len(self.study.trials))],
            "trial_durations": trials_df["duration"].dt.total_seconds().tolist() if "duration" in trials_df.columns else []
        }
        
        return history
    
    def _get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance from the study"""
        if self.study is None or len(self.study.trials) < 2:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return {param: float(imp) for param, imp in importance.items()}
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}
    
    def get_best_model(self, 
                       X_train: np.ndarray, 
                       y_train: pd.Series,
                       X_val: Optional[np.ndarray] = None,
                       y_val: Optional[pd.Series] = None) -> xgb.XGBModel:
        """
        Train and return the best model with optimized hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained XGBoost model with best parameters
            
        Raises:
            ValueError: If optimization hasn't been run yet
        """
        if self.best_params is None:
            raise ValueError("Optimization must be run before getting best model")
        
        logger.info("Training best model with optimized hyperparameters")
        
        # Create model with best parameters
        if self.task_type == "classification":
            model = xgb.XGBClassifier(**self.best_params)
        else:
            model = xgb.XGBRegressor(**self.best_params)
        
        # Train model
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        
        logger.info("Best model training completed")
        return model
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        if self.study is None:
            raise ValueError("No optimization results to save")
        
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
            "optimization_history": self._get_optimization_history(),
            "parameter_importance": self._get_parameter_importance(),
            "study_name": self.study.study_name,
            "task_type": self.task_type,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


def optimize_hyperparameters(config: TrainingConfig,
                            X_train: np.ndarray,
                            y_train: pd.Series,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[pd.Series] = None,
                            task_type: str = "classification") -> Tuple[Dict[str, Any], xgb.XGBModel]:
    """
    Convenience function to perform hyperparameter optimization.
    
    Args:
        config: Training configuration
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        task_type: Type of ML task
        
    Returns:
        Tuple of (optimization results, best model)
    """
    optimizer = HyperparameterOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize(X_train, y_train, X_val, y_val, task_type)
    
    # Get best model
    best_model = optimizer.get_best_model(X_train, y_train, X_val, y_val)
    
    return results, best_model


class OptimizedModelTrainer:
    """
    Enhanced model trainer that integrates hyperparameter optimization.
    Extends the base trainer with automatic hyperparameter tuning.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize optimized trainer"""
        self.config = config
        self.optimizer = None
        self.optimization_results = None
        
    def train_with_optimization(self,
                               X_train: np.ndarray,
                               y_train: pd.Series,
                               X_val: np.ndarray,
                               y_val: pd.Series,
                               task_type: str = "classification") -> Dict[str, Any]:
        """
        Train model with hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            task_type: Type of ML task
            
        Returns:
            Dictionary containing training results with optimization info
        """
        
        if not self.config.hyperparameter_optimization.enabled:
            logger.info("Hyperparameter optimization is disabled, using default parameters")
            # Fall back to regular training
            from .trainer import ModelTrainer
            trainer = ModelTrainer(self.config)
            return trainer.train()
        
        logger.info("Starting training with hyperparameter optimization")
        
        # Initialize optimizer
        self.optimizer = HyperparameterOptimizer(self.config)
        
        # Run optimization
        with mlflow.start_run(run_name=f"{self.config.mlflow.run_name}_hpo"):
            
            # Log optimization configuration
            mlflow.log_params(self.config.hyperparameter_optimization.dict())
            
            # Perform optimization
            self.optimization_results = self.optimizer.optimize(
                X_train, y_train, X_val, y_val, task_type
            )
            
            # Log optimization results
            mlflow.log_metrics({
                "best_hpo_score": self.optimization_results["best_score"],
                "n_hpo_trials": self.optimization_results["n_trials"]
            })
            
            mlflow.log_params(self.optimization_results["best_params"])
            
            # Get best model
            best_model = self.optimizer.get_best_model(X_train, y_train, X_val, y_val)
            
            # Log best model
            if task_type == "classification":
                mlflow.xgboost.log_model(best_model, "optimized_model")
            else:
                mlflow.xgboost.log_model(best_model, "optimized_model")
            
            # Save optimization artifacts
            self.optimizer.save_optimization_results("hpo_results.json")
            mlflow.log_artifact("hpo_results.json")
            
            # Prepare results
            results = {
                "model": best_model,
                "optimization_results": self.optimization_results,
                "best_params": self.optimization_results["best_params"],
                "best_score": self.optimization_results["best_score"],
                "task_type": task_type
            }
            
            logger.info("Training with hyperparameter optimization completed")
            return results