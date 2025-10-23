"""
XGBoost training workflow with MLflow integration.
Provides reproducible training with comprehensive evaluation and logging.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

# MLflow imports
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Sklearn imports for evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    log_loss, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .config import TrainingConfig, XGBoostConfig
from .data_loader import DataLoader, DataPreprocessor, create_train_test_split
from .data_validation import DataValidator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    XGBoost model trainer with MLflow integration and comprehensive evaluation.
    Supports both classification and regression tasks with reproducible training.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.task_type = None
        self.metrics = {}
        
        # Set up MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Set up MLflow tracking and experiment"""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config.mlflow.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config.mlflow.experiment_name,
                    artifact_location=self.config.mlflow.artifact_location
                )
                logger.info(f"Created new experiment: {self.config.mlflow.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.config.mlflow.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_id=experiment_id)
            
        except Exception as e:
            logger.warning(f"Failed to set up MLflow experiment: {e}")
            logger.info("Continuing with default experiment")
    
    def train(self, 
              dataset_name: Optional[str] = None,
              custom_data_path: Optional[str] = None,
              target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Train XGBoost model with comprehensive evaluation and MLflow logging.
        
        Args:
            dataset_name: Name of predefined dataset to use
            custom_data_path: Path to custom dataset file
            target_column: Target column name for custom datasets
            
        Returns:
            Dictionary containing training results and metrics
        """
        with mlflow.start_run(run_name=self.config.mlflow.run_name) as run:
            logger.info(f"Starting training run: {run.info.run_id}")
            
            try:
                # Load and validate data
                X, y = self._load_data(dataset_name, custom_data_path, target_column)
                
                # Determine task type
                self.task_type = self._determine_task_type(y)
                logger.info(f"Detected task type: {self.task_type}")
                
                # Update model configuration for task type
                self._update_model_config_for_task()
                
                # Create train/validation/test splits
                X_train, X_val, X_test, y_train, y_val, y_test = self._create_splits(X, y)
                
                # Encode target labels if classification
                if self.task_type == "classification":
                    y_train, y_val, y_test = self._encode_target_labels(y_train, y_val, y_test)
                
                # Preprocess data
                X_train_processed, X_val_processed, X_test_processed = self._preprocess_data(
                    X_train, X_val, X_test
                )
                
                # Train model (with or without hyperparameter optimization)
                if self.config.hyperparameter_optimization.enabled:
                    self.model = self._train_model_with_optimization(
                        X_train_processed, y_train,
                        X_val_processed, y_val
                    )
                else:
                    self.model = self._train_model(
                        X_train_processed, y_train,
                        X_val_processed, y_val
                    )
                
                # Evaluate model
                train_metrics = self._evaluate_model(X_train_processed, y_train, "train")
                val_metrics = self._evaluate_model(X_val_processed, y_val, "validation")
                test_metrics = self._evaluate_model(X_test_processed, y_test, "test")
                
                # Cross-validation
                cv_metrics = self._cross_validate(X_train_processed, y_train)
                
                # Combine all metrics
                all_metrics = {
                    **train_metrics,
                    **val_metrics, 
                    **test_metrics,
                    **cv_metrics
                }
                
                # Log everything to MLflow
                self._log_to_mlflow(all_metrics, X, y)
                
                # Save artifacts
                if self.config.save_model_artifacts:
                    self._save_artifacts(X_test_processed, y_test)
                
                # Register model if configured
                registered_model_version = None
                if self.config.mlflow.register_model:
                    registered_model_version = self._register_model(run.info.run_id, all_metrics)
                
                # Prepare results
                results = {
                    "run_id": run.info.run_id,
                    "model": self.model,
                    "preprocessor": self.preprocessor,
                    "label_encoder": self.label_encoder,
                    "metrics": all_metrics,
                    "task_type": self.task_type,
                    "feature_names": self.feature_names,
                    "config": self.config.dict(),
                    "registered_model_version": registered_model_version
                }
                
                logger.info("Training completed successfully")
                return results
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
    
    def _load_data(self, 
                   dataset_name: Optional[str],
                   custom_data_path: Optional[str],
                   target_column: Optional[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate data"""
        data_loader = DataLoader()
        
        # Determine data source
        if custom_data_path:
            if not target_column:
                raise ValueError("target_column must be specified for custom datasets")
            X, y = data_loader.load_custom_dataset(custom_data_path, target_column)
        else:
            dataset_name = dataset_name or self.config.data.dataset_name
            X, y = data_loader.load_dataset(dataset_name)
        
        # Validate data
        validator = DataValidator()
        validation_results = validator.validate_dataset(X, y)
        
        if validator.has_critical_issues(validation_results):
            validator.log_validation_results(validation_results)
            raise ValueError("Data validation failed with critical issues")
        
        validator.log_validation_results(validation_results)
        return X, y
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if task is classification or regression"""
        unique_values = y.nunique()
        
        # Heuristic: if target has few unique values or is categorical, it's classification
        if y.dtype == 'object' or unique_values < len(y) * 0.05:
            return "classification"
        else:
            return "regression"
    
    def _update_model_config_for_task(self):
        """Update model configuration based on task type"""
        if self.task_type == "classification":
            # Keep existing classification settings
            pass
        else:
            # Update for regression
            self.config.model.objective = "reg:squarederror"
            self.config.model.eval_metric = "rmse"
    
    def _create_splits(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, ...]:
        """Create train/validation/test splits"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = create_train_test_split(
            X, y, 
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=(self.task_type == "classification")
        )
        
        # Second split: train vs validation
        val_size = self.config.data.validation_size / (1 - self.config.data.test_size)
        X_train, X_val, y_train, y_val = create_train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.data.random_state,
            stratify=(self.task_type == "classification")
        )
        
        logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _encode_target_labels(self, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Encode target labels to start from 0 for XGBoost compatibility"""
        self.label_encoder = LabelEncoder()
        
        # Fit on training data
        y_train_encoded = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
        y_val_encoded = pd.Series(self.label_encoder.transform(y_val), index=y_val.index)
        y_test_encoded = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)
        
        logger.info(f"Encoded target labels: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def _preprocess_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """Preprocess data using the configured preprocessor"""
        self.preprocessor = DataPreprocessor(
            numeric_strategy=self.config.data.numeric_strategy,
            categorical_strategy=self.config.data.categorical_strategy,
            scaling_method=self.config.data.scaling_method,
            encoding_method=self.config.data.encoding_method
        )
        
        # Fit on training data and transform all sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Store feature names
        try:
            self.feature_names = self.preprocessor.get_feature_names()
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            self.feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def _train_model(self, X_train: np.ndarray, y_train: pd.Series,
                     X_val: np.ndarray, y_val: pd.Series) -> xgb.XGBModel:
        """Train XGBoost model with early stopping"""
        
        # Get XGBoost parameters
        xgb_params = self.config.model.to_xgboost_params()
        
        # Create model based on task type
        if self.task_type == "classification":
            model = xgb.XGBClassifier(**xgb_params)
        else:
            model = xgb.XGBRegressor(**xgb_params)
        
        # Prepare evaluation set for early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        logger.info("Starting XGBoost training...")
        
        # For XGBoost 3.x, we'll train without early stopping for simplicity
        # and rely on the n_estimators parameter for training control
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        # Log training completion
        try:
            best_iter = getattr(model, 'best_iteration', 'N/A')
            logger.info(f"Training completed. Best iteration: {best_iter}")
        except:
            logger.info("Training completed.")
        
        return model
    
    def _train_model_with_optimization(self, X_train: np.ndarray, y_train: pd.Series,
                                     X_val: np.ndarray, y_val: pd.Series) -> xgb.XGBModel:
        """Train XGBoost model with hyperparameter optimization"""
        
        try:
            from .hyperparameter_optimizer import HyperparameterOptimizer, OPTUNA_AVAILABLE
            
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna not available")
            
            logger.info("Starting hyperparameter optimization...")
            
            # Initialize optimizer
            optimizer = HyperparameterOptimizer(self.config)
            
            # Run optimization
            optimization_results = optimizer.optimize(
                X_train, y_train, X_val, y_val, self.task_type
            )
            
            # Log optimization results
            mlflow.log_metrics({
                "best_hpo_score": optimization_results["best_score"],
                "n_hpo_trials": optimization_results["n_trials"]
            })
            
            # Get best model
            best_model = optimizer.get_best_model(X_train, y_train, X_val, y_val)
            
            # Save optimization artifacts
            optimizer.save_optimization_results("hpo_results.json")
            mlflow.log_artifact("hpo_results.json")
            
            logger.info(f"Hyperparameter optimization completed. Best score: {optimization_results['best_score']:.4f}")
            return best_model
            
        except ImportError:
            logger.warning("Optuna not available, falling back to default parameters")
            return self._train_model(X_train, y_train, X_val, y_val)
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            logger.info("Falling back to default parameters")
            return self._train_model(X_train, y_train, X_val, y_val)
    
    def _evaluate_model(self, X: np.ndarray, y: pd.Series, split_name: str) -> Dict[str, float]:
        """Evaluate model performance on given dataset"""
        predictions = self.model.predict(X)
        
        metrics = {}
        
        if self.task_type == "classification":
            # Classification metrics
            metrics[f"{split_name}_accuracy"] = accuracy_score(y, predictions)
            
            # Handle multi-class vs binary classification
            if len(np.unique(y)) == 2:
                # Binary classification
                metrics[f"{split_name}_precision"] = precision_score(y, predictions)
                metrics[f"{split_name}_recall"] = recall_score(y, predictions)
                metrics[f"{split_name}_f1"] = f1_score(y, predictions)
                
                # ROC AUC for binary classification
                try:
                    y_proba = self.model.predict_proba(X)[:, 1]
                    metrics[f"{split_name}_roc_auc"] = roc_auc_score(y, y_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
            else:
                # Multi-class classification
                metrics[f"{split_name}_precision"] = precision_score(y, predictions, average='weighted')
                metrics[f"{split_name}_recall"] = recall_score(y, predictions, average='weighted')
                metrics[f"{split_name}_f1"] = f1_score(y, predictions, average='weighted')
                
                # Multi-class ROC AUC
                try:
                    y_proba = self.model.predict_proba(X)
                    metrics[f"{split_name}_roc_auc"] = roc_auc_score(y, y_proba, multi_class='ovr')
                except Exception as e:
                    logger.warning(f"Could not calculate multi-class ROC AUC: {e}")
            
            # Log loss
            try:
                y_proba = self.model.predict_proba(X)
                metrics[f"{split_name}_log_loss"] = log_loss(y, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate log loss: {e}")
        
        else:
            # Regression metrics
            metrics[f"{split_name}_mse"] = mean_squared_error(y, predictions)
            metrics[f"{split_name}_rmse"] = np.sqrt(metrics[f"{split_name}_mse"])
            metrics[f"{split_name}_mae"] = mean_absolute_error(y, predictions)
            metrics[f"{split_name}_r2"] = r2_score(y, predictions)
        
        logger.info(f"{split_name.capitalize()} metrics calculated: {len(metrics)} metrics")
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info("Performing cross-validation...")
        
        cv_folds = self.config.cross_validation_folds
        
        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.data.random_state)
            scoring = 'accuracy'
        else:
            cv = cv_folds
            scoring = 'neg_mean_squared_error'
        
        # Create a fresh model for CV (to avoid data leakage)
        xgb_params = self.config.model.to_xgboost_params()
        if self.task_type == "classification":
            cv_model = xgb.XGBClassifier(**xgb_params)
        else:
            cv_model = xgb.XGBRegressor(**xgb_params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Calculate statistics
        cv_metrics = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_min": cv_scores.min(),
            "cv_max": cv_scores.max()
        }
        
        logger.info(f"Cross-validation completed: {cv_metrics['cv_mean']:.4f} ± {cv_metrics['cv_std']:.4f}")
        return cv_metrics
    
    def _log_to_mlflow(self, metrics: Dict[str, float], X: pd.DataFrame, y: pd.Series):
        """Log parameters, metrics, and model to MLflow"""
        
        # Log parameters
        mlflow.log_params(self.config.model.dict())
        mlflow.log_params(self.config.data.dict())
        
        # Log dataset info
        mlflow.log_param("dataset_shape", f"{X.shape[0]}x{X.shape[1]}")
        mlflow.log_param("task_type", self.task_type)
        mlflow.log_param("n_features", len(self.feature_names))
        
        if self.task_type == "classification":
            mlflow.log_param("n_classes", y.nunique())
            mlflow.log_param("class_distribution", dict(y.value_counts()))
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model with MLflow
        if self.task_type == "classification":
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name=self.config.mlflow.model_name if self.config.mlflow.register_model else None
            )
        else:
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name=self.config.mlflow.model_name if self.config.mlflow.register_model else None
            )
        
        # Log preprocessor
        joblib.dump(self.preprocessor, "preprocessor.pkl")
        mlflow.log_artifact("preprocessor.pkl")
        
        # Log label encoder if classification
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, "label_encoder.pkl")
            mlflow.log_artifact("label_encoder.pkl")
        
        # Log feature names
        with open("feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        mlflow.log_artifact("feature_names.json")
        
        # Log training configuration
        with open("training_config.json", "w") as f:
            json.dump(self.config.dict(), f, indent=2)
        mlflow.log_artifact("training_config.json")
        
        logger.info("Logged all artifacts to MLflow")
    
    def _save_artifacts(self, X_test: np.ndarray, y_test: pd.Series):
        """Save additional artifacts"""
        
        # Save test predictions
        if self.config.save_predictions:
            test_predictions = self.model.predict(X_test)
            
            predictions_df = pd.DataFrame({
                "true_values": y_test.values,
                "predictions": test_predictions
            })
            
            if self.task_type == "classification":
                try:
                    test_probabilities = self.model.predict_proba(X_test)
                    for i, class_name in enumerate(self.model.classes_):
                        predictions_df[f"prob_class_{class_name}"] = test_probabilities[:, i]
                except Exception as e:
                    logger.warning(f"Could not save prediction probabilities: {e}")
            
            predictions_df.to_csv("test_predictions.csv", index=False)
            mlflow.log_artifact("test_predictions.csv")
        
        # Save feature importance
        try:
            importance_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            logger.info("Saved feature importance")
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")
    
    def _register_model(self, run_id: str, metrics: Dict[str, float]):
        """Register model to MLflow registry with promotion workflow"""
        try:
            from .model_registry_manager import ModelPromotionWorkflow
            
            logger.info("Registering model to MLflow registry...")
            
            # Initialize promotion workflow
            workflow = ModelPromotionWorkflow(self.config)
            
            # Register and promote model
            model_version = workflow.register_and_promote_model(
                run_id=run_id,
                metrics=metrics,
                model_path="model"
            )
            
            logger.info(f"Model registered as version {model_version.version} in stage {model_version.current_stage}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")


def train_model(config: Optional[TrainingConfig] = None,
                dataset_name: Optional[str] = None,
                custom_data_path: Optional[str] = None,
                target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to train a model with given configuration.
    
    Args:
        config: Training configuration (uses default if None)
        dataset_name: Name of predefined dataset
        custom_data_path: Path to custom dataset
        target_column: Target column for custom dataset
        
    Returns:
        Training results dictionary
    """
    if config is None:
        from .config import get_training_config
        config = get_training_config()
    
    trainer = ModelTrainer(config)
    return trainer.train(dataset_name, custom_data_path, target_column)