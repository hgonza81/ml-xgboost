"""
Data validation and quality checks for training pipeline.
Ensures data integrity and quality before model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a data validation check"""
    check_name: str
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    passed: bool = True


class DataValidator:
    """
    Comprehensive data validation for ML training datasets.
    Performs quality checks and identifies potential issues.
    """
    
    def __init__(self, 
                 min_samples: int = 50,
                 min_features: int = 2,
                 max_missing_ratio: float = 0.5,
                 min_class_samples: int = 5,
                 max_cardinality_ratio: float = 0.5):
        """
        Initialize data validator with thresholds.
        
        Args:
            min_samples: Minimum number of samples required
            min_features: Minimum number of features required
            max_missing_ratio: Maximum ratio of missing values per feature
            min_class_samples: Minimum samples per class for classification
            max_cardinality_ratio: Maximum cardinality ratio for categorical features
        """
        self.min_samples = min_samples
        self.min_features = min_features
        self.max_missing_ratio = max_missing_ratio
        self.min_class_samples = min_class_samples
        self.max_cardinality_ratio = max_cardinality_ratio
        
    def validate_dataset(self, X: pd.DataFrame, y: pd.Series) -> List[ValidationResult]:
        """
        Perform comprehensive validation on dataset.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            List of validation results
        """
        results = []
        
        # Basic structure validation
        results.extend(self._validate_basic_structure(X, y))
        
        # Missing values validation
        results.extend(self._validate_missing_values(X, y))
        
        # Feature validation
        results.extend(self._validate_features(X))
        
        # Target validation
        results.extend(self._validate_target(y))
        
        # Data consistency validation
        results.extend(self._validate_data_consistency(X, y))
        
        # Statistical validation
        results.extend(self._validate_statistical_properties(X, y))
        
        return results
    
    def _validate_basic_structure(self, X: pd.DataFrame, y: pd.Series) -> List[ValidationResult]:
        """Validate basic dataset structure"""
        results = []
        
        # Check if datasets are empty
        if X.empty:
            results.append(ValidationResult(
                "empty_features", ValidationSeverity.CRITICAL,
                "Features dataset is empty", {"shape": X.shape}, False
            ))
        
        if y.empty:
            results.append(ValidationResult(
                "empty_target", ValidationSeverity.CRITICAL,
                "Target dataset is empty", {"length": len(y)}, False
            ))
        
        # Check minimum samples
        if len(X) < self.min_samples:
            results.append(ValidationResult(
                "insufficient_samples", ValidationSeverity.ERROR,
                f"Dataset has {len(X)} samples, minimum required: {self.min_samples}",
                {"actual": len(X), "required": self.min_samples}, False
            ))
        
        # Check minimum features
        if X.shape[1] < self.min_features:
            results.append(ValidationResult(
                "insufficient_features", ValidationSeverity.ERROR,
                f"Dataset has {X.shape[1]} features, minimum required: {self.min_features}",
                {"actual": X.shape[1], "required": self.min_features}, False
            ))
        
        # Check shape consistency
        if len(X) != len(y):
            results.append(ValidationResult(
                "shape_mismatch", ValidationSeverity.CRITICAL,
                f"Features and target have different lengths: {len(X)} vs {len(y)}",
                {"features_length": len(X), "target_length": len(y)}, False
            ))
        
        if not results:
            results.append(ValidationResult(
                "basic_structure", ValidationSeverity.INFO,
                f"Basic structure validation passed: {X.shape[0]} samples, {X.shape[1]} features"
            ))
        
        return results
    
    def _validate_missing_values(self, X: pd.DataFrame, y: pd.Series) -> List[ValidationResult]:
        """Validate missing values in dataset"""
        results = []
        
        # Check missing values in features
        missing_counts = X.isnull().sum()
        missing_ratios = missing_counts / len(X)
        
        # Features with high missing values
        high_missing = missing_ratios[missing_ratios > self.max_missing_ratio]
        if not high_missing.empty:
            results.append(ValidationResult(
                "high_missing_features", ValidationSeverity.WARNING,
                f"Features with >{self.max_missing_ratio*100}% missing values: {list(high_missing.index)}",
                {"features": high_missing.to_dict()}, False
            ))
        
        # Completely missing features
        completely_missing = missing_ratios[missing_ratios == 1.0]
        if not completely_missing.empty:
            results.append(ValidationResult(
                "completely_missing_features", ValidationSeverity.ERROR,
                f"Features with 100% missing values: {list(completely_missing.index)}",
                {"features": list(completely_missing.index)}, False
            ))
        
        # Missing values in target
        target_missing = y.isnull().sum()
        if target_missing > 0:
            results.append(ValidationResult(
                "missing_target_values", ValidationSeverity.ERROR,
                f"Target has {target_missing} missing values ({target_missing/len(y)*100:.1f}%)",
                {"count": target_missing, "ratio": target_missing/len(y)}, False
            ))
        
        # Summary of missing values
        total_missing = X.isnull().sum().sum()
        if total_missing == 0:
            results.append(ValidationResult(
                "no_missing_values", ValidationSeverity.INFO,
                "No missing values found in features"
            ))
        else:
            results.append(ValidationResult(
                "missing_values_summary", ValidationSeverity.INFO,
                f"Total missing values in features: {total_missing} ({total_missing/(len(X)*len(X.columns))*100:.1f}%)",
                {"total_missing": total_missing, "total_cells": len(X)*len(X.columns)}
            ))
        
        return results
    
    def _validate_features(self, X: pd.DataFrame) -> List[ValidationResult]:
        """Validate feature properties"""
        results = []
        
        # Check for constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            results.append(ValidationResult(
                "constant_features", ValidationSeverity.WARNING,
                f"Constant features found: {constant_features}",
                {"features": constant_features}, False
            ))
        
        # Check for duplicate features
        duplicate_features = []
        for i, col1 in enumerate(X.columns):
            for col2 in X.columns[i+1:]:
                if X[col1].equals(X[col2]):
                    duplicate_features.append((col1, col2))
        
        if duplicate_features:
            results.append(ValidationResult(
                "duplicate_features", ValidationSeverity.WARNING,
                f"Duplicate feature pairs found: {duplicate_features}",
                {"pairs": duplicate_features}, False
            ))
        
        # Check categorical feature cardinality
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        
        for col in categorical_cols:
            cardinality_ratio = X[col].nunique() / len(X)
            if cardinality_ratio > self.max_cardinality_ratio:
                high_cardinality.append((col, cardinality_ratio))
        
        if high_cardinality:
            results.append(ValidationResult(
                "high_cardinality_features", ValidationSeverity.WARNING,
                f"High cardinality categorical features: {[f[0] for f in high_cardinality]}",
                {"features": dict(high_cardinality)}, False
            ))
        
        # Check for infinite values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        infinite_features = []
        
        for col in numeric_cols:
            if np.isinf(X[col]).any():
                infinite_features.append(col)
        
        if infinite_features:
            results.append(ValidationResult(
                "infinite_values", ValidationSeverity.ERROR,
                f"Features with infinite values: {infinite_features}",
                {"features": infinite_features}, False
            ))
        
        return results
    
    def _validate_target(self, y: pd.Series) -> List[ValidationResult]:
        """Validate target variable properties"""
        results = []
        
        # Check target type and distribution
        unique_values = y.nunique()
        value_counts = y.value_counts()
        
        # Determine if classification or regression
        is_classification = unique_values < len(y) * 0.05 or y.dtype == 'object'
        
        if is_classification:
            # Classification validation
            results.append(ValidationResult(
                "target_type", ValidationSeverity.INFO,
                f"Classification target detected with {unique_values} classes",
                {"n_classes": unique_values, "classes": list(y.unique())}
            ))
            
            # Check class balance
            min_class_count = value_counts.min()
            max_class_count = value_counts.max()
            imbalance_ratio = max_class_count / min_class_count
            
            if min_class_count < self.min_class_samples:
                results.append(ValidationResult(
                    "insufficient_class_samples", ValidationSeverity.ERROR,
                    f"Some classes have fewer than {self.min_class_samples} samples",
                    {"min_samples": min_class_count, "class_counts": value_counts.to_dict()}, False
                ))
            
            if imbalance_ratio > 10:
                results.append(ValidationResult(
                    "class_imbalance", ValidationSeverity.WARNING,
                    f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f})",
                    {"imbalance_ratio": imbalance_ratio, "class_counts": value_counts.to_dict()}, False
                ))
        
        else:
            # Regression validation
            results.append(ValidationResult(
                "target_type", ValidationSeverity.INFO,
                f"Regression target detected with {unique_values} unique values",
                {"n_unique": unique_values, "min": y.min(), "max": y.max()}
            ))
            
            # Check for outliers in regression target
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 1.5 * IQR
            outliers = ((y < (Q1 - outlier_threshold)) | (y > (Q3 + outlier_threshold))).sum()
            
            if outliers > 0:
                results.append(ValidationResult(
                    "target_outliers", ValidationSeverity.WARNING,
                    f"Target has {outliers} potential outliers ({outliers/len(y)*100:.1f}%)",
                    {"outlier_count": outliers, "outlier_ratio": outliers/len(y)}
                ))
        
        return results
    
    def _validate_data_consistency(self, X: pd.DataFrame, y: pd.Series) -> List[ValidationResult]:
        """Validate data consistency and integrity"""
        results = []
        
        # Check for duplicate rows
        duplicate_rows = X.duplicated().sum()
        if duplicate_rows > 0:
            results.append(ValidationResult(
                "duplicate_rows", ValidationSeverity.WARNING,
                f"Found {duplicate_rows} duplicate rows ({duplicate_rows/len(X)*100:.1f}%)",
                {"count": duplicate_rows, "ratio": duplicate_rows/len(X)}
            ))
        
        # Check index consistency
        if not X.index.equals(y.index):
            results.append(ValidationResult(
                "index_mismatch", ValidationSeverity.ERROR,
                "Features and target have different indices",
                {"features_index": str(X.index), "target_index": str(y.index)}, False
            ))
        
        return results
    
    def _validate_statistical_properties(self, X: pd.DataFrame, y: pd.Series) -> List[ValidationResult]:
        """Validate statistical properties of the data"""
        results = []
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Check for features with zero variance
            zero_variance = []
            for col in numeric_cols:
                if X[col].var() == 0:
                    zero_variance.append(col)
            
            if zero_variance:
                results.append(ValidationResult(
                    "zero_variance_features", ValidationSeverity.WARNING,
                    f"Features with zero variance: {zero_variance}",
                    {"features": zero_variance}, False
                ))
            
            # Check for highly skewed features
            highly_skewed = []
            for col in numeric_cols:
                skewness = abs(X[col].skew())
                if skewness > 2:  # Threshold for high skewness
                    highly_skewed.append((col, skewness))
            
            if highly_skewed:
                results.append(ValidationResult(
                    "highly_skewed_features", ValidationSeverity.INFO,
                    f"Highly skewed features (|skew| > 2): {[f[0] for f in highly_skewed]}",
                    {"features": dict(highly_skewed)}
                ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            Summary dictionary with counts and critical issues
        """
        summary = {
            "total_checks": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "by_severity": {
                "info": sum(1 for r in results if r.severity == ValidationSeverity.INFO),
                "warning": sum(1 for r in results if r.severity == ValidationSeverity.WARNING),
                "error": sum(1 for r in results if r.severity == ValidationSeverity.ERROR),
                "critical": sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
            },
            "critical_issues": [r.message for r in results if r.severity == ValidationSeverity.CRITICAL and not r.passed],
            "errors": [r.message for r in results if r.severity == ValidationSeverity.ERROR and not r.passed],
            "warnings": [r.message for r in results if r.severity == ValidationSeverity.WARNING and not r.passed]
        }
        
        return summary
    
    def has_critical_issues(self, results: List[ValidationResult]) -> bool:
        """Check if validation results contain critical issues that prevent training"""
        return any(r.severity == ValidationSeverity.CRITICAL and not r.passed for r in results)
    
    def log_validation_results(self, results: List[ValidationResult]) -> None:
        """Log validation results with appropriate log levels"""
        summary = self.get_validation_summary(results)
        
        logger.info(f"Data validation completed: {summary['passed']}/{summary['total_checks']} checks passed")
        
        for result in results:
            if result.severity == ValidationSeverity.CRITICAL:
                logger.critical(f"CRITICAL: {result.message}")
            elif result.severity == ValidationSeverity.ERROR:
                logger.error(f"ERROR: {result.message}")
            elif result.severity == ValidationSeverity.WARNING:
                logger.warning(f"WARNING: {result.message}")
            else:
                logger.info(f"INFO: {result.message}")
        
        if summary["critical_issues"]:
            logger.critical(f"Training cannot proceed due to {len(summary['critical_issues'])} critical issues")
        elif summary["errors"]:
            logger.error(f"Training may fail due to {len(summary['errors'])} errors")
        elif summary["warnings"]:
            logger.warning(f"Training may have issues due to {len(summary['warnings'])} warnings")
        else:
            logger.info("All validation checks passed successfully")