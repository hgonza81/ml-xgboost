"""
Data loading and preprocessing pipeline for tabular datasets.
Supports public datasets with comprehensive validation and quality checks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path
import requests
import zipfile
import io

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Exception raised for data quality issues"""
    pass


class DataLoader:
    """
    Data loader for public tabular datasets with preprocessing capabilities.
    Supports UCI datasets and other common ML datasets.
    """
    
    # Predefined dataset configurations
    DATASETS = {
        "wine": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "target_column": 0,
            "column_names": [
                "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                "proanthocyanins", "color_intensity", "hue", "od280_od315_of_diluted_wines",
                "proline"
            ],
            "task_type": "classification"
        },
        "iris": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "target_column": -1,
            "column_names": [
                "sepal_length", "sepal_width", "petal_length", "petal_width", "class"
            ],
            "task_type": "classification"
        },
        "breast_cancer": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            "target_column": 1,
            "column_names": [
                "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean",
                "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean",
                "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
                "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
            ],
            "task_type": "classification",
            "drop_columns": ["id"]  # ID column should be dropped
        }
    }
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """
        Initialize data loader with caching directory.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a predefined dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            ValueError: If dataset name is not supported
            DataQualityError: If data quality checks fail
        """
        if dataset_name not in self.DATASETS:
            available = ", ".join(self.DATASETS.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
        
        config = self.DATASETS[dataset_name]
        cache_file = self.cache_dir / f"{dataset_name}.csv"
        
        # Load from cache if available
        if cache_file.exists():
            logger.info(f"Loading {dataset_name} from cache: {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            logger.info(f"Downloading {dataset_name} from {config['url']}")
            df = self._download_dataset(config, cache_file)
        
        # Validate data quality
        self._validate_data_quality(df, dataset_name)
        
        # Split features and target
        X, y = self._split_features_target(df, config)
        
        logger.info(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def load_custom_dataset(self, file_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a custom dataset from file.
        
        Args:
            file_path: Path to the dataset file (CSV)
            target_column: Name or index of the target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            DataQualityError: If data quality checks fail
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading custom dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate data quality
        self._validate_data_quality(df, "custom")
        
        # Split features and target
        if isinstance(target_column, str):
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            # Assume integer index
            y = df.iloc[:, target_column]
            X = df.drop(df.columns[target_column], axis=1)
        
        logger.info(f"Loaded custom dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _download_dataset(self, config: Dict[str, Any], cache_file: Path) -> pd.DataFrame:
        """Download dataset from URL and cache it."""
        try:
            response = requests.get(config["url"], timeout=30)
            response.raise_for_status()
            
            # Read data into DataFrame
            data = io.StringIO(response.text)
            df = pd.read_csv(data, header=None, names=config["column_names"])
            
            # Drop specified columns if any
            if "drop_columns" in config:
                df = df.drop(columns=config["drop_columns"])
            
            # Cache the dataset
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached dataset to {cache_file}")
            
            return df
            
        except requests.RequestException as e:
            raise DataQualityError(f"Failed to download dataset: {e}")
        except Exception as e:
            raise DataQualityError(f"Failed to process dataset: {e}")
    
    def _split_features_target(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and target."""
        target_col = config["target_column"]
        
        if isinstance(target_col, int):
            if target_col < 0:
                target_col = len(df.columns) + target_col
            y = df.iloc[:, target_col]
            X = df.drop(df.columns[target_col], axis=1)
        else:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        
        return X, y
    
    def _validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Validate data quality and raise errors for critical issues.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for error messages
            
        Raises:
            DataQualityError: If critical data quality issues are found
        """
        if df.empty:
            raise DataQualityError(f"Dataset '{dataset_name}' is empty")
        
        if df.shape[0] < 10:
            raise DataQualityError(f"Dataset '{dataset_name}' has too few samples: {df.shape[0]}")
        
        if df.shape[1] < 2:
            raise DataQualityError(f"Dataset '{dataset_name}' has too few columns: {df.shape[1]}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Dataset '{dataset_name}' has completely empty columns: {empty_cols}")
        
        # Check missing value percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            logger.warning(f"Dataset '{dataset_name}' has columns with >50% missing values: {high_missing}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Dataset '{dataset_name}' has {duplicates} duplicate rows")
        
        logger.info(f"Data quality validation passed for '{dataset_name}'")


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline with missing value handling,
    categorical encoding, and feature scaling.
    """
    
    def __init__(self, 
                 numeric_strategy: str = "median",
                 categorical_strategy: str = "most_frequent",
                 scaling_method: str = "standard",
                 encoding_method: str = "onehot",
                 handle_unknown: str = "ignore"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            numeric_strategy: Strategy for numeric missing values ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical missing values ('most_frequent', 'constant')
            scaling_method: Scaling method ('standard', 'minmax', 'robust', 'none')
            encoding_method: Categorical encoding ('onehot', 'label', 'target')
            handle_unknown: How to handle unknown categories ('ignore', 'error')
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        
        self.preprocessor = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional, used for target encoding)
            
        Returns:
            Transformed feature array
        """
        self._identify_feature_types(X)
        self._build_preprocessor()
        
        logger.info(f"Fitting preprocessor on {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Numeric features: {len(self.numeric_features)}, Categorical: {len(self.categorical_features)}")
        
        X_transformed = self.preprocessor.fit_transform(X)
        self._store_feature_names(X)
        
        logger.info(f"Preprocessing complete. Output shape: {X_transformed.shape}")
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed feature array
            
        Raises:
            ValueError: If preprocessor hasn't been fitted
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        return self.preprocessor.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.feature_names is None:
            raise ValueError("Feature names not available. Fit preprocessor first.")
        return self.feature_names
    
    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Identify numeric and categorical features."""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle boolean columns as categorical
        bool_features = X.select_dtypes(include=['bool']).columns.tolist()
        self.categorical_features.extend(bool_features)
        
        logger.info(f"Identified {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features")
    
    def _build_preprocessor(self) -> None:
        """Build the preprocessing pipeline."""
        transformers = []
        
        # Numeric preprocessing
        if self.numeric_features:
            numeric_steps = [('imputer', SimpleImputer(strategy=self.numeric_strategy))]
            
            if self.scaling_method == 'standard':
                numeric_steps.append(('scaler', StandardScaler()))
            elif self.scaling_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif self.scaling_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                numeric_steps.append(('scaler', RobustScaler()))
            
            numeric_pipeline = Pipeline(numeric_steps)
            transformers.append(('numeric', numeric_pipeline, self.numeric_features))
        
        # Categorical preprocessing
        if self.categorical_features:
            categorical_steps = [('imputer', SimpleImputer(strategy=self.categorical_strategy, fill_value='missing'))]
            
            if self.encoding_method == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)))
            elif self.encoding_method == 'label':
                categorical_steps.append(('encoder', LabelEncoder()))
            
            categorical_pipeline = Pipeline(categorical_steps)
            transformers.append(('categorical', categorical_pipeline, self.categorical_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    def _store_feature_names(self, X: pd.DataFrame) -> None:
        """Store feature names after preprocessing."""
        feature_names = []
        
        # Get feature names from ColumnTransformer
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            try:
                feature_names = self.preprocessor.get_feature_names_out().tolist()
            except Exception:
                # Fallback to manual construction
                feature_names = self._construct_feature_names_manually(X)
        else:
            feature_names = self._construct_feature_names_manually(X)
        
        self.feature_names = feature_names
    
    def _construct_feature_names_manually(self, X: pd.DataFrame) -> List[str]:
        """Manually construct feature names when automatic method fails."""
        feature_names = []
        
        # Numeric features keep their names
        feature_names.extend(self.numeric_features)
        
        # Categorical features get expanded names for one-hot encoding
        if self.encoding_method == 'onehot' and self.categorical_features:
            for col in self.categorical_features:
                unique_values = X[col].fillna('missing').unique()
                for val in unique_values:
                    feature_names.append(f"{col}_{val}")
        else:
            feature_names.extend(self.categorical_features)
        
        return feature_names


def create_train_test_split(X: pd.DataFrame, 
                          y: pd.Series, 
                          test_size: float = 0.2, 
                          random_state: int = 42,
                          stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create train/test split with optional stratification.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split based on target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Created train/test split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test
        
    except ValueError as e:
        if "stratify" in str(e):
            logger.warning("Stratification failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=None
            )
            return X_train, X_test, y_train, y_test
        else:
            raise