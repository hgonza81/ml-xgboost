"""
Unit tests for MLflow model loader integration.

Tests the core functionality of model loading, caching, and error handling
without requiring a full MLflow server setup.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.api.model_loader import (
    MLflowModelLoader,
    ModelLoadError,
    ModelMetadata,
    get_model_loader
)


class TestModelMetadata:
    """Test ModelMetadata class functionality."""
    
    def test_model_metadata_creation(self):
        """Test creating ModelMetadata instance."""
        metadata = ModelMetadata(
            name="test_model",
            version="1",
            stage="production",
            run_id="test_run_123",
            model_uri="models:/test_model/1",
            loaded_at=datetime.utcnow(),
            metrics={"accuracy": 0.95}
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1"
        assert metadata.stage == "production"
        assert metadata.run_id == "test_run_123"
        assert metadata.metrics["accuracy"] == 0.95
    
    def test_model_metadata_to_dict(self):
        """Test converting ModelMetadata to dictionary."""
        loaded_at = datetime.utcnow()
        metadata = ModelMetadata(
            name="test_model",
            version="1",
            stage="production",
            run_id="test_run_123",
            model_uri="models:/test_model/1",
            loaded_at=loaded_at,
            metrics={"accuracy": 0.95}
        )
        
        result = metadata.to_dict()
        
        assert result["name"] == "test_model"
        assert result["version"] == "1"
        assert result["stage"] == "production"
        assert result["run_id"] == "test_run_123"
        assert result["model_uri"] == "models:/test_model/1"
        assert result["loaded_at"] == loaded_at.isoformat()
        assert result["metrics"]["accuracy"] == 0.95


class TestMLflowModelLoader:
    """Test MLflowModelLoader class functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with mock.patch('src.api.model_loader.get_settings') as mock_get_settings:
            mock_settings = mock.MagicMock()
            mock_settings.mlflow.tracking_uri = "http://localhost:5000"
            mock_get_settings.return_value = mock_settings
            yield mock_settings
    
    @pytest.fixture
    def model_loader(self, mock_settings):
        """Create model loader instance for testing."""
        with mock.patch('mlflow.set_tracking_uri'), \
             mock.patch('src.api.model_loader.MlflowClient'):
            loader = MLflowModelLoader(tracking_uri="http://localhost:5000")
            return loader
    
    def test_cache_ttl_validation(self, model_loader):
        """Test cache TTL validation."""
        # Test cache validity
        now = datetime.utcnow()
        
        # Valid cache (within TTL)
        valid_time = now - timedelta(minutes=15)
        assert model_loader._is_cache_valid(valid_time) is True
        
        # Expired cache (beyond TTL)
        expired_time = now - timedelta(minutes=45)
        assert model_loader._is_cache_valid(expired_time) is False
    
    @mock.patch('src.api.model_loader.MlflowClient')
    def test_get_model_versions_success(self, mock_client_class, model_loader):
        """Test successful model version retrieval."""
        # Mock client and versions
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        model_loader.client = mock_client
        
        mock_version = mock.MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "production"
        mock_client.get_latest_versions.return_value = [mock_version]
        
        # Test getting versions by stage
        versions = model_loader._get_model_versions("test_model", "production")
        
        assert len(versions) == 1
        assert versions[0].version == "1"
        mock_client.get_latest_versions.assert_called_once_with("test_model", stages=["production"])
    
    @mock.patch('src.api.model_loader.MlflowClient')
    def test_get_model_versions_not_found(self, mock_client_class, model_loader):
        """Test model not found error handling."""
        # Mock client to raise RESOURCE_DOES_NOT_EXIST error
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        model_loader.client = mock_client
        
        from mlflow.exceptions import RestException
        mock_client.get_latest_versions.side_effect = RestException(
            {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Model not found"}
        )
        
        # Test that ModelLoadError is raised
        with pytest.raises(ModelLoadError, match="Model 'nonexistent_model' not found"):
            model_loader._get_model_versions("nonexistent_model", "production")
    
    @mock.patch('mlflow.pyfunc.load_model')
    @mock.patch('src.api.model_loader.MlflowClient')
    def test_load_model_from_uri_success(self, mock_client_class, mock_load_model, model_loader):
        """Test successful model loading from URI."""
        # Mock loaded model
        mock_model = mock.MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock client and model version
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        model_loader.client = mock_client
        
        mock_model_version = mock.MagicMock()
        mock_model_version.current_stage = "production"
        mock_model_version.run_id = "test_run_123"
        mock_client.get_model_version.return_value = mock_model_version
        
        mock_run = mock.MagicMock()
        mock_run.data.metrics = {"accuracy": 0.95}
        mock_client.get_run.return_value = mock_run
        
        # Test loading model
        model, metadata = model_loader._load_model_from_uri(
            "models:/test_model/1", "test_model", "1"
        )
        
        assert model == mock_model
        assert metadata.name == "test_model"
        assert metadata.version == "1"
        assert metadata.stage == "production"
        assert metadata.run_id == "test_run_123"
        assert metadata.metrics["accuracy"] == 0.95
    
    def test_clear_cache(self, model_loader):
        """Test cache clearing functionality."""
        # Add some items to cache
        model_loader._model_cache["model1:production"] = (mock.MagicMock(), mock.MagicMock(), datetime.utcnow())
        model_loader._model_cache["model2:staging"] = (mock.MagicMock(), mock.MagicMock(), datetime.utcnow())
        
        # Clear specific model
        model_loader.clear_cache("model1")
        assert "model1:production" not in model_loader._model_cache
        assert "model2:staging" in model_loader._model_cache
        
        # Clear all cache
        model_loader.clear_cache()
        assert len(model_loader._model_cache) == 0
    
    @mock.patch('src.api.model_loader.MlflowClient')
    def test_health_check_success(self, mock_client_class, model_loader):
        """Test successful health check."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        model_loader.client = mock_client
        
        mock_client.search_experiments.return_value = []
        
        health = model_loader.health_check()
        
        assert health["mlflow_connection"] == "healthy"
        assert health["tracking_uri"] == "http://localhost:5000"
        assert "cache_size" in health
    
    @mock.patch('src.api.model_loader.MlflowClient')
    def test_health_check_failure(self, mock_client_class, model_loader):
        """Test health check failure."""
        mock_client = mock.MagicMock()
        mock_client_class.return_value = mock_client
        model_loader.client = mock_client
        
        mock_client.search_experiments.side_effect = Exception("Connection failed")
        
        health = model_loader.health_check()
        
        assert health["mlflow_connection"] == "unhealthy"
        assert health["tracking_uri"] == "http://localhost:5000"
        assert "error" in health


class TestModelLoaderIntegration:
    """Integration tests for model loader functionality."""
    
    @mock.patch('src.api.model_loader.get_settings')
    @mock.patch('src.api.model_loader.MLflowModelLoader')
    def test_get_model_loader_singleton(self, mock_loader_class, mock_get_settings):
        """Test that get_model_loader returns singleton instance."""
        mock_instance = mock.MagicMock()
        mock_loader_class.return_value = mock_instance
        
        # Clear any existing instance
        import src.api.model_loader
        src.api.model_loader._model_loader = None
        
        # Get loader instances
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        
        # Should be the same instance
        assert loader1 == loader2
        # Should only create one instance
        mock_loader_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])