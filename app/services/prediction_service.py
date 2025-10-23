"""
Prediction service containing business logic for model inference.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Tuple, Any
from datetime import datetime

from app.services.model_service import ModelService
from app.schemas.prediction import PredictionRequest, PredictionResponse, ModelMetadata

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service class for handling prediction business logic.
    """
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
    
    async def predict(self, request: PredictionRequest, model_name: str, model_stage: str) -> PredictionResponse:
        """
        Process prediction request and return response.
        
        Args:
            request: Validated prediction request
            model_name: Name of the model to use
            model_stage: Stage of the model (production, staging, etc.)
            
        Returns:
            PredictionResponse with predictions and metadata
            
        Raises:
            ValueError: If data conversion fails
            Exception: If model loading or prediction fails
        """
        logger.info(f"Processing prediction request with {len(request.data)} samples")
        
        # Convert input data to pandas DataFrame
        try:
            input_df = pd.DataFrame(request.data)
            logger.info(f"Converted {len(input_df)} samples to DataFrame with columns: {list(input_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to convert input data to DataFrame: {e}")
            raise ValueError(f"Invalid data format for model input: {str(e)}")
        
        # Load model and make predictions
        model, metadata = await self.model_service.load_model(model_name, model_stage)
        logger.info(f"Loaded model '{metadata.name}' version '{metadata.version}' from stage '{metadata.stage}'")
        
        # Make predictions using the loaded model
        predictions = model.predict(input_df)
        
        # Process predictions to ensure JSON serialization
        processed_predictions = self._process_predictions(predictions)
        
        logger.info(f"Generated {len(processed_predictions)} predictions using model version {metadata.version}")
        
        # Create response with model metadata
        model_metadata = ModelMetadata(
            name=metadata.name,
            version=metadata.version,
            stage=metadata.stage,
            run_id=metadata.run_id
        )
        
        response = PredictionResponse(
            predictions=processed_predictions,
            model_version=f"{metadata.name}-v{metadata.version}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            samples_processed=len(request.data),
            model_metadata=model_metadata
        )
        
        return response
    
    def _process_predictions(self, predictions: Any) -> List[Union[int, float]]:
        """
        Process model predictions to ensure JSON serialization.
        
        Args:
            predictions: Raw predictions from model
            
        Returns:
            List of processed predictions
        """
        # Convert predictions to list and ensure they are JSON serializable
        if isinstance(predictions, np.ndarray):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)
        
        # Ensure predictions are numeric (int or float)
        processed_predictions = []
        for pred in predictions_list:
            if isinstance(pred, (int, float, np.integer, np.floating)):
                processed_predictions.append(float(pred) if isinstance(pred, (np.floating, float)) else int(pred))
            else:
                # Handle string predictions or other types
                try:
                    processed_predictions.append(float(pred))
                except (ValueError, TypeError):
                    processed_predictions.append(str(pred))
        
        return processed_predictions