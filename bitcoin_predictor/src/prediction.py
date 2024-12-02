# bitcoin_predictor/src/prediction.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .model import BitcoinPriceModel
from .data_processor import DataProcessor
from ..utils.config import ModelConfig, DataConfig

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self, model: BitcoinPriceModel, data_processor: DataProcessor):
        self.model = model
        self.data_processor = data_processor
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        try:
            predictions = self.model.model.predict(features)
            return self.data_processor.target_scaler.inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Add percentage metrics
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['accuracy'] = 100 - metrics['mape']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
            
    def save_predictions(self, predictions: np.ndarray, 
                        file_path: Path) -> None:
        """Save predictions to file"""
        try:
            pd.DataFrame(predictions, columns=['Predicted_Price']).to_csv(
                file_path, index=True
            )
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise