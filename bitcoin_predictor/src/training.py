# bitcoin_predictor/src/training.py
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import logging
from .model import BitcoinPriceModel
from ..utils.config import ModelConfig

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model: BitcoinPriceModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.callbacks = self._setup_callbacks()
        
    def _setup_callbacks(self) -> list:
        """Setup training callbacks"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        log_dir = Path("logs/fit") / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_lr,
                verbose=1
            ),
            ModelCheckpoint(
                str(checkpoint_dir / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(log_dir=str(log_dir))
        ]
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the model"""
        try:
            validation_data = (X_val, y_val) if X_val is not None else None
            
            history = self.model.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                validation_split=self.config.validation_split if validation_data is None else None,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=self.callbacks,
                verbose=1
            )
            
            return history.history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate model performance"""
        try:
            loss, mae = self.model.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test Loss: {loss:.4f}")
            logger.info(f"Test MAE: {mae:.4f}")
            return loss, mae
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            return self.model.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise