# bitcoin_predictor/src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from bitcoin_predictor.utils.config import ModelConfig  # Updated import
import logging

logger = logging.getLogger(__name__)

class BitcoinPriceModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build LSTM model architecture"""
        try:
            model = Sequential([
                Input(shape=(self.config.sequence_length, self.config.n_features)),
                
                LSTM(self.config.lstm_units[0], return_sequences=True,
                     kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                LSTM(self.config.lstm_units[1], return_sequences=True,
                     kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                LSTM(self.config.lstm_units[2],
                     kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                Dense(self.config.dense_units[0], activation='relu',
                     kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dense(self.config.dense_units[1])
            ])
            
            optimizer = Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=1.0
            )
            
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
            
    def summary(self):
        """Print model summary"""
        return self.model.summary()