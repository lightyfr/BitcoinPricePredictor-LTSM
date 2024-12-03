# bitcoin_predictor/src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, 
    Bidirectional, Attention, MultiHeadAttention, LayerNormalization,
    Concatenate, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from bitcoin_predictor.utils.config import ModelConfig  # Updated import
import logging

logger = logging.getLogger(__name__)

class BitcoinPriceModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """Build enhanced LSTM model architecture"""
        try:
            model = Sequential([
                # Input layer
                Input(shape=(self.config.sequence_length, self.config.n_features)),
                
                # First Bidirectional LSTM layer
                Bidirectional(LSTM(self.config.lstm_units[0], 
                                 return_sequences=True,
                                 kernel_regularizer=l2(0.001),
                                 recurrent_regularizer=l2(0.001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3))),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                # Second Bidirectional LSTM layer
                Bidirectional(LSTM(self.config.lstm_units[1], 
                                 return_sequences=True,
                                 kernel_regularizer=l2(0.001),
                                 recurrent_regularizer=l2(0.001),
                                 kernel_constraint=MaxNorm(3),
                                 recurrent_constraint=MaxNorm(3))),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                # Third LSTM layer
                LSTM(self.config.lstm_units[2],
                     kernel_regularizer=l2(0.001),
                     recurrent_regularizer=l2(0.001),
                     kernel_constraint=MaxNorm(3),
                     recurrent_constraint=MaxNorm(3)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate),
                
                # Dense layers with residual connections
                Dense(self.config.dense_units[0], activation='selu',
                     kernel_regularizer=l2(0.001),
                     kernel_constraint=MaxNorm(3)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate/2),
                
                Dense(self.config.dense_units[0]//2, activation='selu',
                     kernel_regularizer=l2(0.001),
                     kernel_constraint=MaxNorm(3)),
                BatchNormalization(),
                Dropout(self.config.dropout_rate/2),
                
                Dense(1)
            ])
            
            # Use fixed learning rate instead of schedule
            optimizer = Adam(
                learning_rate=self.config.learning_rate,
                clipnorm=1.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
            
    def summary(self):
        """Print model summary"""
        return self.model.summary()