# bitcoin_predictor/src/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import yfinance as yf
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.feature_scalers = {}
        self.target_scaler = RobustScaler()

    def fetch_data(self, symbol: str = "BTC-USD", period: str = "max") -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        try:
            data = yf.Ticker(symbol).history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            df['MA7'] = df['Close'].rolling(window=7, min_periods=1).mean()
            df['MA21'] = df['Close'].rolling(window=21, min_periods=1).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'] = self._calculate_macd(df['Close'])
            df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            return df.ffill().bfill()
        except Exception as e:
            logger.error(f"Error adding features: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = prices.ewm(span=26, adjust=False, min_periods=1).mean()
        return exp1 - exp2

    def prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequence data for training"""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def scale_data(self, features: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scale features and target data"""
        try:
            scaled_features = np.zeros_like(features, dtype=np.float32)
            for i in range(features.shape[1]):
                self.feature_scalers[i] = RobustScaler()
                scaled_features[:, i] = self.feature_scalers[i].fit_transform(
                    features[:, i].reshape(-1, 1)
                ).ravel()
            
            scaled_target = None
            if target is not None:
                scaled_target = self.target_scaler.fit_transform(
                    target.reshape(-1, 1)
                ).ravel()
            
            return scaled_features, scaled_target
        except Exception as e:
            logger.error(f"Error scaling data: {e}")
            raise