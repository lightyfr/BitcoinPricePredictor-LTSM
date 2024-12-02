# bitcoin_predictor/utils/config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    sequence_length: int = 60
    n_features: int = 10
    lstm_units: List[int] = field(default_factory=lambda: [64, 32, 16])
    dense_units: List[int] = field(default_factory=lambda: [8, 1])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.1
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

@dataclass
class DataConfig:
    symbol: str = "BTC-USD"
    period: str = "max"
    test_size: float = 0.2
    feature_columns: List[str] = field(default_factory=lambda: [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA7', 'MA21', 'RSI', 'MACD', 'Volume_MA'
    ])
    target_column: str = 'Close'