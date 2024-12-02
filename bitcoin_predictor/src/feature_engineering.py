# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
import talib

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        # Moving averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'].values)
        
        # MACD
        df['MACD'], df['MACD_Signal'], _# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List
import talib

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        # Moving averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        df['RSI'] = talib.RSI(df['Close'].values)
        
        # MACD
        df['MACD'], df['MACD_Signal'], _