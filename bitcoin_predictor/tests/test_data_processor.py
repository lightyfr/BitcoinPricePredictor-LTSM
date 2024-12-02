# bitcoin_predictor/tests/test_data_processor.py
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from ..src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 900, 1200, 1000]
        })
    
    def test_fetch_data(self):
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = self.sample_data
            data = self.processor.fetch_data()
            self.assertIsInstance(data, pd.DataFrame)
            mock_ticker.assert_called_once_with("BTC-USD")
    
    def test_add_features(self):
        result = self.processor.add_features(self.sample_data)
        required_features = ['MA7', 'MA21', 'RSI', 'MACD']
        for feature in required_features:
            self.assertIn(feature, result.columns)
    
    def test_calculate_rsi(self):
        rsi = self.processor._calculate_rsi(self.sample_data['Close'])
        self.assertTrue(all(0 <= x <= 100 for x in rsi))
    
    def test_calculate_macd(self):
        macd = self.processor._calculate_macd(self.sample_data['Close'])
        self.assertEqual(len(macd), len(self.sample_data))
    
    def test_prepare_sequences(self):
        features = np.random.random((100, 5))
        target = np.random.random(100)
        X, y = self.processor.prepare_sequences(features, target)
        self.assertEqual(X.shape[1], self.processor.sequence_length)
        self.assertEqual(len(X), len(y))
    
    def test_scale_data(self):
        features = np.random.random((100, 5))
        target = np.random.random(100)
        scaled_features, scaled_target = self.processor.scale_data(features, target)
        self.assertEqual(scaled_features.shape, features.shape)
        self.assertEqual(scaled_target.shape, target.shape)