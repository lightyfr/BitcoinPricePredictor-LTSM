# Add these test methods to the TestPredictionEngine class

def test_predict_with_empty_features(self):
    # Test handling of empty features array
    empty_features = np.array([])
    
    with self.assertRaises(ValueError):
        self.predictor.predict(empty_features)
        
def test_predict_with_invalid_shape(self):
    """Test handling of features with wrong dimensions"""
    invalid_features = np.random.random((100, 30))  # Missing sequence dimension
    
    with self.assertRaises(ValueError):
        self.predictor.predict(invalid_features)

def test_predict_with_nan_values(self):
    """Test handling of NaN values in features"""
    features = np.random.random((100, 60, 10))
    features[0, 0, 0] = np.nan
    
    with self.assertRaises(ValueError):
        self.predictor.predict(features)

def test_evaluate_empty_arrays(self):
    # Test evaluation with empty arrays
    empty_true = np.array([])
    empty_pred = np.array([])
    
    with self.assertRaises(ValueError):
        self.predictor.evaluate(empty_true, empty_pred)

def test_evaluate_mismatched_shapes(self):
    # Test evaluation with mismatched array shapes
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200])
    
    with self.assertRaises(ValueError):
        self.predictor.evaluate(y_true, y_pred)

def test_evaluate_metrics(self):
    """Test calculation of all evaluation metrics"""
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 160, 190, 240, 310])
    
    metrics = self.predictor.evaluate(y_true, y_pred)
    
    required_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape', 'accuracy']
    for metric in required_metrics:
        self.assertIn(metric, metrics)
        self.assertIsInstance(metrics[metric], float)
        self.assertFalse(np.isnan(metrics[metric]))

def test_save_predictions_invalid_path(self):
    # Test saving to invalid file path
    predictions = np.array([[100], [200]])
    invalid_path = '/invalid/path/predictions.csv'
    
    with self.assertRaises(OSError):
        self.predictor.save_predictions(predictions, invalid_path)
        
def test_save_predictions_empty_array(self):
    # Test saving empty predictions
    empty_predictions = np.array([])
    file_path = 'empty_predictions.csv'
    
    with self.assertRaises(ValueError):
        self.predictor.save_predictions(empty_predictions, file_path)
        self.data_config = DataConfig()
        
        # Mock model and data processor
        self.mock_model = Mock(spec=BitcoinPriceModel)
        self.mock_processor = Mock(spec=DataProcessor)
        
        self.predictor = PredictionEngine(self.mock_model, self.mock_processor)
        
    def test_predict(self):
        # Setup mock data
        mock_features = np.random.random((100, 60, 10))
        mock_predictions = np.random.random((100, 1))
        
        # Configure mocks
        self.mock_model.model.predict.return_value = mock_predictions
        self.mock_processor.target_scaler.inverse_transform.return_value = mock_predictions * 1000
        
        # Make prediction
        predictions = self.predictor.predict(mock_features)
        
        # Assertions
        self.mock_model.model.predict.assert_called_once_with(mock_features)
        self.mock_processor.target_scaler.inverse_transform.assert_called_once()
        self.assertEqual(predictions.shape, (100, 1))
        
    def test_evaluate(self):
        # Setup test data
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        # Calculate metrics
        metrics = self.predictor.evaluate(y_true, y_pred)
        
        # Assertions
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('accuracy', metrics)
        
        # Check basic metric properties
        self.assertGreaterEqual(metrics['r2'], 0)
        self.assertLessEqual(metrics['r2'], 1)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 100)
        
    @patch('pandas.DataFrame.to_csv')
    def test_save_predictions(self, mock_to_csv):
        """Test saving predictions to file"""
        predictions = np.array([[100], [200], [300]])
        temp_file = Path('test_predictions.csv')
        
        try:
            self.predictor.save_predictions(predictions, temp_file)
            self.assertTrue(temp_file.exists())
            
            # Verify file contents
            saved_data = pd.read_csv(temp_file)
            self.assertEqual(len(saved_data), len(predictions))
            self.assertIn('Predicted_Price', saved_data.columns)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_model_integration(self):
        """Test end-to-end model integration"""
        # Setup test data
        X = np.random.random((100, 60, 10))
        y = np.random.random(100)
        
        # Configure mock model
        self.mock_model.model.predict.return_value = np.random.random((100, 1))
        self.mock_processor.target_scaler.inverse_transform.return_value = np.random.random((100, 1))
        
        # Test prediction flow
        predictions = self.predictor.predict(X)
        
        # Verify predictions shape and type
        self.assertEqual(predictions.shape, (100, 1))
        self.assertEqual(predictions.dtype, np.float64)
        
        # Verify mock calls
        self.mock_model.model.predict.assert_called_once()
        self.mock_processor.target_scaler.inverse_transform.assert_called_once()

    def test_error_handling(self):
        """Test error handling in prediction process"""
        # Setup mock to raise exception
        self.mock_model.model.predict.side_effect = RuntimeError("Model prediction failed")
        
        with self.assertRaises(RuntimeError):
            self.predictor.predict(np.random.random((100, 60, 10)))

    def test_feature_scaling(self):
        """Test feature scaling in prediction pipeline"""
        features = np.random.random((100, 60, 10))
        
        # Configure mock scalers
        self.mock_processor.feature_scalers = {
            i: Mock() for i in range(10)
        }
        
        for scaler in self.mock_processor.feature_scalers.values():
            scaler.transform.return_value = np.random.random((100, 1))
        
        # Make prediction
        predictions = self.predictor.predict(features)
        
        # Verify scaling was applied
        for scaler in self.mock_processor.feature_scalers.values():
            scaler.transform.assert_called()

if __name__ == '__main__':
    unittest.main()