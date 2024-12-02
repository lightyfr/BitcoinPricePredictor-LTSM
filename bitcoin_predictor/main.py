# bitcoin_predictor/main.py
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bitcoin_predictor.src.data_processor import DataProcessor
from bitcoin_predictor.src.model import BitcoinPriceModel
from bitcoin_predictor.src.training import ModelTrainer
from bitcoin_predictor.utils.config import ModelConfig, DataConfig
from bitcoin_predictor.utils.visualization import plot_training_history, plot_predictions
from bitcoin_predictor.utils.performance_viz import PerformanceVisualizer
from bitcoin_predictor.utils.metrics_viz import MetricsVisualizer

logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description='Bitcoin Price Predictor')
    parser.add_argument('--symbol', default='BTC-USD', help='Trading symbol')
    parser.add_argument('--period', default='max', help='Data period')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    return parser.parse_args()

def main():
    args = setup_args()
    
    # Initialize components
    data_config = DataConfig(symbol=args.symbol, period=args.period)
    model_config = ModelConfig(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    processor = DataProcessor(sequence_length=args.sequence_length)
    visualizer = PerformanceVisualizer()
    
    try:
        # Fetch and process data
        data = processor.fetch_data(symbol=data_config.symbol, period=data_config.period)
        data = processor.add_features(data)
        
        # Prepare features and target
        features = data[data_config.feature_columns].values
        target = data[data_config.target_column].values
        
        # Scale data
        scaled_features, scaled_target = processor.scale_data(features, target)
        
        # Create sequences
        X, y = processor.prepare_sequences(scaled_features, scaled_target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Initialize and train model
        model = BitcoinPriceModel(model_config)
        trainer = ModelTrainer(model, model_config)
        
        history = trainer.train(X_train, y_train)
        plot_training_history(history)
        visualizer.plot_training_history(history)
        visualizer.plot_model_architecture(model.model)
        
        # Make predictions
        y_pred = trainer.predict(X_test)
        y_pred = processor.target_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test = processor.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Plot results
        plot_predictions(y_test, y_pred, 'Bitcoin Price Prediction (Test Set)')
        visualizer.plot_predictions(y_test, y_pred, data.index[-len(y_test):])
        visualizer.plot_residuals(y_test, y_pred)
        
        # Feature correlations
        feature_df = pd.DataFrame(features, columns=data_config.feature_columns)
        visualizer.plot_feature_correlations(feature_df)
        
        # Save model
        model_path = Path('models/best_model.keras')
        model_path.parent.mkdir(exist_ok=True)
        trainer.save_model(model_path)
        
        # After training and predictions
        metrics_viz = MetricsVisualizer()
        
        # Generate comprehensive metrics
        metrics_viz.plot_metrics_dashboard(y_test, y_pred, history)
        
        # Generate and print report
        report = metrics_viz.generate_metrics_report(y_test, y_pred, history)
        print(report)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()