# bitcoin_predictor/utils/performance_viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd

class PerformanceVisualizer:
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _save_plot(self, plot_type: str):
        """Save plot with timestamp"""
        filename = f"{plot_type}_{self.timestamp}.png"
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, history: Dict[str, Any]):
        """Plot training metrics history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE plot
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        self._save_plot('training_history')

    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, 
                        dates: pd.DatetimeIndex = None):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        if dates is not None:
            plt.plot(dates, actual, label='Actual Price')
            plt.plot(dates, predicted, label='Predicted Price')
            plt.gcf().autofmt_xdate()  # Rotate date labels
        else:
            plt.plot(actual, label='Actual Price')
            plt.plot(predicted, label='Predicted Price')
            
        plt.title('Bitcoin Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        
        self._save_plot('predictions')

    def plot_feature_correlations(self, features_df: pd.DataFrame):
        """Plot feature correlation heatmap"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = features_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        
        self._save_plot('feature_correlations')

    def plot_residuals(self, actual: np.ndarray, predicted: np.ndarray):
        """Plot prediction residuals"""
        residuals = actual - predicted
        
        plt.figure(figsize=(12, 5))
        
        # Residuals over time
        plt.subplot(1, 2, 1)
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals Over Time')
        plt.xlabel('Sample')
        plt.ylabel('Residual')
        
        # Residual distribution
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=50)
        plt.title('Residual Distribution')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        self._save_plot('residuals')

    def plot_model_architecture(self, model: tf.keras.Model):
        """Plot model architecture"""
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=str(self.save_dir / f'model_architecture_{self.timestamp}.png'),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True
            )
        except Exception as e:
            print(f"Failed to plot model architecture: {e}")