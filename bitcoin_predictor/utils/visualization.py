# bitcoin_predictor/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def plot_training_history(history: Dict[str, Any], save_path: str = None):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                    title: str = 'Bitcoin Price Prediction',
                    save_path: str = None):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()