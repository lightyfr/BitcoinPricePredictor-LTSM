# bitcoin_predictor/utils/metrics_viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class MetricsVisualizer:
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_plot(self, plot_type: str):
        filename = f"{plot_type}_{self.timestamp}.png"
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_metrics_to_csv(self, metrics: Dict[str, float]):
        filename = f"metrics_{self.timestamp}.csv"
        pd.DataFrame([metrics]).to_csv(self.save_dir / filename, index=False)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of performance metrics"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'Explained_Variance': explained_variance_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Accuracy': 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Add directional accuracy
        direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        metrics['Directional_Accuracy'] = direction_correct / (len(y_true) - 1) * 100
        
        return metrics

    def plot_metrics_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             history: Dict[str, Any]):
        """Create comprehensive metrics dashboard"""
        # Ensure 1D arrays for calculations
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        
        metrics = self.calculate_metrics(y_true, y_pred)
        self._save_metrics_to_csv(metrics)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Training History
        plt.subplot(2, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 2. Error Distribution
        plt.subplot(2, 3, 2)
        errors = y_true - y_pred
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        
        # 3. Actual vs Predicted Scatter
        plt.subplot(2, 3, 3)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        
        # 4. Metrics Table
        plt.subplot(2, 3, 4)
        plt.axis('off')
        cell_text = [[f"{k}: {v:.4f}"] for k, v in metrics.items()]
        plt.table(cellText=cell_text, loc='center', cellLoc='left')
        plt.title('Performance Metrics')
        
        # 5. Rolling Window Metrics
        plt.subplot(2, 3, 5)
        window = 20
        rolling_mae = pd.Series(np.abs(errors)).rolling(window).mean()
        rolling_mape = pd.Series(np.abs(errors/y_true)*100).rolling(window).mean()
        plt.plot(rolling_mae, label='Rolling MAE')
        plt.plot(rolling_mape, label='Rolling MAPE')
        plt.title(f'Rolling Metrics (window={window})')
        plt.legend()
        
        # 6. Prediction Intervals
        plt.subplot(2, 3, 6)
        std_dev = np.std(errors)
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.fill_between(range(len(y_pred)), 
                        y_pred - 2*std_dev, 
                        y_pred + 2*std_dev, 
                        alpha=0.2, 
                        label='95% Confidence')
        plt.title('Predictions with Confidence Intervals')
        plt.legend()
        
        plt.tight_layout()
        self._save_plot('metrics_dashboard')

    def generate_metrics_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              history: Dict[str, Any]) -> str:
        """Generate text report of model performance"""
        metrics = self.calculate_metrics(y_true, y_pred)
        
        report = [
            "Model Performance Report",
            "=====================",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\nKey Metrics:",
            "------------"
        ]
        
        for metric, value in metrics.items():
            report.append(f"{metric}: {value:.4f}")
            
        report.extend([
            "\nTraining Summary:",
            "-----------------",
            f"Final training loss: {history['loss'][-1]:.4f}",
            f"Final validation loss: {history['val_loss'][-1]:.4f}",
            f"Best validation loss: {min(history['val_loss']):.4f}",
            f"Number of epochs: {len(history['loss'])}"
        ])
        
        report_path = self.save_dir / f"metrics_report_{self.timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        return '\n'.join(report)