# Bitcoin Price Predictor

An advanced deep learning model for predicting Bitcoin prices using LSTM neural networks and technical analysis.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)

## 🚀 Features

- Real-time Bitcoin price data fetching via `yfinance`
- Advanced technical indicators (MA, RSI, MACD) 
- LSTM neural network with dropout and batch normalization
- Comprehensive visualization suite
- Multiple performance metrics

## 📊 Latest Model Performance

| Metric | Value |
|--------|-------|
| MSE | 51,790,828 |
| RMSE | 7,334 |
| MAE | 4,434 |
| R² Score | 0.863 |
| Accuracy | 91.67% |

## 🛠️ Installation


# Clone repository
```bash
git clone https://github.com/yourusername/bitcoin-price-predictor.git
```
```bash
cd bitcoin-price-predictor
```

# Install dependencies
```bash
pip install -e .
```

# Basic Usage
```bash
python -m bitcoin_predictor.main
```

# Advanced Configuration
```bash
python -m bitcoin_predictor.main \
    --symbol BTC-USD \
    --period max \
    --sequence-length 60 \
    --batch-size 32 \
    --epochs 100
```
# Running tests:

```bash
python -m unittest discover bitcoin_predictor/tests
```

## Visualizations

The model generates several visualizations in the `visualizations` directory:
- Training history plots
- Prediction vs Actual comparisons
- Feature correlation heatmaps
- Error distribution analysis
- Rolling metrics plots

## Dependencies

- numpy
- pandas
- scikit-learn
- tensorflow
- yfinance
- matplotlib
- seaborn
- python-graphviz

## Logging

The project uses Python's logging framework. Logs are stored in the `logs` directory with timestamped filenames.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under AGPL-3.0


