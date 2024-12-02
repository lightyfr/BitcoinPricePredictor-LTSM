# Bitcoin Price Predictor

An advanced deep learning model for predicting Bitcoin prices using LSTM neural networks and technical analysis.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- Real-time Bitcoin price data fetching via `yfinance`
- Advanced technical indicators (MA, RSI, MACD) 
- LSTM neural network with dropout and batch normalization
- Comprehensive visualization suite
- Multiple performance metrics

## 📊 Latest Model Performance

| Metric | Value |
|--------|-------|
| MSE | 107,119,960 |
| RMSE | 10,349.88 |
| MAE | 7,973.74 |
| R² Score | 0.728 |
| Accuracy | 82.03% |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bitcoin-price-predictor.git
cd bitcoin-price-predictor

# Install dependencies
pip install -e .

# Basic Usage
python -m bitcoin_predictor.main

#Advanced Configuration
python -m bitcoin_predictor.main \
    --symbol BTC-USD \
    --period max \
    --sequence-length 60 \
    --batch-size 32 \
    --epochs 100

# Bitcoin Price Predictor

An advanced deep learning model for predicting Bitcoin prices using LSTM neural networks and technical analysis.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- Real-time Bitcoin price data fetching via `yfinance`
- Advanced technical indicators (MA, RSI, MACD) 
- LSTM neural network with dropout and batch normalization
- Comprehensive visualization suite
- Multiple performance metrics

## 📊 Latest Model Performance

| Metric | Value |
|--------|-------|
| MSE | 107,119,960 |
| RMSE | 10,349.88 |
| MAE | 7,973.74 |
| R² Score | 0.728 |
| Accuracy | 82.03% |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bitcoin-price-predictor.git
cd bitcoin-price-predictor

# Install dependencies
pip install -e .
```

## Usage

Run the predictor with default parameters:

```bash
python -m bitcoin_predictor.main
```

Custom parameters:

```bash
python -m bitcoin_predictor.main \
    --symbol BTC-USD \
    --period max \
    --sequence-length 60 \
    --batch-size 32 \
    --epochs 100
```

## Project Structure

```
bitcoin_predictor/
├── src/                  # Core implementation
│   ├── data_processor.py # Data processing
│   ├── model.py          # LSTM architecture
│   ├── prediction.py     # Prediction engine
│   └── training.py       # Training logic
├── utils/                # Utilities
├── tests/                # Unit tests
└── data/                 # Data storage
```

## Development

Running tests:

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

