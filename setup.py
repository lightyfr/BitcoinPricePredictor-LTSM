# setup.py
from setuptools import setup, find_packages

setup(
    name="bitcoin_predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "yfinance",
        "matplotlib",
        "seaborn",
        "python-graphviz"
    ]
)