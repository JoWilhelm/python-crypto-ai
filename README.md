# Python Crypto AI

This project applies LSTM RNNs to cryptocurrency trading. It covers the full pipeline—from building and classifying historical datasets, through training NN models, to backtesting trading strategies and executing live trades.

## Project Overview

- **Collecting & Preparing Data:** Fetching, labeling and preprocessing historical cryptocurrency data form an exchange API.
- **Training Models:** Building and training LSTMs to classify market conditions and predict trends.
- **Backtesting Strategies:** Evaluating predictive models by simulating trading scenarios.
- **Live Trading:** Deploying strategies in a live market environment on the exchange.

## Repository Structure and File Descriptions




### Detailed File Explanations

- `historicalData.py`  
  Fetches raw market data from the Poloniex exchange API. Then formats the data to prepare it for classification and model training. Example outputs are stored in `HistoricalData.csv`.

- `classification.py`  
  Implements different algorithms for classifying market data. It assigns labels (e.g., bullish, bearish, neutral) that are used for training the predictive models. Example outputs are stored in `HistoricalDataClassified.csv`.

- `classificationVisualization.py`
  Offers visualization tools to analyze and adjust the ground-truth classification. It generates plots and charts that illustrate how the data labeled.

- `modelTrainPreClassified.ipynb`
  A Jupyter Notebook for the process of training the LSTM using the pre-classified data. It covers preprocessing, model architecture, and training.
  
- `r20t0-18.h5`
  A saved model file (in HDF5 format) of a trained network. This model can be used in both backtesting and live trading scenarios.

- `backtestingModel.py`
  Contains routines to backtest the performance of the trained model on new unseen data. This script simulates historical trading based on the model’s raw predictions to evaluate its effectiveness. The model's outputs are stored in `modelOutput_r20t0-18.csv`. The file is useful for performance analysis and refining trading strategies that are built on top of the raw output.

- `backtestingStrategy.py`  
  Provides an interface to simulate trading strategies using the raw predictions from the model. It tests buy/sell decision rules against historical (unseen) data to assess potential profitability.

- `traderLive.py`
  Demonstrates the integration of the trained model with real-time data to execute live trades. It connects to Poloniex exchange to automate trading decisions based on model outputs.


## Conclusion

This repository contains an end-to-end project, demonstrating the practical application of machine learning in financial markets. Its modular design—from data acquisition, -preparation and -classification over NN model design and -training to strategy backtesting and live trading—makes it an excellent starting point for your own experimentations in the field.

