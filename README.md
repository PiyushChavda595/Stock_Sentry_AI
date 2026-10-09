# 📈 Stock-Sentry AI

### Deep Learning for Financial Forecasting

Stock-Sentry AI is a deep learning application designed to forecast stock prices across multiple future time horizons. The project compares SimpleRNN, LSTM, and GRU architectures to analyze historical stock data and generate short-term, mid-term, and long-term price predictions.

The system uses TensorFlow/Keras for model development and Streamlit for the interactive dashboard, with Plotly charts to visualize predicted trends against historical prices.

## 📌 Features

### 📈 Multi-Horizon Stock Price Forecasting

Predicts stock prices across three time horizons:

- **Short-term:** 1 day
- **Mid-term:** 5 days
- **Long-term:** 10 days

### 🧠 Deep Learning Models

Compares three recurrent neural network architectures:

- SimpleRNN
- LSTM
- GRU

### 📊 Technical Indicator Analysis

Uses technical indicators such as RSI and MACD to represent market conditions and momentum.

### 🔍 Time-Series Preprocessing

Uses chronological data splitting, a 60-day sliding window, and MinMax scaling to prepare historical market data for forecasting.

### ⚙️ Hyperparameter Optimization

Uses GridSearchCV to evaluate model configurations involving the number of units, dropout rate, and learning rate.

### 📉 Interactive Visualizations

Uses Plotly charts to compare historical stock prices with predicted trends through the Streamlit dashboard.

### 📂 Historical Data Analysis

The documented workflow uses the `TSLA.csv` dataset for historical Tesla stock analysis.

## 🏗️ System Architecture

```text
       Historical Stock Data
             TSLA.csv
                │
                ▼
       Data Preprocessing
                │
                ▼
       Technical Indicators
          RSI and MACD
                │
                ▼
       Sliding Window
          60 Days
                │
                ▼
       Model Optimization
          GridSearchCV
                │
         ┌──────┼──────┐
         │      │      │
         ▼      ▼      ▼
      SimpleRNN LSTM   GRU
         │      │      │
         └──────┼──────┘
                │
                ▼
        Model Evaluation
                │
                ▼
       Saved Forecasting Model
                │
                ▼
        Streamlit Dashboard
                │
                ▼
        Plotly Visualizations
                │
                ▼
      Forecasts and Market Insights
```

## 🔄 Application Workflow

```text
Historical Stock Data
         ↓
Chronological Sorting
         ↓
Stationarity Testing
         ↓
Feature Engineering
         ↓
MinMax Scaling
         ↓
60-Day Sliding Window
         ↓
Train and Test Split
         ↓
Model Training
         ↓
Hyperparameter Optimization
         ↓
RMSE and MAPE Evaluation
         ↓
Model Serialization
         ↓
Streamlit Inference
         ↓
Interactive Forecast Visualization
```

## 🤖 Deep Learning Models

The project compares three recurrent neural network architectures for stock price forecasting.

### 1. SimpleRNN

**Model Type:** Recurrent Neural Network

SimpleRNN serves as the baseline model. The report notes that it struggled to retain information over long sequences and did not adequately capture the 2020 Tesla stock breakout.

### 2. LSTM

**Full Name:** Long Short-Term Memory

LSTM uses input, output, and forget gates to retain relevant information across longer sequences.

The project selected LSTM as the best-performing architecture based on its reported RMSE and its ability to capture nonlinear stock price trends.

### 3. GRU

**Full Name:** Gated Recurrent Unit

GRU is a recurrent architecture with fewer parameters than a comparable LSTM configuration.

The report describes its forecasting performance as nearly identical to LSTM, with lower computational cost.

## 📊 Data Processing and Feature Engineering

The project uses historical financial time-series data and prepares it for deep learning.

### Chronological Data Handling

The data is sorted by date to prevent look-ahead bias, where future information could unintentionally influence past predictions.

### Stationarity Testing

The Augmented Dickey-Fuller (ADF) test is used to examine the stationarity of the historical time series.

### Technical Indicators

The project generates market context features, including:

- **RSI (Relative Strength Index):** Used to analyze overbought and oversold conditions.
- **MACD (Moving Average Convergence Divergence):** Used to track price momentum.

### Normalization

MinMax scaling transforms numerical features into the range of 0 to 1 to support model training.

### Sliding Window

The project uses a 60-day sliding window to create sequences from historical data.

Each sequence provides historical context for forecasting stock prices at the 1-day, 5-day, and 10-day horizons.

## ⚙️ Hyperparameter Optimization

The project uses GridSearchCV with Keras models to evaluate different model configurations.

The reported search parameters include:

| Parameter | Values |
|---|---|
| Units | 32, 50, 64 |
| Dropout Rate | 0.1, 0.2 |
| Learning Rate | 0.001, 0.0005 |

Dropout is used to help reduce overfitting, while the search process identifies suitable model configurations.

## 📊 Model Performance

The project evaluates its models using RMSE and MAPE on an unseen test set covering 2019–2020.

| Model | Reported Result |
|---|---|
| SimpleRNN | Higher variance and error; failed to capture the 2020 breakout |
| LSTM | Lowest RMSE; captured nonlinear Tesla price growth |
| GRU | Nearly identical performance to LSTM with lower computational cost |

The report does not provide exact numerical RMSE or MAPE values for the final models, so no specific scores are included here.

## 📉 Key Findings

### Trend Capture

The report identifies LSTM as more effective than the baseline SimpleRNN at distinguishing temporary market fluctuations from sustained trend changes.

### Trading Volume and Volatility

The project's statistical analysis found a relationship between higher trading volume and increased price volatility.

### Model Efficiency

GRU achieved performance comparable to LSTM while requiring less computational effort, making it an alternative for environments with limited resources.

## 🖥️ Streamlit Dashboard

The project uses Streamlit to provide an interactive interface for stock price forecasting.

According to the documented workflow, the dashboard:

- Loads the trained forecasting model.
- Processes historical market data.
- Generates predictions for multiple future horizons.
- Displays predicted trends alongside historical prices.
- Uses Plotly charts for interactive visualization.

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Baseline Model | SimpleRNN |
| Sequence Models | LSTM, GRU |
| Hyperparameter Optimization | GridSearchCV |
| Feature Engineering | RSI, MACD |
| Stationarity Testing | Augmented Dickey-Fuller (ADF) |
| Data Scaling | MinMaxScaler |
| Data Source | Historical `TSLA.csv` |
| Visualization | Plotly |
| Web Application | Streamlit |

## 🚀 Setup and Deployment

The project report documents the modeling pipeline and Streamlit deployment but does not provide a complete dependency list, exact installation commands, or the full repository structure.

For local setup:

1. Obtain the project source code.
2. Review the repository's dependency files and install the required packages.
3. Ensure the historical stock dataset is available.
4. Verify that the trained model artifacts are present or run the training workflow.
5. Launch the Streamlit application using the entry point defined in the project.

The report confirms that the model is serialized and loaded into the Streamlit application for inference, but does not specify the exact artifact filenames or launch command.

## 💼 Business Value

Stock-Sentry AI demonstrates how recurrent neural networks can be used to analyze historical stock data and generate multi-horizon forecasts.

The system provides a framework for:

- Exploring short-term and longer-term price trends.
- Comparing different sequence-model architectures.
- Visualizing predicted prices alongside historical market movements.
- Investigating relationships between technical indicators and market behavior.

The forecasts are model estimates and should not be interpreted as guaranteed future prices or definitive trading recommendations.

## ⚠️ Limitations and Considerations

### Market Volatility

Stock prices are affected by market events, economic conditions, investor behavior, and other external factors that historical data alone may not capture.

### Historical Dataset

The documented workflow uses historical Tesla stock data. The reported results do not establish equivalent performance across other stocks or market conditions.

### Forecast Uncertainty

The report does not provide exact numerical RMSE and MAPE results, confidence intervals, or a verified live market-data integration.

### Buy/Sell Signals

The project description identifies Buy/Sell signals as an objective, but the detailed workflow primarily documents multi-horizon price forecasting and visualization. A complete trading strategy or validated trading-performance evaluation is not documented.

### Financial Risk

Model predictions should be used for analysis and research rather than as a substitute for independent financial assessment.

## 🔮 Future Scope

The project report identifies several potential improvements.

### 📰 Sentiment Analysis

Integrate natural language processing to analyze news headlines and social media posts and investigate how market sentiment relates to price movements.

### 🌍 Macroeconomic Integration

Incorporate external economic indicators such as interest rates and inflation data to improve forecasting context.

### 📊 Portfolio Optimization

Extend the system to investigate asset allocation strategies using forecasted prices and volatility estimates.

## 👨‍💻 Project Information

- **Project:** Stock-Sentry AI
- **Full Name:** Deep Learning for Financial Forecasting
- **Author:** Piyush Chavda
- **Report Date:** January 12, 2026
- **Purpose:** Multi-horizon stock price forecasting using recurrent neural networks
