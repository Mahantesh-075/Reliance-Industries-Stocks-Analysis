# 📈 Reliance Industries — Stock Market Analysis & Forecasting

> End-to-end time-series analysis on 30+ years (1994–2025) of Reliance Industries stock data, covering EDA, Technical Indicators, Stationarity Testing, ARIMA/SARIMA Forecasting, and an interactive Streamlit Dashboard.

---

## 🎯 Project Overview

This project performs a comprehensive analysis of Reliance Industries Limited (NSE: RELIANCE) stock data spanning from **November 1994 to August 2025** — over 8,000 trading days. It covers:

- **Data Cleaning** — Deduplication, validation, feature engineering
- **Exploratory Data Analysis** — 7 charts covering price trends, returns, volatility, seasonality
- **Technical Indicators** — SMA, EMA, RSI, Bollinger Bands, MACD
- **Stationarity Testing** — ADF tests, ACF/PACF, seasonal decomposition
- **ARIMA/SARIMA Forecasting** — Grid search, model evaluation, 30/60-day forecasts
- **Interactive Dashboard** — Streamlit app with real-time chart controls

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualization | plotly, matplotlib, seaborn, mplfinance |
| Statistical Analysis | statsmodels, scipy |
| Machine Learning | scikit-learn |
| Dashboard | streamlit |
| Static Export | kaleido |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset | 8,037 trading days (1994–2025) |
| 30-Year Total Return | ~257% |
| Best Year | 2007 (+127.0%) |
| Worst Year | 2008 (-56.7%) |
| All-Time High | ₹3,220.85 |
| Daily Return Skewness | -4.06 (left-tailed) |
| Excess Kurtosis | 91.3 (extreme fat tails) |
| Best ARIMA Model | ARIMA(2,1,2) |
| Avg 30d Volatility | 33.2% annualized |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Analysis Scripts (in order)
```bash
python scripts/01_data_cleaning.py
python scripts/02_eda.py
python scripts/03_technical_indicators.py
python scripts/04_stationarity_testing.py
python scripts/05_arima_forecasting.py
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
reliance-stock-analysis/
├── data/
│   ├── reliance_raw.csv              # Original dataset
│   └── reliance_cleaned.csv          # Cleaned + engineered features
├── scripts/
│   ├── 01_data_cleaning.py           # Phase 1: Load, clean, validate
│   ├── 02_eda.py                     # Phase 2: 7 EDA charts
│   ├── 03_technical_indicators.py    # Phase 3: SMA, EMA, RSI, BB, MACD
│   ├── 04_stationarity_testing.py    # Phase 4: ADF, ACF/PACF, decomposition
│   └── 05_arima_forecasting.py       # Phase 5: Grid search, forecasts
├── outputs/
│   ├── plots/                        # All generated charts (HTML + PNG)
│   │   ├── ril_30yr_price.html
│   │   ├── ril_annual_returns.html
│   │   ├── ril_price_volume.html
│   │   ├── rolling_volatility.html
│   │   ├── returns_distribution.png
│   │   ├── monthly_return_heatmap.png
│   │   ├── candlestick_ma.html
│   │   ├── bollinger_bands.html
│   │   ├── rsi_chart.html
│   │   ├── macd_chart.html
│   │   ├── acf_pacf.png
│   │   ├── seasonal_decomposition.png
│   │   ├── stationarity_comparison.png
│   │   ├── arima_forecast_vs_actual.html
│   │   ├── arima_30day_forecast.html
│   │   └── arima_60day_forecast.html
│   └── reports/                      # CSV reports & metrics
│       ├── decade_stats.csv
│       ├── stationarity_report.txt
│       ├── arima_grid_search.csv
│       ├── arima_metrics.csv
│       ├── model_comparison.csv
│       ├── arima_30day_forecast.csv
│       └── arima_60day_forecast.csv
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 📈 Analysis Highlights

### Price History (1994–2025)
- Reliance stock has gone from ₹396 (1994) to ₹1,413 (2025), with a peak of ₹3,221
- Major events annotated: Dot-com crash, GFC 2008, Jio launch (2016), COVID crash (2020)
- 2017 stock split (1:1 bonus) created a significant price discontinuity

### Returns Analysis
- Returns are **NOT normally distributed** (Shapiro-Wilk p ≈ 0)
- Heavy negative skewness (-4.06) = more extreme negative returns
- Extreme fat tails (kurtosis = 91.3) = black swan events more frequent than expected

### Technical Signals
- Golden/Death crosses (SMA-50 vs SMA-200) identified
- RSI overbought/oversold zones correlate with price reversals
- Bollinger Band squeezes precede volatility breakouts
- MACD crossovers used for trend confirmation

### Forecasting
- ARIMA(2,1,2) selected via AIC-based grid search
- Short-term (30-day) forecasts with 95% confidence intervals
- Note: Stock prices are inherently unpredictable; ARIMA provides baseline forecasts, not trading advice

---

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**. It does not constitute financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

This project is provided as-is for educational use.
