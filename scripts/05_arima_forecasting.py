"""
Phase 5 — ARIMA/SARIMA Forecasting
Reliance Industries Stock Analysis (1994-2025)

Steps:
1. Use last 5 years (2020-2025) for better convergence
2. 80/20 train-test split
3. ARIMA grid search (p=0..3, d=1, q=0..3)
4. Fit best ARIMA model
5. Evaluate: RMSE, MAE, MAPE
6. 30-day & 60-day future forecasts with confidence intervals
7. SARIMA comparison
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os, itertools
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
PLOTS = os.path.join(BASE, 'outputs', 'plots')
REPORTS = os.path.join(BASE, 'outputs', 'reports')

print("=" * 60)
print("  PHASE 5 — ARIMA/SARIMA FORECASTING")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, 'reliance_cleaned.csv'), index_col=0, parse_dates=True)

# Use last 5 years for better ARIMA convergence
df_model = df['2020-01-01':].copy()
close = df_model['close']
print(f"\n   Using {len(close)} data points ({close.index.min().date()} to {close.index.max().date()})")

# ══════════════════════════════════════════════════════════════════════
# Step 1: Train-Test Split (80/20)
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] Train-Test Split...")
split_idx = int(len(close) * 0.80)
train = close[:split_idx]
test = close[split_idx:]
print(f"   Train: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
print(f"   Test:  {len(test)} days ({test.index.min().date()} to {test.index.max().date()})")

# ══════════════════════════════════════════════════════════════════════
# Step 2: ARIMA Grid Search
# ══════════════════════════════════════════════════════════════════════
print("\n[2/7] ARIMA Grid Search (p=0..3, d=1, q=0..3)...")
print("   This may take a few minutes...\n")

p_range = range(0, 4)
d = 1
q_range = range(0, 4)

results = []
for p, q in itertools.product(p_range, q_range):
    try:
        model = ARIMA(train, order=(p, d, q))
        fitted = model.fit()
        results.append({
            'order': f'({p},{d},{q})',
            'p': p, 'd': d, 'q': q,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
        })
        print(f"   ARIMA{p},{d},{q}  AIC={fitted.aic:.2f}  BIC={fitted.bic:.2f}")
    except Exception as e:
        print(f"   ARIMA{p},{d},{q}  FAILED: {str(e)[:50]}")

results_df = pd.DataFrame(results).sort_values('AIC')
results_df.to_csv(os.path.join(REPORTS, 'arima_grid_search.csv'), index=False)

best = results_df.iloc[0]
best_order = (int(best['p']), int(best['d']), int(best['q']))
print(f"\n   Best model by AIC: ARIMA{best['order']} (AIC={best['AIC']:.2f})")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Fit Best ARIMA Model
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3/7] Fitting ARIMA{best_order}...")
model = ARIMA(train, order=best_order)
model_fit = model.fit()
print(model_fit.summary().tables[1])

# ══════════════════════════════════════════════════════════════════════
# Step 4: Evaluate on Test Set
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Evaluating on Test Set...")

# In-sample fitted values
train_pred = model_fit.fittedvalues

# Out-of-sample forecast for test period
forecast_result = model_fit.get_forecast(steps=len(test))
test_pred = forecast_result.predicted_mean
test_ci = forecast_result.conf_int(alpha=0.05)

# Metrics
rmse = np.sqrt(mean_squared_error(test, test_pred))
mae = mean_absolute_error(test, test_pred)
mape = np.mean(np.abs((test.values - test_pred.values) / test.values)) * 100

print(f"   RMSE:  {rmse:.2f}")
print(f"   MAE:   {mae:.2f}")
print(f"   MAPE:  {mape:.2f}%")

# Save metrics
metrics = {
    'Model': f'ARIMA{best_order}',
    'RMSE': round(rmse, 2),
    'MAE': round(mae, 2),
    'MAPE': round(mape, 2),
    'AIC': round(best['AIC'], 2),
    'BIC': round(best['BIC'], 2),
    'Train_Size': len(train),
    'Test_Size': len(test)
}
pd.DataFrame([metrics]).to_csv(os.path.join(REPORTS, 'arima_metrics.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════
# Step 5: Forecast vs Actual Chart
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Forecast vs Actual Chart...")

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Train', line=dict(color='steelblue', width=1.5)))
fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Actual Test', line=dict(color='#2ecc71', width=2)))
fig.add_trace(go.Scatter(x=test.index, y=test_pred.values, name='ARIMA Forecast',
                          line=dict(color='#e74c3c', width=2, dash='dash')))

# Confidence interval
fig.add_trace(go.Scatter(
    x=list(test.index) + list(test.index[::-1]),
    y=list(test_ci.iloc[:, 1]) + list(test_ci.iloc[:, 0][::-1]),
    fill='toself', fillcolor='rgba(231,76,60,0.1)',
    line=dict(width=0), name='95% CI'
))

fig.update_layout(
    title=f'Reliance Industries - ARIMA{best_order} Forecast vs Actual (MAPE={mape:.2f}%)',
    xaxis_title='Date', yaxis_title='Close Price (INR)',
    template='plotly_white', height=550, hovermode='x unified'
)
fig.write_html(os.path.join(PLOTS, 'arima_forecast_vs_actual.html'))
print("   Saved: arima_forecast_vs_actual.html")

# ══════════════════════════════════════════════════════════════════════
# Step 6: 30-Day & 60-Day Future Forecasts
# ══════════════════════════════════════════════════════════════════════
print("[6/7] Future Forecasts (30-day & 60-day)...")

# Refit on full data for future forecast
full_model = ARIMA(close, order=best_order)
full_fit = full_model.fit()

for horizon, color in [(30, '#3498db'), (60, '#8e44ad')]:
    fc = full_fit.get_forecast(steps=horizon)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=0.05)

    # Generate future date index
    future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fc_mean.index = future_dates
    fc_ci.index = future_dates

    fig = go.Figure()
    # Last 6 months of actual
    last_6m = close[-130:]
    fig.add_trace(go.Scatter(x=last_6m.index, y=last_6m.values, name='Historical',
                              line=dict(color='steelblue', width=1.5)))
    fig.add_trace(go.Scatter(x=future_dates, y=fc_mean.values, name=f'{horizon}-Day Forecast',
                              line=dict(color=color, width=2, dash='dash')))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0][::-1]),
        fill='toself', fillcolor=f'rgba(52,152,219,0.12)',
        line=dict(width=0), name='95% CI'
    ))

    fig.update_layout(
        title=f'Reliance Industries - {horizon}-Day ARIMA Forecast',
        xaxis_title='Date', yaxis_title='Close Price (INR)',
        template='plotly_white', height=500, hovermode='x unified'
    )
    fig.write_html(os.path.join(PLOTS, f'arima_{horizon}day_forecast.html'))
    print(f"   Saved: arima_{horizon}day_forecast.html")

    # Save forecast CSV
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': fc_mean.values,
        'lower_95': fc_ci.iloc[:, 0].values,
        'upper_95': fc_ci.iloc[:, 1].values
    })
    forecast_df.to_csv(os.path.join(REPORTS, f'arima_{horizon}day_forecast.csv'), index=False)
    print(f"   Saved: arima_{horizon}day_forecast.csv")

# ══════════════════════════════════════════════════════════════════════
# Step 7: SARIMA Comparison
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] SARIMA Comparison...")
try:
    sarima = SARIMAX(train, order=best_order, seasonal_order=(1, 1, 1, 5),
                     enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False, maxiter=100)

    sarima_fc = sarima_fit.get_forecast(steps=len(test))
    sarima_pred = sarima_fc.predicted_mean

    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_pred))
    sarima_mae = mean_absolute_error(test, sarima_pred)
    sarima_mape = np.mean(np.abs((test.values - sarima_pred.values) / test.values)) * 100

    print(f"   SARIMA RMSE:  {sarima_rmse:.2f}")
    print(f"   SARIMA MAE:   {sarima_mae:.2f}")
    print(f"   SARIMA MAPE:  {sarima_mape:.2f}%")
    print(f"   SARIMA AIC:   {sarima_fit.aic:.2f}")

    # Save comparison
    comparison = pd.DataFrame([
        {'Model': f'ARIMA{best_order}', 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': best['AIC']},
        {'Model': f'SARIMA{best_order}x(1,1,1,5)', 'RMSE': sarima_rmse, 'MAE': sarima_mae,
         'MAPE': sarima_mape, 'AIC': sarima_fit.aic}
    ]).round(2)
    comparison.to_csv(os.path.join(REPORTS, 'model_comparison.csv'), index=False)
    print("\n   Model Comparison:")
    print(comparison.to_string(index=False))
    print("   Saved: model_comparison.csv")

except Exception as e:
    print(f"   SARIMA fitting failed: {e}")
    print("   Proceeding with ARIMA only.")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 5 COMPLETE")
print("=" * 60)
print(f"  Best Model:  ARIMA{best_order}")
print(f"  Test MAPE:   {mape:.2f}%")
print(f"  Forecast files saved to outputs/reports/")
print(f"  Charts saved to outputs/plots/")
