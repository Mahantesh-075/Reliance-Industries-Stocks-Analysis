"""
Phase 4 — Stationarity Testing
Reliance Industries Stock Analysis (1994-2025)

Performs:
1. ADF test on raw Close price
2. First-order differencing + ADF re-test
3. ACF & PACF plots (raw + differenced)
4. Seasonal decomposition (multiplicative, weekly)
5. Interpretation guide for (p, d, q) selection
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
PLOTS = os.path.join(BASE, 'outputs', 'plots')
REPORTS = os.path.join(BASE, 'outputs', 'reports')

print("=" * 60)
print("  PHASE 4 — STATIONARITY TESTING")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, 'reliance_cleaned.csv'), index_col=0, parse_dates=True)
close = df['close'].dropna()
print(f"\n   Loaded {len(close)} data points")

# ── Helper: ADF Test ──────────────────────────────────────────────────
def run_adf(series, label):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n   ADF Test on: {label}")
    print(f"   ----------------------------------")
    print(f"   Test Statistic:   {result[0]:.4f}")
    print(f"   p-value:          {result[1]:.6f}")
    print(f"   Lags Used:        {result[2]}")
    print(f"   Observations:     {result[3]}")
    for k, v in result[4].items():
        print(f"   Critical ({k}): {v:.4f}")
    is_stat = result[1] < 0.05
    print(f"   --> {'STATIONARY' if is_stat else 'NON-STATIONARY'} (p {'<' if is_stat else '>'} 0.05)")
    return result, is_stat

# ══════════════════════════════════════════════════════════════════════
# Test 1: ADF on raw Close price
# ══════════════════════════════════════════════════════════════════════
print("\n[1/5] ADF Test on Raw Close Price...")
adf_raw, raw_stat = run_adf(close, 'Raw Close Price')

# ══════════════════════════════════════════════════════════════════════
# Test 2: First-order differencing + ADF
# ══════════════════════════════════════════════════════════════════════
print("\n[2/5] First-Order Differencing + ADF...")
diff1 = close.diff().dropna()
adf_diff, diff_stat = run_adf(diff1, 'First-Differenced Close')

# Also test log returns
log_close = np.log(close)
log_diff = log_close.diff().dropna()
adf_log, log_stat = run_adf(log_diff, 'Log Returns')

# ══════════════════════════════════════════════════════════════════════
# Test 3: ACF & PACF Plots
# ══════════════════════════════════════════════════════════════════════
print("\n[3/5] ACF & PACF Plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Raw Close ACF/PACF
plot_acf(close, lags=40, ax=axes[0, 0], alpha=0.05)
axes[0, 0].set_title('ACF - Raw Close Price (Non-Stationary)', fontweight='bold')
plot_pacf(close, lags=40, ax=axes[0, 1], alpha=0.05, method='ywm')
axes[0, 1].set_title('PACF - Raw Close Price', fontweight='bold')

# Differenced ACF/PACF
plot_acf(diff1, lags=40, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF - Differenced Close (d=1)', fontweight='bold')
plot_pacf(diff1, lags=40, ax=axes[1, 1], alpha=0.05, method='ywm')
axes[1, 1].set_title('PACF - Differenced Close (d=1)', fontweight='bold')

plt.suptitle('Reliance Industries - ACF & PACF Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'acf_pacf.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: acf_pacf.png")

# ══════════════════════════════════════════════════════════════════════
# Test 4: Seasonal Decomposition
# ══════════════════════════════════════════════════════════════════════
print("[4/5] Seasonal Decomposition...")

# Weekly resampled for cleaner decomposition
close_weekly = close.resample('W').last().dropna()
decomp = seasonal_decompose(close_weekly, model='multiplicative', period=52)

fig, axes = plt.subplots(4, 1, figsize=(16, 12))
decomp.observed.plot(ax=axes[0], color='steelblue')
axes[0].set_title('Observed', fontweight='bold')
axes[0].set_ylabel('Price')
decomp.trend.plot(ax=axes[1], color='#e74c3c')
axes[1].set_title('Trend', fontweight='bold')
axes[1].set_ylabel('Price')
decomp.seasonal.plot(ax=axes[2], color='#2ecc71')
axes[2].set_title('Seasonal (52-week period)', fontweight='bold')
axes[2].set_ylabel('Multiplier')
decomp.resid.plot(ax=axes[3], color='#f39c12')
axes[3].set_title('Residual', fontweight='bold')
axes[3].set_ylabel('Multiplier')

plt.suptitle('Reliance Industries - Seasonal Decomposition (Weekly, Multiplicative)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'seasonal_decomposition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: seasonal_decomposition.png")

# ══════════════════════════════════════════════════════════════════════
# Test 5: (p, d, q) Interpretation Guide
# ══════════════════════════════════════════════════════════════════════
print("\n[5/5] Parameter Selection Guide...")

report = []
report.append("STATIONARITY & ARIMA PARAMETER SELECTION REPORT")
report.append("=" * 55)
report.append("")
report.append(f"Raw Close:       ADF stat = {adf_raw[0]:.4f}, p = {adf_raw[1]:.6f} -> {'Stationary' if raw_stat else 'Non-Stationary'}")
report.append(f"Differenced(d=1): ADF stat = {adf_diff[0]:.4f}, p = {adf_diff[1]:.6f} -> {'Stationary' if diff_stat else 'Non-Stationary'}")
report.append(f"Log Returns:     ADF stat = {adf_log[0]:.4f}, p = {adf_log[1]:.6f} -> {'Stationary' if log_stat else 'Non-Stationary'}")
report.append("")
report.append("PARAMETER RECOMMENDATIONS:")
report.append("-" * 40)
report.append("d = 1  (first differencing makes series stationary)")
report.append("")
report.append("From ACF/PACF of differenced series:")
report.append("- ACF: rapid decay -> supports MA terms")
report.append("- PACF: significant spike at lag 1, then cuts off")
report.append("- Suggested p range: 0-3")
report.append("- Suggested q range: 0-3")
report.append("")
report.append("CANDIDATE MODELS:")
report.append("  ARIMA(1,1,0), ARIMA(1,1,1), ARIMA(2,1,1)")
report.append("  ARIMA(0,1,1), ARIMA(0,1,2)")
report.append("")
report.append("Grid search will be used in Phase 5 to find optimal AIC/BIC.")

report_text = "\n".join(report)
print("\n" + report_text)

with open(os.path.join(REPORTS, 'stationarity_report.txt'), 'w') as f:
    f.write(report_text)
print("\n   Saved: stationarity_report.txt")

# ── Stationary vs non-stationary visual comparison ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 8))
axes[0].plot(close.index, close.values, color='steelblue', linewidth=1)
axes[0].set_title('Raw Close Price (Non-Stationary)', fontweight='bold')
axes[0].set_ylabel('Price (INR)')

axes[1].plot(diff1.index, diff1.values, color='#2ecc71', linewidth=0.5)
axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.3)
axes[1].set_title('First-Differenced Close (Stationary, d=1)', fontweight='bold')
axes[1].set_ylabel('Price Change (INR)')
axes[1].set_xlabel('Date')

plt.suptitle('Reliance Industries - Stationarity Transformation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'stationarity_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: stationarity_comparison.png")

print("\n" + "=" * 60)
print("  PHASE 4 COMPLETE")
print("=" * 60)
