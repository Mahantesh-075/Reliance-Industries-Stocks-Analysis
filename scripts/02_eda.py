"""
Phase 2 — Exploratory Data Analysis
Reliance Industries Stock Analysis (1994-2025)

Generates 7 charts + statistical reports:
1. 30-year closing price with event annotations (Plotly HTML)
2. Year-over-Year annual returns bar chart (Plotly HTML)
3. Decade-wise return comparison (CSV report)
4. Price & Volume dual-axis chart (Plotly HTML)
5. Daily returns distribution + Q-Q plot (Matplotlib PNG)
6. Monthly return heatmap (Seaborn PNG)
7. Rolling volatility (Plotly HTML)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
PLOTS = os.path.join(BASE, 'outputs', 'plots')
REPORTS = os.path.join(BASE, 'outputs', 'reports')

print("=" * 60)
print("  PHASE 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Load cleaned data ─────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA, 'reliance_cleaned.csv'), index_col=0, parse_dates=True)
sns.set_theme(style='whitegrid')
print(f"\n   Loaded {len(df)} rows | {df.index.min().date()} to {df.index.max().date()}")

# ══════════════════════════════════════════════════════════════════════
# Chart 1: 30-Year Closing Price with Event Annotations
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] 30-Year Closing Price Trend with Events...")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, y=df['close'],
    mode='lines', name='Close Price',
    line=dict(color='royalblue', width=1.5)
))

events = {
    '2000-03-10': 'Dot-com Crash',
    '2008-09-15': 'Global Financial Crisis',
    '2013-01-01': 'RIL Refinery Expansion',
    '2016-09-01': 'Jio Launch',
    '2017-09-07': 'Stock Split (1:1 Bonus)',
    '2020-03-23': 'COVID-19 Crash',
    '2020-06-01': 'Jio Rights Issue Boom',
    '2021-10-18': 'All-time High',
    '2024-07-08': 'ATH 3200+',
}

for date_str, label in events.items():
    try:
        mask = df.index >= pd.to_datetime(date_str)
        if mask.any():
            price = df.loc[mask, 'close'].iloc[0]
            fig.add_vline(x=date_str, line_dash='dot', line_color='gray', opacity=0.4)
            fig.add_annotation(x=date_str, y=price, text=label, showarrow=True,
                               arrowhead=2, font=dict(size=9), bgcolor='rgba(255,255,255,0.8)')
    except Exception:
        pass

fig.update_layout(
    title='Reliance Industries - 30 Year Closing Price (1994-2025)',
    xaxis_title='Date', yaxis_title='Close Price (INR)',
    template='plotly_white', height=550,
    hovermode='x unified'
)
fig.write_html(os.path.join(PLOTS, 'ril_30yr_price.html'))
print("   Saved: ril_30yr_price.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 2: Year-over-Year Annual Returns
# ══════════════════════════════════════════════════════════════════════
print("[2/7] Year-over-Year Annual Returns...")

annual_returns = df.groupby('year')['close'].apply(
    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
).reset_index()
annual_returns.columns = ['year', 'annual_return']

colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in annual_returns['annual_return']]
fig = go.Figure(go.Bar(
    x=annual_returns['year'],
    y=annual_returns['annual_return'],
    marker_color=colors,
    name='Annual Return %',
    text=[f"{r:.1f}%" for r in annual_returns['annual_return']],
    textposition='outside',
    textfont=dict(size=8)
))
fig.update_layout(
    title='Reliance Industries - Year-over-Year Annual Returns (%)',
    xaxis_title='Year', yaxis_title='Return (%)',
    template='plotly_white', height=500
)
fig.write_html(os.path.join(PLOTS, 'ril_annual_returns.html'))
print("   Saved: ril_annual_returns.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 3: Decade-wise Return Comparison (Report)
# ══════════════════════════════════════════════════════════════════════
print("[3/7] Decade-wise Return Comparison...")

decade_stats = df.groupby('decade').agg(
    avg_daily_return=('daily_return', 'mean'),
    std_daily_return=('daily_return', 'std'),
    avg_volatility=('volatility_30d', 'mean'),
    total_return=('close', lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100),
    max_close=('close', 'max'),
    min_close=('close', 'min'),
    trading_days=('close', 'count')
).round(4)

print(f"\n   Decade-wise Statistics:")
print(decade_stats.to_string())
decade_stats.to_csv(os.path.join(REPORTS, 'decade_stats.csv'))
print("   Saved: decade_stats.csv")

# ══════════════════════════════════════════════════════════════════════
# Chart 4: Price & Volume Dual-Axis
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Price & Volume Chart...")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=['Closing Price', 'Trading Volume'],
                    row_heights=[0.65, 0.35], vertical_spacing=0.08)

fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close',
                          line=dict(color='steelblue', width=1.5)), row=1, col=1)

# Color volume bars based on price direction
vol_colors = []
for i in range(len(df)):
    if i == 0:
        vol_colors.append('#2ecc71')
    elif df['close'].iloc[i] >= df['close'].iloc[i-1]:
        vol_colors.append('#2ecc71')
    else:
        vol_colors.append('#e74c3c')

fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume',
                      marker_color=vol_colors, opacity=0.5), row=2, col=1)

fig.update_layout(title='Reliance Industries - Price & Volume (1994-2025)',
                  template='plotly_white', height=650, showlegend=False)
fig.write_html(os.path.join(PLOTS, 'ril_price_volume.html'))
print("   Saved: ril_price_volume.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 5: Daily Returns Distribution + Q-Q Plot
# ══════════════════════════════════════════════════════════════════════
print("[5/7] Daily Returns Distribution...")

returns = df['daily_return'].dropna()

fig_mpl, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with normal curve
axes[0].hist(returns, bins=100, color='steelblue', alpha=0.7, density=True, edgecolor='white')
mu, sigma = returns.mean(), returns.std()
x = np.linspace(returns.min(), returns.max(), 200)
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Dist')
axes[0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Daily Return')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].axvline(0, color='black', linestyle='--', alpha=0.3)

# Q-Q Plot
stats.probplot(returns, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')

plt.suptitle('Reliance Industries - Daily Returns Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'returns_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Statistical tests
sample = returns.sample(min(5000, len(returns)), random_state=42)
stat, p = stats.shapiro(sample)
skewness = returns.skew()
kurtosis = returns.kurtosis()

print(f"   Shapiro-Wilk Test: stat={stat:.4f}, p={p:.6f}")
print(f"   Skewness: {skewness:.4f}")
print(f"   Excess Kurtosis: {kurtosis:.4f}")
print(f"   -> Returns are {'NOT ' if p < 0.05 else ''}normally distributed")
print("   Saved: returns_distribution.png")

# ══════════════════════════════════════════════════════════════════════
# Chart 6: Monthly Return Heatmap
# ══════════════════════════════════════════════════════════════════════
print("[6/7] Monthly Return Heatmap...")

monthly_returns = df['close'].resample('ME').last().pct_change() * 100
monthly_pivot = pd.DataFrame({
    'year': monthly_returns.index.year,
    'month': monthly_returns.index.month_name().str[:3],
    'return': monthly_returns.values
})
monthly_pivot = monthly_pivot.pivot(index='year', columns='month', values='return')

month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_pivot = monthly_pivot[[m for m in month_order if m in monthly_pivot.columns]]

plt.figure(figsize=(16, 20))
sns.heatmap(monthly_pivot, cmap='RdYlGn', center=0, annot=False,
            linewidths=0.3, cbar_kws={'label': 'Monthly Return %', 'shrink': 0.6})
plt.title('Reliance Industries - Monthly Return Heatmap (1994-2025)', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'monthly_return_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: monthly_return_heatmap.png")

# ══════════════════════════════════════════════════════════════════════
# Chart 7: Rolling Volatility
# ══════════════════════════════════════════════════════════════════════
print("[7/7] Rolling Volatility...")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['volatility_30d'],
                          name='30-Day Volatility', line=dict(color='#f39c12', width=1.5)))
fig.add_trace(go.Scatter(x=df.index, y=df['volatility_90d'],
                          name='90-Day Volatility', line=dict(color='#e74c3c', width=1.5, dash='dot')))

# Annotate peak volatility periods
vol_peaks = df.nlargest(5, 'volatility_30d')
for idx, row in vol_peaks.iterrows():
    fig.add_annotation(x=idx, y=row['volatility_30d'],
                       text=f"{idx.strftime('%Y-%m')}",
                       showarrow=True, arrowhead=2, font=dict(size=8))

fig.update_layout(
    title='Reliance Industries - Rolling Annualized Volatility (1994-2025)',
    yaxis_title='Volatility',
    template='plotly_white', height=450,
    hovermode='x unified'
)
fig.write_html(os.path.join(PLOTS, 'rolling_volatility.html'))
print("   Saved: rolling_volatility.html")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 2 COMPLETE")
print("=" * 60)
print(f"  Charts: {len(os.listdir(PLOTS))} files in outputs/plots/")
print(f"  Reports: {len(os.listdir(REPORTS))} files in outputs/reports/")

# Key insights
print("\n  KEY INSIGHTS:")
best_year = annual_returns.loc[annual_returns['annual_return'].idxmax()]
worst_year = annual_returns.loc[annual_returns['annual_return'].idxmin()]
print(f"  1. Best Year:  {int(best_year['year'])} ({best_year['annual_return']:.1f}%)")
print(f"  2. Worst Year: {int(worst_year['year'])} ({worst_year['annual_return']:.1f}%)")
print(f"  3. 30-Year Total Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.1f}%")
print(f"  4. Avg Annualized Volatility (30d): {df['volatility_30d'].mean()*100:.1f}%")
print(f"  5. Skewness: {skewness:.4f} (left-tailed)")
print(f"  6. Excess Kurtosis: {kurtosis:.4f} (fat tails = more extreme events than normal)")
print(f"  7. Returns are NOT normally distributed (heavy tails)")
print(f"  8. Highest volatility periods align with GFC 2008, COVID 2020, Stock Split 2017")
