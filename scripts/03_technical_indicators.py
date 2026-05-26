"""
Phase 3 — Technical Indicators
Reliance Industries Stock Analysis (1994-2025)

Computes and charts:
1. Moving Averages: SMA-50, SMA-100, SMA-200, EMA-20, EMA-50
2. RSI-14 with overbought/oversold bands
3. Bollinger Bands (20-day, +/-2 std dev) with squeeze detection
4. MACD (12, 26, 9) with signal line and histogram
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
PLOTS = os.path.join(BASE, 'outputs', 'plots')
CLEAN_PATH = os.path.join(DATA, 'reliance_cleaned.csv')

print("=" * 60)
print("  PHASE 3 — TECHNICAL INDICATORS")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
print(f"\n   Loaded {len(df)} rows")

# ══════════════════════════════════════════════════════════════════════
# Compute All Technical Indicators
# ══════════════════════════════════════════════════════════════════════
print("\n   Computing technical indicators...")

# --- Moving Averages ---
df['SMA_50']  = df['close'].rolling(50).mean()
df['SMA_100'] = df['close'].rolling(100).mean()
df['SMA_200'] = df['close'].rolling(200).mean()
df['EMA_20']  = df['close'].ewm(span=20, adjust=False).mean()
df['EMA_50']  = df['close'].ewm(span=50, adjust=False).mean()

# --- RSI (14-period) ---
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# --- Bollinger Bands (20-day, +/-2 sigma) ---
df['BB_mid']   = df['close'].rolling(20).mean()
bb_std         = df['close'].rolling(20).std()
df['BB_upper'] = df['BB_mid'] + 2 * bb_std
df['BB_lower'] = df['BB_mid'] - 2 * bb_std
df['BB_width'] = df['BB_upper'] - df['BB_lower']

# Squeeze detection (width below 20th percentile)
bb_pct20 = df['BB_width'].quantile(0.20)
df['BB_squeeze'] = df['BB_width'] < bb_pct20

# --- MACD (12, 26, 9) ---
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD']        = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist']   = df['MACD'] - df['MACD_signal']

print("   All indicators computed: SMA, EMA, RSI, Bollinger, MACD")

# ══════════════════════════════════════════════════════════════════════
# Chart 1: Candlestick + Moving Averages (2023-2025)
# ══════════════════════════════════════════════════════════════════════
print("\n[1/4] Candlestick Chart with Moving Averages...")

df_plot = df['2023-01-01':]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['open'], high=df_plot['high'],
    low=df_plot['low'], close=df_plot['close'],
    name='OHLC', increasing_line_color='#2ecc71',
    decreasing_line_color='#e74c3c'
))

ma_configs = [
    ('SMA_50', '#3498db', 'solid', 1.5),
    ('SMA_200', '#8e44ad', 'solid', 1.5),
    ('EMA_20', '#f39c12', 'dash', 1.2),
]
for col, color, dash, width in ma_configs:
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot[col],
        name=col, line=dict(color=color, width=width, dash=dash)
    ))

fig.update_layout(
    title='Reliance Industries - Candlestick + Moving Averages (2023-2025)',
    xaxis_rangeslider_visible=False,
    template='plotly_white', height=600,
    yaxis_title='Price (INR)',
    hovermode='x unified'
)
fig.write_html(os.path.join(PLOTS, 'candlestick_ma.html'))
print("   Saved: candlestick_ma.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 2: Bollinger Bands
# ══════════════════════════════════════════════════════════════════════
print("[2/4] Bollinger Bands Chart...")

df_plot = df['2022-01-01':]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot['open'], high=df_plot['high'],
    low=df_plot['low'], close=df_plot['close'],
    name='OHLC', increasing_line_color='#2ecc71',
    decreasing_line_color='#e74c3c', showlegend=False
))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_upper'],
                          name='Upper Band', line=dict(color='rgba(150,150,150,0.6)', dash='dot')))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_mid'],
                          name='Middle Band (SMA-20)', line=dict(color='#f39c12', width=1.5)))
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_lower'],
                          name='Lower Band', line=dict(color='rgba(150,150,150,0.6)', dash='dot'),
                          fill='tonexty', fillcolor='rgba(52,152,219,0.08)'))

# Mark squeeze zones
squeeze = df_plot[df_plot['BB_squeeze']]
if len(squeeze) > 0:
    fig.add_trace(go.Scatter(x=squeeze.index, y=squeeze['close'],
                              mode='markers', name='Squeeze',
                              marker=dict(size=4, color='red', symbol='diamond')))

fig.update_layout(
    title='Reliance Industries - Bollinger Bands (2022-2025)',
    xaxis_rangeslider_visible=False,
    template='plotly_white', height=550,
    yaxis_title='Price (INR)'
)
fig.write_html(os.path.join(PLOTS, 'bollinger_bands.html'))
print("   Saved: bollinger_bands.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 3: RSI with Close Price
# ══════════════════════════════════════════════════════════════════════
print("[3/4] RSI Chart...")

df_plot = df['2023-01-01':]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06,
                    subplot_titles=['Close Price', 'RSI (14-period)'])

fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'],
                          name='Close', line=dict(color='steelblue', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'],
                          name='SMA-50', line=dict(color='#8e44ad', width=1, dash='dot')), row=1, col=1)

fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI_14'],
                          name='RSI-14', line=dict(color='#8e44ad', width=1.5)), row=2, col=1)

# Overbought/Oversold bands with fill
fig.add_hrect(y0=70, y1=100, fillcolor='rgba(231,76,60,0.1)', line_width=0, row=2, col=1)
fig.add_hrect(y0=0, y1=30, fillcolor='rgba(46,204,113,0.1)', line_width=0, row=2, col=1)
fig.add_hline(y=70, line_dash='dash', line_color='#e74c3c', row=2, col=1,
              annotation_text='Overbought (70)', annotation_position='top right')
fig.add_hline(y=30, line_dash='dash', line_color='#2ecc71', row=2, col=1,
              annotation_text='Oversold (30)', annotation_position='bottom right')
fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.4, row=2, col=1)

fig.update_yaxes(range=[0, 100], row=2, col=1)
fig.update_layout(
    title='Reliance Industries - RSI-14 Analysis (2023-2025)',
    template='plotly_white', height=650
)
fig.write_html(os.path.join(PLOTS, 'rsi_chart.html'))
print("   Saved: rsi_chart.html")

# ══════════════════════════════════════════════════════════════════════
# Chart 4: MACD
# ══════════════════════════════════════════════════════════════════════
print("[4/4] MACD Chart...")

df_plot = df['2023-01-01':]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.4], vertical_spacing=0.06,
                    subplot_titles=['Close Price', 'MACD (12, 26, 9)'])

fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'],
                          name='Close', line=dict(color='steelblue', width=1.5)), row=1, col=1)

fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'],
                          name='MACD', line=dict(color='#3498db', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'],
                          name='Signal', line=dict(color='#f39c12', width=1.5)), row=2, col=1)

hist_colors = ['#2ecc71' if h >= 0 else '#e74c3c' for h in df_plot['MACD_hist']]
fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'],
                      name='Histogram', marker_color=hist_colors, opacity=0.6), row=2, col=1)

fig.add_hline(y=0, line_dash='solid', line_color='gray', opacity=0.3, row=2, col=1)

fig.update_layout(
    title='Reliance Industries - MACD (12, 26, 9) Analysis (2023-2025)',
    template='plotly_white', height=650
)
fig.write_html(os.path.join(PLOTS, 'macd_chart.html'))
print("   Saved: macd_chart.html")

# ── Save updated dataset with indicators ──────────────────────────────
df.to_csv(CLEAN_PATH)
print(f"\n   Updated reliance_cleaned.csv with {len(df.columns)} columns")

print("\n" + "=" * 60)
print("  PHASE 3 COMPLETE - All 4 technical indicator charts saved")
print("=" * 60)
