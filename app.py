"""
Phase 6 — Interactive Streamlit Dashboard
Reliance Industries Stock Analysis (1994-2025)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reliance Industries Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }

    .kpi-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }
    .kpi-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem; }
    .kpi-delta-pos { color: #2ecc71; font-size: 0.85rem; font-weight: 500; }
    .kpi-delta-neg { color: #e74c3c; font-size: 0.85rem; font-weight: 500; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, 'data', 'reliance_cleaned.csv'),
                     index_col=0, parse_dates=True)
    return df

@st.cache_data
def load_forecasts():
    base = os.path.dirname(os.path.abspath(__file__))
    reports = os.path.join(base, 'outputs', 'reports')
    fc30, fc60 = None, None
    try:
        fc30 = pd.read_csv(os.path.join(reports, 'arima_30day_forecast.csv'), parse_dates=['date'])
        fc60 = pd.read_csv(os.path.join(reports, 'arima_60day_forecast.csv'), parse_dates=['date'])
    except FileNotFoundError:
        pass
    return fc30, fc60

df = load_data()
fc30, fc60 = load_forecasts()

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 Reliance Industries — Stock Analysis Dashboard</h1>
    <p>Comprehensive time-series analysis covering 30+ years of market data (1994–2025)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ──────────────────────────────────────────────────
st.sidebar.markdown("### 🎛️ Controls")
date_range = st.sidebar.slider(
    "Date Range",
    min_value=df.index.min().to_pydatetime(),
    max_value=df.index.max().to_pydatetime(),
    value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
    format="YYYY-MM-DD"
)

chart_type = st.sidebar.selectbox("Price Chart Style", ["Candlestick", "Line", "Area"])

st.sidebar.markdown("### 📊 Overlays")
show_sma50 = st.sidebar.checkbox("SMA-50", value=True)
show_sma200 = st.sidebar.checkbox("SMA-200", value=True)
show_ema20 = st.sidebar.checkbox("EMA-20", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
show_volume = st.sidebar.checkbox("Volume Bars", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Forecasts")
show_forecast = st.sidebar.selectbox("Forecast Horizon", ["None", "30-Day", "60-Day"])

# Filter data by date range
mask = (df.index >= pd.Timestamp(date_range[0])) & (df.index <= pd.Timestamp(date_range[1]))
dff = df[mask]

# ── KPI Cards ──────────────────────────────────────────────────────────
start_price = dff['close'].iloc[0]
end_price = dff['close'].iloc[-1]
total_return = ((end_price - start_price) / start_price) * 100
ath = dff['close'].max()
ath_date = dff['close'].idxmax().strftime('%Y-%m-%d')
avg_vol = dff['volume'].mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">₹{start_price:,.2f}</div>
        <div class="kpi-label">Start Price</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">₹{end_price:,.2f}</div>
        <div class="kpi-label">Latest Price</div>
    </div>""", unsafe_allow_html=True)

with col3:
    delta_class = "kpi-delta-pos" if total_return > 0 else "kpi-delta-neg"
    sign = "+" if total_return > 0 else ""
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value {delta_class}">{sign}{total_return:.1f}%</div>
        <div class="kpi-label">Total Return</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">₹{ath:,.2f}</div>
        <div class="kpi-label">All-Time High ({ath_date})</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-value">{avg_vol/1e6:.1f}M</div>
        <div class="kpi-label">Avg Daily Volume</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════
# Main Price Chart
# ══════════════════════════════════════════════════════════════════════
n_rows = 2 if show_volume else 1
heights = [0.75, 0.25] if show_volume else [1.0]
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                    row_heights=heights, vertical_spacing=0.06)

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=dff.index, open=dff['open'], high=dff['high'],
        low=dff['low'], close=dff['close'], name='OHLC',
        increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
    ), row=1, col=1)
elif chart_type == "Line":
    fig.add_trace(go.Scatter(x=dff.index, y=dff['close'], name='Close',
                              line=dict(color='#3498db', width=1.5)), row=1, col=1)
else:  # Area
    fig.add_trace(go.Scatter(x=dff.index, y=dff['close'], name='Close',
                              fill='tozeroy', fillcolor='rgba(52,152,219,0.15)',
                              line=dict(color='#3498db', width=1.5)), row=1, col=1)

# Overlays
if show_sma50 and 'SMA_50' in dff.columns:
    fig.add_trace(go.Scatter(x=dff.index, y=dff['SMA_50'], name='SMA-50',
                              line=dict(color='#f39c12', width=1, dash='dash')), row=1, col=1)
if show_sma200 and 'SMA_200' in dff.columns:
    fig.add_trace(go.Scatter(x=dff.index, y=dff['SMA_200'], name='SMA-200',
                              line=dict(color='#8e44ad', width=1, dash='dash')), row=1, col=1)
if show_ema20 and 'EMA_20' in dff.columns:
    fig.add_trace(go.Scatter(x=dff.index, y=dff['EMA_20'], name='EMA-20',
                              line=dict(color='#e74c3c', width=1, dash='dot')), row=1, col=1)

if show_bb and 'BB_upper' in dff.columns:
    fig.add_trace(go.Scatter(x=dff.index, y=dff['BB_upper'], name='BB Upper',
                              line=dict(color='rgba(150,150,150,0.5)', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dff.index, y=dff['BB_lower'], name='BB Lower',
                              line=dict(color='rgba(150,150,150,0.5)', dash='dot'),
                              fill='tonexty', fillcolor='rgba(52,152,219,0.06)'), row=1, col=1)

if show_volume:
    vol_colors = ['#2ecc71' if dff['close'].iloc[i] >= dff['close'].iloc[max(0,i-1)]
                  else '#e74c3c' for i in range(len(dff))]
    fig.add_trace(go.Bar(x=dff.index, y=dff['volume'], name='Volume',
                          marker_color=vol_colors, opacity=0.5), row=2, col=1)

fig.update_layout(
    title=f'Reliance Industries — Price Chart ({date_range[0].strftime("%b %Y")} to {date_range[1].strftime("%b %Y")})',
    template='plotly_white', height=550, xaxis_rangeslider_visible=False,
    hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# RSI Panel
# ══════════════════════════════════════════════════════════════════════
st.markdown("### RSI-14 Momentum Indicator")

if 'RSI_14' in dff.columns:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=dff.index, y=dff['RSI_14'], name='RSI-14',
                                  line=dict(color='#8e44ad', width=1.5)))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(231,76,60,0.1)', line_width=0)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor='rgba(46,204,113,0.1)', line_width=0)
    fig_rsi.add_hline(y=70, line_dash='dash', line_color='#e74c3c',
                      annotation_text='Overbought')
    fig_rsi.add_hline(y=30, line_dash='dash', line_color='#2ecc71',
                      annotation_text='Oversold')
    fig_rsi.update_layout(template='plotly_white', height=300, yaxis=dict(range=[0, 100]),
                          hovermode='x unified')
    st.plotly_chart(fig_rsi, use_container_width=True)
else:
    st.info("RSI data not available for this date range.")

# ══════════════════════════════════════════════════════════════════════
# Forecast Panel
# ══════════════════════════════════════════════════════════════════════
if show_forecast != "None":
    st.markdown(f"### 🔮 {show_forecast} ARIMA Forecast")
    fc_data = fc30 if show_forecast == "30-Day" else fc60

    if fc_data is not None:
        fig_fc = go.Figure()
        last_90 = dff['close'][-90:]
        fig_fc.add_trace(go.Scatter(x=last_90.index, y=last_90.values, name='Historical',
                                     line=dict(color='steelblue', width=1.5)))
        fig_fc.add_trace(go.Scatter(x=fc_data['date'], y=fc_data['forecast'],
                                     name='Forecast', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig_fc.add_trace(go.Scatter(
            x=list(fc_data['date']) + list(fc_data['date'][::-1]),
            y=list(fc_data['upper_95']) + list(fc_data['lower_95'][::-1]),
            fill='toself', fillcolor='rgba(231,76,60,0.1)',
            line=dict(width=0), name='95% CI'
        ))
        fig_fc.update_layout(template='plotly_white', height=400, hovermode='x unified',
                             yaxis_title='Close Price (INR)')
        st.plotly_chart(fig_fc, use_container_width=True)

        # Show forecast table
        with st.expander("📋 Forecast Data Table"):
            st.dataframe(fc_data.style.format({
                'forecast': '₹{:.2f}',
                'lower_95': '₹{:.2f}',
                'upper_95': '₹{:.2f}'
            }), use_container_width=True)
    else:
        st.warning("Forecast data not found. Please run Phase 5 (05_arima_forecasting.py) first.")

# ══════════════════════════════════════════════════════════════════════
# Additional Analysis Tabs
# ══════════════════════════════════════════════════════════════════════
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Annual Returns", "📈 Volatility", "📋 Data Explorer"])

with tab1:
    annual = dff.groupby('year')['close'].apply(lambda x: (x.iloc[-1]-x.iloc[0])/x.iloc[0]*100).reset_index()
    annual.columns = ['Year', 'Return (%)']
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in annual['Return (%)']]
    fig_ann = go.Figure(go.Bar(x=annual['Year'], y=annual['Return (%)'],
                                marker_color=colors,
                                text=[f"{r:.1f}%" for r in annual['Return (%)']],
                                textposition='outside', textfont=dict(size=9)))
    fig_ann.update_layout(template='plotly_white', height=400, yaxis_title='Return (%)',
                          title='Year-over-Year Returns')
    st.plotly_chart(fig_ann, use_container_width=True)

with tab2:
    if 'volatility_30d' in dff.columns:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=dff.index, y=dff['volatility_30d'],
                                      name='30-Day', line=dict(color='#f39c12', width=1.5)))
        fig_vol.add_trace(go.Scatter(x=dff.index, y=dff['volatility_90d'],
                                      name='90-Day', line=dict(color='#e74c3c', width=1.5, dash='dot')))
        fig_vol.update_layout(template='plotly_white', height=400,
                              yaxis_title='Annualized Volatility',
                              title='Rolling Volatility')
        st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.dataframe(dff[['open','high','low','close','volume','daily_return']].tail(50).style.format({
        'open': '₹{:.2f}', 'high': '₹{:.2f}', 'low': '₹{:.2f}',
        'close': '₹{:.2f}', 'volume': '{:,.0f}', 'daily_return': '{:.4f}'
    }), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
    <p>Reliance Industries Stock Analysis Dashboard | Data: NSE 1994-2025 | Built with Streamlit + Plotly</p>
</div>
""", unsafe_allow_html=True)
