"""
Phase 1 — Data Loading & Cleaning
Reliance Industries Stock Analysis (1994-2025)

This script:
1. Loads the primary dataset (RELIANCE_NSE_1994-2025.csv)
2. Cleans, deduplicates, validates OHLCV data
3. Engineers new features (returns, volatility, decade labels)
4. Merges supplementary RSI & PE data
5. Saves reliance_cleaned.csv
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
RAW_PATH = os.path.join(DATA, 'reliance_raw.csv')
RSI_PE_PATH = os.path.join(os.path.dirname(BASE),
    'reliance stock prices with rsi and pe ratio', 'Reliance_Stocks_RSI_PE.csv')
CLEAN_PATH = os.path.join(DATA, 'reliance_cleaned.csv')

print("=" * 60)
print("  PHASE 1 — DATA LOADING & CLEANING")
print("=" * 60)

# ── Step 1: Load raw data ─────────────────────────────────────────────
print("\n📂 Loading raw dataset...")
df = pd.read_csv(RAW_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   First 3 rows:")
print(df.head(3).to_string())

# ── Step 2: Drop unnamed index column & standardize names ─────────────
print("\n🔧 Standardizing columns...")
# The first column is an unnamed row index
if df.columns[0] == '' or 'Unnamed' in str(df.columns[0]):
    df = df.drop(df.columns[0], axis=1)

col_map = {
    'Date': 'date',
    'Symbol': 'symbol',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'PrevClose': 'prev_close',
    'Volume': 'volume',
    'Turnover': 'turnover',
    'VWAP': 'vwap',
    'Trades': 'trades',
    'Daily_Return_%': 'drop_daily_return',
    'Cumulative_Return_%': 'drop_cum_return',
    'MA_20': 'drop_ma20',
    'MA_50': 'drop_ma50'
}
df.rename(columns=col_map, inplace=True)

# Drop pre-computed columns (we'll recompute)
drop_cols = [c for c in df.columns if c.startswith('drop_')]
df.drop(columns=drop_cols, inplace=True, errors='ignore')
print(f"   Columns after cleanup: {df.columns.tolist()}")

# ── Step 3: Parse dates & deduplicate ─────────────────────────────────
print("\n📅 Parsing dates & deduplicating...")
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
df.sort_values('date', inplace=True)

dupes_before = df.duplicated(subset=['date'], keep='first').sum()
print(f"   Duplicate dates found: {dupes_before}")
df = df.drop_duplicates(subset=['date'], keep='first')
print(f"   Shape after dedup: {df.shape}")

df.set_index('date', inplace=True)

# Drop symbol column (all RELIANCE)
df.drop(columns=['symbol'], inplace=True, errors='ignore')

print(f"   Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"   Total trading days: {len(df)}")

# ── Step 4: Validate OHLCV consistency ────────────────────────────────
print("\n✅ Validating OHLCV consistency...")

# Convert to numeric (some columns may have mixed types)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

invalid_hl = df[df['high'] < df['low']]
invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
zero_vol = df[df['volume'] <= 0]

print(f"   Rows where High < Low: {len(invalid_hl)}")
print(f"   Rows where Close outside High-Low: {len(invalid_close)}")
print(f"   Rows with zero/negative volume: {len(zero_vol)}")

# Fix: drop invalid OHLC rows
df = df[df['high'] >= df['low']]
df = df[(df['close'] <= df['high']) & (df['close'] >= df['low'])]

# Replace zero volume with NaN and forward-fill
df.loc[df['volume'] <= 0, 'volume'] = np.nan

print(f"   Clean dataset shape: {df.shape}")

# ── Step 5: Handle missing values & reindex ───────────────────────────
print("\n🔄 Handling missing values...")
print(f"   Null counts before reindex:\n{df.isnull().sum()}")

# Reindex to all business days and forward-fill gaps (market holidays)
all_business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
df = df.reindex(all_business_days)
df.ffill(inplace=True)

# Drop any remaining rows with nulls in critical columns
df.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)

# Fill remaining volume NaN with 0
df['volume'] = df['volume'].fillna(0)

print(f"   After reindex + ffill shape: {df.shape}")
print(f"   Remaining nulls: {df.isnull().sum().sum()}")

# ── Step 6: Feature Engineering ───────────────────────────────────────
print("\n🛠️ Engineering features...")

# Daily returns
df['daily_return'] = df['close'].pct_change()

# Log returns (better for statistical modelling)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Price range
df['price_range'] = df['high'] - df['low']

# Rolling volatility (annualized)
df['volatility_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(252)
df['volatility_90d'] = df['daily_return'].rolling(90).std() * np.sqrt(252)

# Decade label for period analysis
def get_decade(date):
    year = date.year
    if year < 2000: return '1990s'
    elif year < 2010: return '2000s'
    elif year < 2020: return '2010s'
    else: return '2020s'

df['decade'] = df.index.map(get_decade)

# Year and Month
df['year'] = df.index.year
df['month'] = df.index.month

# Drop first row (NaN from pct_change)
df.dropna(subset=['daily_return'], inplace=True)

print(f"   Engineered columns: daily_return, log_return, price_range, volatility_30d/90d, decade, year, month")

# ── Step 7: Merge RSI & PE Ratio from supplementary data ─────────────
print("\n📊 Merging RSI & PE Ratio data...")
try:
    rsi_df = pd.read_csv(RSI_PE_PATH, skiprows=[1])  # Skip the RELIANCE.NS sub-header row
    rsi_df['Date'] = pd.to_datetime(rsi_df['Date'])
    rsi_df.set_index('Date', inplace=True)
    rsi_df = rsi_df[['RSI', 'PE_Ratio']]
    rsi_df['RSI'] = pd.to_numeric(rsi_df['RSI'], errors='coerce')
    rsi_df['PE_Ratio'] = pd.to_numeric(rsi_df['PE_Ratio'], errors='coerce')

    # Merge on date index
    df = df.join(rsi_df, how='left')
    matched = df['RSI'].notna().sum()
    print(f"   Matched RSI values: {matched}")
    print(f"   Matched PE values: {df['PE_Ratio'].notna().sum()}")
except Exception as e:
    print(f"   ⚠️ Could not merge RSI/PE data: {e}")
    df['RSI'] = np.nan
    df['PE_Ratio'] = np.nan

# ── Step 8: Summary statistics ────────────────────────────────────────
print("\n" + "=" * 60)
print("  📋 CLEANED DATASET SUMMARY")
print("=" * 60)
print(f"  Shape:          {df.shape}")
print(f"  Date Range:     {df.index.min().date()} → {df.index.max().date()}")
print(f"  Total Days:     {len(df)}")
print(f"  Columns:        {df.columns.tolist()}")
print(f"\n  Price Stats:")
print(f"    Min Close:    ₹{df['close'].min():,.2f}")
print(f"    Max Close:    ₹{df['close'].max():,.2f}")
print(f"    Latest Close: ₹{df['close'].iloc[-1]:,.2f}")
print(f"\n  Volume Stats:")
print(f"    Avg Volume:   {df['volume'].mean():,.0f}")
print(f"    Max Volume:   {df['volume'].max():,.0f}")
print(f"\n  Return Stats:")
print(f"    Mean Daily:   {df['daily_return'].mean()*100:.4f}%")
print(f"    Std Daily:    {df['daily_return'].std()*100:.4f}%")
print(f"    Best Day:     {df['daily_return'].max()*100:.2f}%")
print(f"    Worst Day:    {df['daily_return'].min()*100:.2f}%")

# ── Step 9: Save ──────────────────────────────────────────────────────
df.to_csv(CLEAN_PATH)
print(f"\n✅ Cleaned dataset saved to: {CLEAN_PATH}")
print(f"   File size: {os.path.getsize(CLEAN_PATH) / 1024:.1f} KB")
