import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import plotly.graph_objects as go

st.set_page_config(page_title="多股票趨勢監控", layout="wide")

# ============================
# Telegram 推播（防重複）
# ============================
sent_alerts = set()

def send_telegram(text):
    if text in sent_alerts:
        return
    try:
        token = st.secrets["telegram_token"]
        chat_id = st.secrets["telegram_chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        sent_alerts.add(text)
    except:
        pass

# ============================
# SuperTrend（最穩版本）
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2

    # 完全避免任何 2D 問題
    tr0 = df['High'] - df['Low']
    tr1 = (df['High'] - df['Close'].shift()).abs()
    tr2 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)  # 這行安全
    atr = tr.rolling(period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df['UpperBand'] = upper
    df['LowerBand'] = lower
    df['SuperTrend'] = 0
    df['ST_Line'] = np.nan

    trend = 0
    for i in range(period, len(df)):
        close_i = df['Close'].iloc[i]
        upper_prev = upper.iloc[i-1]
        lower_prev = lower.iloc[i-1]

        if close_i > upper_prev:
            trend = 1
        elif close_i < lower_prev:
            trend = -1
        else:
            if trend == 1 and close_i < lower.iloc[i]:
                trend = -1
            elif trend == -1 and close_i > upper.iloc[i]:
                trend = 1

        df.iat[i, df.columns.get_loc('SuperTrend')] = trend
        df.iat[i, df.columns.get_loc('ST_Line')] = lower.iloc[i] if trend == 1 else upper.iloc[i]

    return df

# ============================
# 其他指標（全部改用 .values 避免 pandas 維度問題）
# ============================
def add_vwap(df):
    volume = df['Volume'].values
    close = df['Close'].values
    pv = close * volume
    cum_pv = pd.Series(pv, index=df.index).groupby(df.index.date).cumsum()
    cum_vol = pd.Series(volume, index=df.index).groupby(df.index.date).cumsum()
    df['VWAP'] = cum_pv / cum_vol
    return df

def add_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df):
    close = df['Close']
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

# 最安全寫法的 ADX（再也不會出錯）
def add_adx(df, period=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    tr0 = high - low
    tr1 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
    tr2 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    atr = pd.Series(tr, index=df.index).rolling(period).mean()
    df['ADX'] = atr.rolling(period).mean().fillna(0)
    return df

# ============================
# 美觀 10 根 K 線圖
# ============================
def plot_candlestick(df, symbol):
    df_plot = df.tail(10).copy()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name="K線",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))

    # SuperTrend 動態顏色
    color = 'green' if df_plot['SuperTrend'].iloc[-1] == 1 else 'red'
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['ST_Line'],
        mode='lines', name='SuperTrend',
        line=dict(width=4, color=color)
    ))

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', name='MA50', line=dict(color='purple')))

    fig.update_layout(
        title=f"{symbol}　最近 10 根 K 線",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================
# 主程式
# ============================
st.title("多股票趨勢監控 + 專業K線圖")

symbols = st.text_input("輸入股票（逗號分隔）", "AAPL,TSLA,NVDA,BTC-USD").upper().split(",")
timeframe = st.selectbox("時間框", ["1m","5m","15m","30m","1h","1d"])
period = st.selectbox("資料期間", ["1d","5d","30d","60d","1y"])

refresh_map = {"不刷新":0, "30秒":30, "1分鐘":60, "3分鐘":180, "5分鐘":300}
refresh_sec = refresh_map[st.selectbox("自動刷新", list(refresh_map.keys()))]

if refresh_sec > 0:
    st.write(f"每 {refresh_sec} 秒自動刷新")
    time.sleep(refresh_sec)
    st.rerun()

for symbol in [s.strip() for s in symbols if s.strip()]:
    st.subheader(f"{symbol}")

    try:
        df = yf.download(symbol, period=period, interval=timeframe, progress=False)
        if df.empty or len(df) < 50:
            st.warning("資料不足")
            continue

        df = df.dropna()
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df = supertrend(df)

        direction = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強" if df["ADX"].iloc[-1] > df["ADX"].mean() else "弱"
        st.markdown(f"**趨勢**：{direction}　**強度**：{strength}")

        plot_candlestick(df, symbol)

    except Exception as e:
        st.error(f"錯誤：{e}")

st.success("全部完成！")
