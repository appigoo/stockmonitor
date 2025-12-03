import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import plotly.graph_objects as go   # 新增這行

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
# SuperTrend（穩定版）
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2
    tr0 = df['High'] - df['Low']
    tr1 = abs(df['High'] - df['Close'].shift(1))
    tr2 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df['Upper'] = upper
    df['Lower'] = lower
    df['SuperTrend'] = 0.0

    trend = 0
    for i in range(period, len(df)):
        close = df['Close'].iloc[i]
        if close > upper.iloc[i-1]:
            trend = 1
        elif close < lower.iloc[i-1]:
            trend = -1
        else:
            if trend == 1 and close < lower.iloc[i]:
                trend = -1
            if trend == -1 and close > upper.iloc[i]:
                trend = 1
        df.iat[i, df.columns.get_loc('SuperTrend')] = trend

    # 產生 SuperTrend 線（多頭綠、空頭紅）
    df['ST_Line'] = np.nan
    df.loc[df['SuperTrend'] == 1, 'ST_Line'] = lower[df['SuperTrend'] == 1]
    df.loc[df['SuperTrend'] == -1, 'ST_Line'] = upper[df['SuperTrend'] == -1]

    return df

# ============================
# 其餘指標（保持不變）
# ============================
def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df.groupby(df.index.date)["PV"].cumsum()
    df["CumVol"] = df.groupby(df.index.date)["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df):
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df

def add_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    df['ADX'] = (atr / close).rolling(period).mean() * 10000
    return df

# ============================
# 警報函數（略，同上版）
# ============================
def analyze_trend(df):
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    direction = "上升" if macd > signal else "下降"
    strength = "強" if df["ADX"].iloc[-1] > 25 else "弱"
    duration = "持續中" if df["Hist"].iloc[-1] > 0 else "變動中"
    return direction, strength, duration

# （以下 6 個 alert 函數完全相同，省略以節省篇幅，複製前一版的即可）

# ============================
# 美觀 K 線圖函數（重點！）
# ============================
def plot_candlestick(df, symbol):
    df_plot = df.tail(10).copy()  # 只取最近 10 根

    fig = go.Figure()

    # 蠟燭圖
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name="K線",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ))

    # SuperTrend 線
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['ST_Line'],
        mode='lines+markers',
        name='SuperTrend',
        line=dict(width=3, color=['green' if x==1 else 'red' for x in df_plot['SuperTrend']])
    ))

    # VWAP
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', dash='dot')))

    # MA20 & MA50
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', name='MA50', line=dict(color='purple')))

    fig.update_layout(
        title=f"{symbol} 最近 10 根 K 線",
        xaxis_title="時間",
        yaxis_title="價格",
        template="plotly_dark",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')

    st.plotly_chart(fig, use_container_width=True)

# ============================
# 主程式
# ============================
st.title("多股票趨勢監控 + 專業K線圖")

symbols = st.text_input("輸入股票（逗號分隔）", "AAPL,TSLA,NVDA").upper().split(",")
timeframe = st.selectbox("時間框", ["1m","5m","15m","30m","1h","1d"])
period = st.selectbox("資料期間", ["1d","5d","30d","60d","1y"])

refresh_map = {"不刷新":0, "30秒":30, "1分鐘":60, "3分鐘":180, "5分鐘":300}
refresh_sec = refresh_map[st.selectbox("自動刷新", list(refresh_map.keys()), index=2)]

if refresh_sec > 0:
    st.write(f"每 {refresh_sec} 秒自動刷新")
    time.sleep(refresh_sec)
    st.rerun()

for symbol in [s.strip() for s in symbols if s.strip()]:
    st.subheader(f"{symbol}")

    try:
        df = yf.download(symbol, period=period, interval=timeframe, progress=False)
        if df.empty or len(df) < 30:
            st.warning("資料不足")
            continue

        df = df.dropna()
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close["Close"].rolling(50).mean()
        df = supertrend(df)

        direction, strength, duration = analyze_trend(df)
        st.markdown(f"**趨勢**：{direction}　**強度**：{strength}　**狀態**：{duration}")

        # 這裡就是重點：美觀的最近10根K線圖
        plot_candlestick(df, symbol)

        # 警報（保留原功能）
        # alert_xxx(df, symbol) ...

    except Exception as e:
        st.error(f"錯誤：{e}")

st.success("全部完成")
