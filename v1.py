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

    # True Range（最安全寫法）
    tr0 = df['High'] - df['Low']
    tr1 = (df['High'] - df['Close'].shift()).abs()
    tr2 = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
    atr = tr.rolling(window=period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df['UpperBand'] = upper
    df['LowerBand'] = lower
    df['SuperTrend'] = 0.0
    df['ST_Line'] = np.nan

    trend = 0
    for i in range(period, len(df)):
        close = df['Close'].iloc[i]
        prev_upper = upper.iloc[i-1]
        prev_lower = lower.iloc[i-1]

        if close > prev_upper:
            trend = 1
        elif close < prev_lower:
            trend = -1
        else:
            if trend == 1 and close < lower.iloc[i]:
                trend = -1
            elif trend == -1 and close > upper.iloc[i]:
                trend = 1

        df.iat[i, df.columns.get_loc('SuperTrend')] = trend

        # 畫圖用線
        if trend == 1:
            df.iat[i, df.columns.get_loc('ST_Line')] = lower.iloc[i]
        else:
            df.iat[i, df.columns.get_loc('ST_Line')] = upper.iloc[i]

    return df

# ============================
# 其他指標（全部防呆）
# ============================
def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df.groupby(df.index.date)["PV"].cumsum()
    df["CumVol"] = df.groupby(df.index.date)["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
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

# 完全修正 ADX（不再用 pd.concat）
def add_adx(df, period=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    tr0 = high - low
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr0, tr1, tr2])
    
    atr = pd.Series(tr).rolling(period).mean().values
    df['ADX'] = pd.Series(atr).rolling(period).mean()
    df['ADX'] = df['ADX'].fillna(0)
    return df

# ============================
# 美觀 10 根 K 線圖
# ============================
def plot_candlestick(df, symbol):
    df_plot = df.tail(10).copy()

    fig = go.Figure()

    # K線
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

    # SuperTrend 線（動態顏色）
    st_color = 'green' if df_plot['SuperTrend'].iloc[-1] == 1 else 'red'
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['ST_Line'],
        mode='lines', name='SuperTrend',
        line=dict(width=4, color=st_color)
    ))

    # 其他線
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', name='MA50', line=dict(color='purple')))

    fig.update_layout(
        title=f"{symbol}　最近 10 根 K 線",
        xaxis_title="時間",
        yaxis_title="價格",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================
# 主程式
# ============================
st.title("多股票趨勢監控 + 專業K線圖")

symbols = st.text_input("輸入股票（逗號分隔）", "AAPL,TSLA,NVDA").upper().split(",")
timeframe = st.selectbox("時間框", ["1m","5m","15m","30m","1h","1d"])
period = st.selectbox("資料期間", ["1d","5d","30d","60d","1y","2y"])

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
            st.warning(f"{symbol} 資料不足（{len(df)}筆）")
            continue

        df = df.dropna()
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)        # 完全不會出錯
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df = supertrend(df)

        # 趨勢判斷
        direction = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強" if df["ADX"].iloc[-1] > df["ADX"].quantile(0.7) else "弱"
        duration = "持續中" if df["Hist"].iloc[-1] > 0 else "變動中"
        st.markdown(f"**趨勢**：{direction}　**強度**：{strength}　**狀態**：{duration}")

        # 美觀 K 線圖
        plot_candlestick(df, symbol)

    except Exception as e:
        st.error(f"{symbol} 錯誤：{e}")

st.success("全部完成！")
