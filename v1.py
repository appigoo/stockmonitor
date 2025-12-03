import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import plotly.graph_objects as go

st.set_page_config(page_title="多股票趨勢監控", layout="wide")

# ============================
# Telegram 推播
# ============================
sent_alerts = set()
def send_telegram(text):
    if text in sent_alerts:
        return
    try:
        token = st.secrets["telegram_token"]
        chat_id = st.secrets["telegram_chat_id"]
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        sent_alerts.add(text)
    except Exception:
        # 保持 streamlit 不因推播錯誤中斷
        pass

# ============================
# 完全安全的 True Range
# ============================
def calculate_tr(df: pd.DataFrame) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    # 避免第一筆 NaN
    prev_close = close.shift(1).fillna(close.iloc[0])

    tr0 = high - low
    tr1 = (high - prev_close).abs()
    tr2 = (low - prev_close).abs()

    # 強制轉 ndarray，避免 shape 問題
    tr = np.maximum(tr0.values, np.maximum(tr1.values, tr2.values))
    return pd.Series(tr, index=df.index, name="TR")

# ============================
# SuperTrend
# ============================
def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    atr = calculate_tr(df).rolling(period).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df["UpperBand"] = upper
    df["LowerBand"] = lower
    df["SuperTrend"] = 0
    df["ST_Line"] = np.nan

    trend = 0
    # 若資料長度不足 period+1，直接返回，避免迴圈越界
    if len(df) <= period:
        return df

    for i in range(period, len(df)):
        close = df["Close"].iloc[i]
        prev_upper = upper.iloc[i - 1]
        prev_lower = lower.iloc[i - 1]

        # 若前一根的 band 為 NaN，則跳過這一根
        if pd.isna(prev_upper) or pd.isna(prev_lower):
            continue

        if close > prev_upper:
            trend = 1
        elif close < prev_lower:
            trend = -1
        else:
            # 目前這根的 band 也可能是 NaN，要判斷
            cur_upper = upper.iloc[i]
            cur_lower = lower.iloc[i]
            if trend == 1 and not pd.isna(cur_lower) and close < cur_lower:
                trend = -1
            elif trend == -1 and not pd.isna(cur_upper) and close > cur_upper:
                trend = 1

        df.iat[i, df.columns.get_loc("SuperTrend")] = trend
        if trend == 1:
            df.iat[i, df.columns.get_loc("ST_Line")] = lower.iloc[i]
        elif trend == -1:
            df.iat[i, df.columns.get_loc("ST_Line")] = upper.iloc[i]

    return df

# ============================
# 其他指標
# ============================
def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 確保 index 是 datetime，並用 normalize() 按天歸一
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    date_index = df.index.normalize()
    pv = df["Close"] * df["Volume"]
    cum_pv = pv.groupby(date_index).cumsum()
    cum_vol = df["Volume"].groupby(date_index).cumsum()
    df["VWAP"] = cum_pv / cum_vol
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    atr = calculate_tr(df).rolling(period).mean()
    df["ADX"] = atr.rolling(period).mean().fillna(0)
    return df

# ============================
# K 線圖
# ============================
def plot_candlestick(df: pd.DataFrame, symbol: str):
    # 保證有足夠資料畫 10 根
    df_plot = df.tail(10).copy()
    if df_plot.empty:
        st.warning(f"{symbol} 無足夠資料繪製 K 線")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="K線",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        )
    )

    # SuperTrend 顏色
    st_val = df_plot["SuperTrend"].iloc[-1]
    color = "green" if st_val == 1 else "red"

    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["ST_Line"],
            mode="lines",
            name="SuperTrend",
            line=dict(width=4, color=color),
        )
    )

    if "VWAP" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["VWAP"],
                mode="lines",
                name="VWAP",
                line=dict(color="orange", dash="dot"),
            )
        )
    if "MA20" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["MA20"],
                mode="lines",
                name="MA20",
                line=dict(color="blue"),
            )
        )
    if "MA50" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot["MA50"],
                mode="lines",
                name="MA50",
                line=dict(color="purple"),
            )
        )

    fig.update_layout(
        title=f"{symbol} 最近 10 根 K 線",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================
# 主程式
# ============================
st.title("多股票趨勢監控 + 專業K線圖")

symbols_input = st.text_input(
    "輸入股票（逗號分隔）", "AAPL,TSLA,NVDA,BTC-USD,0050.TW"
)
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

timeframe = st.selectbox("時間框", ["1m", "5m", "15m", "30m", "1h", "1d"])
period = st.selectbox("資料期間", ["1d", "5d", "30d", "60d", "1y"])

refresh_map = {"不刷新": 0, "30秒": 30, "1分鐘": 60, "3分鐘": 180, "5分鐘": 300}
refresh_sec = refresh_map[st.selectbox("自動刷新", list(refresh_map.keys()))]

if refresh_sec > 0:
    st.write(f"每 {refresh_sec} 秒自動刷新")
    time.sleep(refresh_sec)
    st.rerun()

for symbol in symbols:
    st.subheader(symbol)

    try:
        df = yf.download(
            symbol, period=period, interval=timeframe, progress=False, auto_adjust=False
        )

        # 檢查資料
        if df is None or df.empty:
            st.warning(f"{symbol} 無資料")
            continue

        # 有些情況 index 不是 DatetimeIndex，轉一下
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 先丟掉完全缺失的
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        if len(df) < 50:
            st.warning(f"{symbol} 資料不足（{len(df)} 筆）")
            continue

        # 指標計算
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        df = supertrend(df)
        # 再做一次 dropna，確保後面用到的欄位都完整
        df = df.dropna(subset=["MACD", "Signal", "ADX", "SuperTrend", "ST_Line"])

        if df.empty:
            st.warning(f"{symbol} 指標計算後無足夠資料")
            continue

        direction = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強" if df["ADX"].iloc[-1] > df["ADX"].mean() else "弱"
        st.markdown(f"**趨勢**：{direction}　**強度**：{strength}")

        plot_candlestick(df, symbol)

    except Exception as e:
        st.error(f"{symbol} 發生錯誤：{e}")

st.success("全部完成！穩定強化版")
