import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

st.set_page_config(page_title="多股票趨勢監控", layout="wide")

# ============================
# Telegram 推播
# ============================
def send_telegram(text):
    token = st.secrets["telegram_token"]
    chat_id = st.secrets["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass


# ============================
# SuperTrend
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2
    # 修正 ATR 計算：使用 True Range 的移動平均
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(period).mean()

    df['Upper'] = hl2 + multiplier * df['ATR']
    df['Lower'] = hl2 - multiplier * df['ATR']

    df['SuperTrend'] = np.nan
    trend = 1

    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Upper'].iloc[i - 1]:
            trend = 1
        elif df['Close'].iloc[i] < df['Lower'].iloc[i - 1]:
            trend = -1
        df['SuperTrend'].iloc[i] = trend

    return df


# ============================
# VWAP
# ============================
def add_vwap(df):
    # 修正 VWAP：按日期分組重置累計（避免跨日累積）
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df.groupby('Date')["PV"].cumsum()
    df["CumVol"] = df.groupby('Date')["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df


# ============================
# RSI
# ============================
def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ============================
# MACD
# ============================
def add_macd(df):
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df


# ============================
# ADX
# ============================
def add_adx(df, period=14):
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                np.maximum(abs(df['High'] - df['Close'].shift(1)),
                           abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where(df['High'] > df['High'].shift(1),
                         df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where(df['Low'] < df['Low'].shift(1),
                         df['Low'].shift(1) - df['Low'], 0)

    df['TR14'] = df['TR'].rolling(period).sum()
    df['+DM14'] = df['+DM'].rolling(period).sum()
    df['-DM14'] = df['-DM'].rolling(period).sum()

    df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
    df['-DI'] = 100 * (df['-DM14'] / df['TR14'])

    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(period).mean()
    return df

# ============================
# 趨勢方向 / 強度 / 持續性
# ============================
def analyze_trend(df):
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    adx = df["ADX"].iloc[-1]

    direction = "上升 📈" if macd > signal else "下降 📉"
    strength = "強" if adx > 25 else "弱"
    duration = "持續中" if df["Hist"].iloc[-1] > 0 else "變動中"

    return direction, strength, duration


# ============================
# 通知 1：MACD Hist 三連升 + ADX > 25
# ============================
def alert_macd_hist(df, symbol):
    hist = df["Hist"]
    adx = df["ADX"].iloc[-1]

    if len(hist) > 3 and hist.iloc[-3] < hist.iloc[-2] < hist.iloc[-1] and adx > 25:
        send_telegram(f"📢 {symbol}\nMACD Hist 連 3 上升 + ADX > 25\n可能強勢啟動")
        return True
    return False


# ============================
# 通知 2：RSI 背離
# ============================
def alert_rsi_div(df, symbol):
    price_low = df["Close"].iloc[-3:].idxmin()
    rsi_low = df["RSI"].iloc[-3:].idxmin()

    if price_low != rsi_low:
        send_telegram(f"📢 {symbol}\nRSI 背離偵測：價格破底但 RSI 未破\n可能反轉訊號")
        return True
    return False


# ============================
# 通知 3：SuperTrend 翻轉
# ============================
def alert_supertrend(df, symbol):
    if df["SuperTrend"].iloc[-2] != df["SuperTrend"].iloc[-1]:
        direction = "上漲轉強 🔵" if df["SuperTrend"].iloc[-1] == 1 else "轉弱 🔴"
        send_telegram(f"📢 {symbol}\nSuperTrend 翻轉：{direction}")
        return True
    return False


# ============================
# 通知 4：MA20/MA50
# ============================
def alert_ma_cross(df, symbol):
    ma20_prev, ma20_now = df["MA20"].iloc[-2], df["MA20"].iloc[-1]
    ma50_prev, ma50_now = df["MA50"].iloc[-2], df["MA50"].iloc[-1]

    if ma20_prev < ma50_prev and ma20_now > ma50_now:
        send_telegram(f"📢 {symbol}\nMA20/MA50 金叉（看多）")
        return True

    if ma20_prev > ma50_prev and ma20_now < ma50_now:
        send_telegram(f"📢 {symbol}\nMA20/MA50 死叉（看空）")
        return True

    return False


# ============================
# 通知 5：VWAP 突破
# ============================
def alert_vwap(df, symbol):
    close_prev, close_now = df["Close"].iloc[-2], df["Close"].iloc[-1]
    vwap_prev, vwap_now = df["VWAP"].iloc[-2], df["VWAP"].iloc[-1]

    if close_prev < vwap_prev and close_now > vwap_now:
        send_telegram(f"📢 {symbol}\nVWAP 上穿 → 看多突破")
        return True

    if close_prev > vwap_prev and close_now < vwap_now:
        send_telegram(f"📢 {symbol}\nVWAP 下穿 → 看空突破")
        return True

    return False


# ============================
# 通知 6：MACD 翻正 / 翻負 預警
# ============================
def alert_macd_predict(df, symbol):
    hist = df["Hist"]
    if hist.iloc[-1] > hist.iloc[-2] and hist.iloc[-2] > hist.iloc[-3]:
        send_telegram(f"📢 {symbol}\nMACD 可能即將翻正 → 預警")
        return True
    if hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]:
        send_telegram(f"📢 {symbol}\nMACD 可能即將翻負 → 預警")
        return True
    return False

# ============================
# Streamlit UI
# ============================
st.title("📈 多股票趨勢監控（含 Telegram 通知）")

symbols = st.text_input("輸入股票（逗號分隔）", "TSLA,AAPL,NVDA").upper().split(",")

timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m"])
period = st.selectbox("歷史區間", ["1d", "5d", "30d", "1y", "2y", "5y"])

# 自動刷新選擇
refresh_map = {
    "不刷新": 0,
    "30 秒": 30,
    "1 分鐘": 60,
    "5 分鐘": 300,
    "15 分鐘": 900,
}

refresh_choice = st.selectbox("自動刷新頻率", list(refresh_map.keys()))
refresh_sec = refresh_map[refresh_choice]

if refresh_sec > 0:
    st.write(f"⏳ 自動刷新：每 {refresh_sec} 秒")
    time.sleep(refresh_sec)
    st.rerun()


# ============================
# 主程式：下載 → 指標 → 趨勢 → 警報
# ============================
for symbol in symbols:
    symbol = symbol.strip()
    st.subheader(f"📌 {symbol}")

    df = yf.download(symbol, period=period, interval=timeframe)
    df.dropna(inplace=True)
    # 新增日期欄位，用於 VWAP 每日重置
    df['Date'] = df.index.date

    df = add_macd(df)
    df = add_rsi(df)
    df = add_adx(df)
    df = add_vwap(df)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df = supertrend(df)

    direction, strength, duration = analyze_trend(df)
    st.write(f"""
    👉 趨勢方向：**{direction}**  
    👉 趨勢強度：**{strength}**  
    👉 持續性：**{duration}**  
    """)

    st.line_chart(df[["Close", "MA20", "MA50"]])

    # ======================
    # 通知觸發（全部）
    # ======================
    alert_macd_hist(df, symbol)
    alert_rsi_div(df, symbol)
    alert_supertrend(df, symbol)
    alert_ma_cross(df, symbol)
    alert_vwap(df, symbol)
    alert_macd_predict(df, symbol)

st.success("完成刷新 ✓")
