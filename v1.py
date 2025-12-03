import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time

st.set_page_config(page_title="多股票趨勢監控", layout="wide")

# ============================
# Telegram 推播（加上簡單防重複）
# ============================
sent_alerts = set()   # 用來記錄已經發過的訊息（本次執行期間不會重複發）

def send_telegram(text):
    global sent_alerts
    if text in sent_alerts:
        return
    token = st.secrets["telegram_token"]
    chat_id = st.secrets["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
        sent_alerts.add(text)
    except:
        pass


# ============================
# SuperTrend（修正 ATR 與對齊問題）
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    hl2 = (df['High'] + df['Low']) / 2

    # 正確 True Range + ATR
    tr = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    atr = pd.Series(tr).rolling(period).mean()
    
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df['Upper'] = upper
    df['Lower'] = lower
    df['SuperTrend'] = np.nan

    trend = 0   # 0=未定, 1=多頭, -1=空頭
    for i in range(period, len(df)):
        # 基本翻轉條件
        if df['Close'].iloc[i] > df['Upper'].iloc[i-1]:
            trend = 1
        elif df['Close'].iloc[i] < df['Lower'].iloc[i-1]:
            trend = -1
        # 關鍵：趨勢鎖定（只有碰到反向帶才翻）
        else:
            if trend == 1 and df['Close'].iloc[i] < df['Lower'].iloc[i]:
                trend = -1
            elif trend == -1 and df['Close'].iloc[i] > df['Upper'].iloc[i]:
                trend = 1
        df.loc[df.index[i], 'SuperTrend'] = trend

    return df


# ============================
# VWAP（每日重置）
# ============================
def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df.groupby(df.index.date)["PV"].cumsum()
    df["CumVol"] = df.groupby(df.index.date)["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df


# ============================
# 其餘指標（完全保留原版，僅小調整避免 NaN）
# ============================
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
    tr = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    plus_dm = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                       np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    minus_dm = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                        np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    tr_smooth = pd.Series(tr).rolling(period).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / tr_smooth

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(period).mean()
    return df

# ============================
# 趨勢分析與所有警報函數（完全保留原版）
# ============================
def analyze_trend(df):
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 0

    direction = "上升" if macd > signal else "下降"
    strength = "強" if adx > 25 else "弱"
    duration = "持續中" if df["Hist"].iloc[-1] > 0 else "變動中"

    return direction, strength, duration

# 以下 6 個 alert 函數完全不變，只改小地方避免 index error
def alert_macd_hist(df, symbol):
    if len(df) < 4: return False
    hist = df["Hist"]
    if hist.iloc[-3] < hist.iloc[-2] < hist.iloc[-1] and df["ADX"].iloc[-1] > 25:
        send_telegram(f"{symbol}\nMACD Hist 連 3 上升 + ADX > 25\n可能強勢啟動")
        return True
    return False

def alert_rsi_div(df, symbol):
    if len(df) < 5: return False
    recent = df.iloc[-5:]
    if recent["Close"].idxmin() != recent["RSI"].idxmin():
        send_telegram(f"{symbol}\nRSI 背離偵測：價格破底但 RSI 未破\n可能反轉訊號")
        return True
    return False

def alert_supertrend(df, symbol):
    if len(df) < 2: return False
    if df["SuperTrend"].iloc[-2] != df["SuperTrend"].iloc[-1]:
        dir_text = "上漲轉強" if df["SuperTrend"].iloc[-1] == 1 else "轉弱"
        send_telegram(f"{symbol}\nSuperTrend 翻轉：{dir_text}")
        return True
    return False

def alert_ma_cross(df, symbol):
    if len(df) < 50: return False
    ma20_prev, ma20_now = df["MA20"].iloc[-2], df["MA20"].iloc[-1]
    ma50_prev, ma50_now = df["MA50"].iloc[-2], df["MA50"].iloc[-1]
    if ma20_prev <= ma50_prev and ma20_now > ma50_now:
        send_telegram(f"{symbol}\nMA20/MA50 金叉（看多）")
        return True
    if ma20_prev >= ma50_prev and ma20_now < ma50_now:
        send_telegram(f"{symbol}\nMA20/MA50 死叉（看空）")
        return True
    return False

def alert_vwap(df, symbol):
    if len(df) < 2: return False
    if df["Close"].iloc[-2] <= df["VWAP"].iloc[-2] < df["Close"].iloc[-1]:
        send_telegram(f"{symbol}\nVWAP 上穿 → 看多突破")
        return True
    if df["Close"].iloc[-2] >= df["VWAP"].iloc[-2] > df["Close"].iloc[-1]:
        send_telegram(f"{symbol}\nVWAP 下穿 → 看空突破")
        return True
    return False

def alert_macd_predict(df, symbol):
    if len(df) < 3: return False
    h = df["Hist"]
    if h.iloc[-1] > h.iloc[-2] > h.iloc[-3]:
        send_telegram(f"{symbol}\nMACD 可能即將翻正 → 預警")
        return True
    if h.iloc[-1] < h.iloc[-2] < h.iloc[-3]:
        send_telegram(f"{symbol}\nMACD 可能即將翻負 → 預警")
        return True
    return False


# ============================
# Streamlit UI
# ============================
st.title("多股票趨勢監控（含 Telegram 通知）")

symbols = st.text_input("輸入股票（逗號分隔）", "TSLA,AAPL,NVDA").upper().split(",")
timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"])
period = st.selectbox("歷史區間", ["1d", "5d", "30d", "60d", "1y", "2y", "5y"])

refresh_map = {"不刷新": 0, "30 秒": 30, "1 分鐘": 60, "5 分鐘": 300, "15 分鐘": 900}
refresh_choice = st.selectbox("自動刷新頻率", list(refresh_map.keys()))
refresh_sec = refresh_map[refresh_choice]

if refresh_sec > 0:
    st.write(f"自動刷新：每 {refresh_sec} 秒")
    time.sleep(refresh_sec)
    st.rerun()   # 最新寫法

# ============================
# 主程式
# ============================
for symbol in [s.strip() for s in symbols if s.strip()]:
    st.subheader(f"{symbol}")

    try:
        df = yf.download(symbol, period=period, interval=timeframe, progress=False)
        if df.empty:
            st.error(f"{symbol} 無資料")
            continue
        df = df.dropna()

        # 必要欄位
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df = supertrend(df)

        direction, strength, duration = analyze_trend(df)
        st.write(f"""
        趨勢方向：**{direction}**  
        趨勢強度：**{strength}**  
        持續性：**{duration}**  
        """)

        st.line_chart(df[["Close", "MA20", "MA50", "VWAP"]])

        # 警報
        alert_macd_hist(df, symbol)
        alert_rsi_div(df, symbol)
        alert_supertrend(df, symbol)
        alert_ma_cross(df, symbol)
        alert_vwap(df, symbol)
        alert_macd_predict(df, symbol)

    except Exception as e:
        st.error(f"{symbol} 發生錯誤：{e}")

st.success("全部完成 ✓")
